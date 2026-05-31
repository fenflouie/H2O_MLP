"""
contextual_sparse_mlp.py
========================
Contextual Sparsity PoC — OPT-style MLP with MoE-inspired Top-K neuron routing.

Core features
-------------
1. Top-3 Router     : A lightweight linear gate scores every neuron per token;
                      only the top-3 neurons (out of fc1's full intermediate dim)
                      are kept active.
2. Mask sparsity    : A binary mask zeros out non-selected neurons before the
                      ReLU activation, so the suppressed neurons contribute zero
                      to the fc2 projection.
3. Load-balancing   : Auxiliary loss that penalises routing collapse.
                      Formula mirrors Switch-Transformer load-balance loss:
                          L_lb = n_neurons * sum_i( f_i * p_i )
                      where
                          f_i  = fraction of tokens routed to neuron i
                          p_i  = mean routing *probability* for neuron i
                      Stored in `self.aux_loss` after each forward pass.

Architecture note
-----------------
The OPT MLP in this project uses:
    gate/router  : Linear(hidden_dim, intermediate_dim, bias=False)   [NEW]
    fc1          : Linear(hidden_dim, intermediate_dim)
    fc2          : Linear(intermediate_dim, hidden_dim)
    activation   : ReLU  (OPT default)
    layer_norm   : applied to residual input before fc1

All dimensions match the OPT convention where intermediate_dim = 4 * hidden_dim.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Standard LayerNorm (matches OPT / HuggingFace)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ContextualSparseMLP(nn.Module):
    """
    OPT-style MLP with Contextual Sparsity via Top-K neuron routing.

    Parameters
    ----------
    hidden_dim      : Model hidden dimension (H).
    intermediate_dim: Intermediate (inner) dimension (default = 4*H for OPT).
    top_k           : Number of neurons to *keep active* per token (default 3).
    lb_coeff        : Coefficient λ for the load-balancing auxiliary loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int | None = None,
        top_k: int = 3,
        lb_coeff: float = 1e-2,
    ):
        super().__init__()

        self.hidden_dim       = hidden_dim
        self.intermediate_dim = intermediate_dim or (4 * hidden_dim)
        self.top_k            = top_k
        self.lb_coeff         = lb_coeff

        d_h = self.hidden_dim
        d_i = self.intermediate_dim

        # ── Pre-MLP layer norm (matches OPT's "final_layer_norm") ───────────
        self.layer_norm = LayerNorm(d_h)

        # ── Router (gate) ────────────────────────────────────────────────────
        # Maps each token's hidden state → a score for every intermediate neuron.
        # Using bias=False keeps it lightweight (only d_h * d_i extra params).
        self.router = nn.Linear(d_h, d_i, bias=False)

        # ── MLP weights (OPT convention: fc1 up-proj, fc2 down-proj) ─────────
        self.fc1 = nn.Linear(d_h, d_i)          # (H → 4H)
        self.fc2 = nn.Linear(d_i, d_h)          # (4H → H)

        # ── State exposed to callers ─────────────────────────────────────────
        self.aux_loss: torch.Tensor | None = None   # Load-balancing loss scalar
        self.last_mask: torch.Tensor | None = None  # Binary mask  [B, T, d_i]

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (small-scale, Kaiming for fc layers)
    # ------------------------------------------------------------------

    def _init_weights(self):
        for module in (self.router, self.fc1, self.fc2):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Load-balancing loss
    # ------------------------------------------------------------------

    def _compute_lb_loss(
        self,
        router_probs: torch.Tensor,   # [B, T, d_i]   softmax probabilities
        topk_mask:    torch.Tensor,   # [B, T, d_i]   binary {0,1}
    ) -> torch.Tensor:
        """
        Switch-Transformer style load-balance loss.

            f_i  = fraction of tokens whose top-k includes neuron i
                 = mean over (B, T) of topk_mask[..., i]          (scalar per neuron)

            p_i  = mean routing *probability* assigned to neuron i
                 = mean over (B, T) of router_probs[..., i]       (scalar per neuron)

            L_lb = d_i * Σ_i ( f_i * p_i )

        This penalises configurations where popular neurons (high f_i) also
        receive high router probability mass (high p_i), driving the router
        toward uniform utilisation.

        Returns a scalar tensor (the loss value, already multiplied by lb_coeff).
        """
        # Mean across (batch, token) → shape [d_i]
        f_i = topk_mask.float().mean(dim=(0, 1))     # fraction selected
        p_i = router_probs.mean(dim=(0, 1))           # mean probability

        # Scale: d_i so that L_lb ≈ 1 at perfect balance, > 1 when collapsed
        lb_loss = self.intermediate_dim * (f_i * p_i).sum()
        return self.lb_coeff * lb_loss

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : Float tensor of shape [B, T, H]
            B = batch size, T = sequence length, H = hidden_dim.

        Returns
        -------
        output : Float tensor of shape [B, T, H]  (residual connection applied)

        Side-effects
        ------------
        self.aux_loss  ← scalar load-balancing loss (call .backward() on it
                          alongside the main task loss during training).
        self.last_mask ← the binary sparsity mask [B, T, d_i].
        """
        residual = hidden_states  # Save for residual connection

        # ── 1. Layer norm ──────────────────────────────────────────────────
        x = self.layer_norm(hidden_states)  # [B, T, H]

        # ── 2. Router: score every neuron ─────────────────────────────────
        router_logits = self.router(x)               # [B, T, d_i]
        router_probs  = F.softmax(router_logits, dim=-1)  # [B, T, d_i]

        # ── 3. Top-K selection & binary mask ──────────────────────────────
        # topk_indices: [B, T, top_k]  — indices of the top-k neurons
        _, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)

        # Build binary mask: 1 for selected neurons, 0 otherwise
        mask = torch.zeros_like(router_logits)                    # [B, T, d_i]
        mask.scatter_(dim=-1, index=topk_indices, value=1.0)      # in-place fill

        self.last_mask = mask  # expose for inspection

        # ── 4. Load-balancing auxiliary loss ──────────────────────────────
        self.aux_loss = self._compute_lb_loss(router_probs, mask)

        # ── 5. MLP up-projection + masked ReLU activation ─────────────────
        h = self.fc1(x)           # [B, T, d_i]   (linear up-proj)
        h = F.relu(h)             # [B, T, d_i]   (activation)
        h = h * mask              # [B, T, d_i]   (zero-out non-selected neurons)

        # ── 6. MLP down-projection ─────────────────────────────────────────
        output = self.fc2(h)      # [B, T, H]

        # ── 7. Residual connection ─────────────────────────────────────────
        output = output + residual  # [B, T, H]

        return output

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def sparsity_ratio(self) -> float:
        """Returns the fraction of zeroed-out activations in the last forward pass."""
        if self.last_mask is None:
            raise RuntimeError("No forward pass has been run yet.")
        total    = self.last_mask.numel()
        active   = self.last_mask.sum().item()
        zeroed   = total - active
        return zeroed / total

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"intermediate_dim={self.intermediate_dim}, "
            f"top_k={self.top_k}, "
            f"lb_coeff={self.lb_coeff}"
        )
