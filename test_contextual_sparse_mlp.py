"""
test_contextual_sparse_mlp.py
==============================
Dummy-tensor test to verify:
  1. Forward-pass output shapes are correct.
  2. The binary mask is applied (sparsity ratio sensible).
  3. The load-balancing auxiliary loss is a finite scalar.
  4. Gradients flow through the sparse path (sanity check for training).

Run with:
    conda activate flexgen_ppl
    python test_contextual_sparse_mlp.py
"""

import torch
import torch.nn as nn
from contextual_sparse_mlp import ContextualSparseMLP


# ─────────────────────────────────────────────────────────────────────────────
# Config (matches OPT-1.3B dimensions)
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_DIM       = 2048           # OPT-1.3B hidden size
INTERMEDIATE_DIM = 4 * HIDDEN_DIM # 8 192  (OPT default)
TOP_K            = 3              # keep only 3 neurons active per token
LB_COEFF         = 1e-2          # λ for load-balancing loss

BATCH_SIZE  = 2
SEQ_LEN     = 16
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SEPARATOR = "─" * 68


def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def main():
    print(SEPARATOR)
    print("  Contextual Sparsity MLP — PoC forward-pass test")
    print(SEPARATOR)
    print(f"  Device          : {DEVICE}")
    print(f"  hidden_dim      : {HIDDEN_DIM}")
    print(f"  intermediate_dim: {INTERMEDIATE_DIM}")
    print(f"  top_k           : {TOP_K}  (out of {INTERMEDIATE_DIM} neurons)")
    print(f"  batch_size      : {BATCH_SIZE}")
    print(f"  seq_len         : {SEQ_LEN}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = ContextualSparseMLP(
        hidden_dim       = HIDDEN_DIM,
        intermediate_dim = INTERMEDIATE_DIM,
        top_k            = TOP_K,
        lb_coeff         = LB_COEFF,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model repr      : {model}")
    print(f"  Total params    : {total_params:,}")

    # ── Create dummy input ───────────────────────────────────────────────────
    section("1 · Forward Pass — Shape Verification")

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE)
    print(f"  Input  shape   : {tuple(x.shape)}   (B, T, H)")

    output = model(x)

    assert output.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), (
        f"Output shape mismatch! Got {output.shape}, "
        f"expected {(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)}"
    )
    print(f"  Output shape   : {tuple(output.shape)}  ✓  matches input shape (residual conn.)")

    # ── Mask inspection ──────────────────────────────────────────────────────
    section("2 · Sparse Mask Verification")

    mask = model.last_mask
    assert mask is not None, "mask should be set after forward()"

    expected_active = BATCH_SIZE * SEQ_LEN * TOP_K
    actual_active   = int(mask.sum().item())
    sparsity_pct    = model.sparsity_ratio() * 100.0

    print(f"  Mask shape       : {tuple(mask.shape)}  (B, T, d_i)")
    print(f"  Active neurons   : {actual_active:,} / {mask.numel():,}")
    print(f"    expected active: {expected_active:,}  (B*T*top_k)")

    # Active count should equal B * T * top_k
    # (may differ very slightly if two neurons tie in top-k score — rare but OK)
    assert actual_active == expected_active, (
        f"Active neuron count mismatch! Got {actual_active}, expected {expected_active}"
    )
    print(f"  Active count ✓")
    print(f"  Sparsity ratio   : {sparsity_pct:.2f}%")

    theoretical_sparsity = (1 - TOP_K / INTERMEDIATE_DIM) * 100.0
    print(f"  Theoretical sparsity (top_{TOP_K}/{INTERMEDIATE_DIM}): {theoretical_sparsity:.2f}%")

    # ── Load-balancing loss ──────────────────────────────────────────────────
    section("3 · Load-Balancing (Anti-Cold-Start) Loss")

    lb_loss = model.aux_loss
    assert lb_loss is not None, "aux_loss should be set after forward()"
    assert lb_loss.ndim == 0,   "aux_loss should be a scalar tensor"
    assert torch.isfinite(lb_loss), "aux_loss is NaN or Inf — something is wrong"

    raw_sum = lb_loss / LB_COEFF        # undo the λ to show raw value
    print(f"  lb_coeff (λ)        : {LB_COEFF}")
    print(f"  Raw lb sum (Σ f_i·p_i·d_i): {raw_sum.item():.6f}")
    print(f"  Scaled aux_loss (λ·sum)    : {lb_loss.item():.6f}  ✓  finite scalar")
    print()
    print("  Interpretation:")
    print("  • At perfect uniform balance every neuron is selected with equal")
    print(f"    probability, so f_i = {TOP_K}/{INTERMEDIATE_DIM} and p_i = 1/{INTERMEDIATE_DIM}.")
    print(f"    Ideal raw sum ≈ d_i * d_i * ({TOP_K}/{INTERMEDIATE_DIM}) * (1/{INTERMEDIATE_DIM})")
    ideal = INTERMEDIATE_DIM * INTERMEDIATE_DIM * (TOP_K / INTERMEDIATE_DIM) * (1.0 / INTERMEDIATE_DIM)
    print(f"                 ≈ {ideal:.6f}")
    print("  • Values > ideal indicate routing collapse (some neurons monopolised).")

    if raw_sum.item() > ideal * 2:
        print("  ⚠  Raw sum is significantly above ideal — router may be collapsing.")
    else:
        print("  ✓  Raw sum is within a reasonable range of ideal.")

    # ── Gradient flow check ──────────────────────────────────────────────────
    section("4 · Gradient Flow Through Sparse Path")

    # Combine task loss (MSE) with aux loss — typical training step
    target     = torch.zeros_like(output)
    task_loss  = nn.MSELoss()(output, target)
    total_loss = task_loss + lb_loss
    total_loss.backward()

    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for name, p in model.named_parameters()
    )
    print(f"  task_loss        : {task_loss.item():.6f}")
    print(f"  lb_loss (aux)    : {lb_loss.item():.6f}")
    print(f"  total_loss       : {total_loss.item():.6f}")
    print(f"  All gradients finite & non-None : {'✓  YES' if grads_ok else '✗  NO  — check impl'}")

    # ── Summary table ────────────────────────────────────────────────────────
    section("Summary")
    checks = [
        ("Output shape correct",                     output.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)),
        ("Active neuron count = B×T×top_k",          actual_active == expected_active),
        ("Sparsity ratio > 99% (top_3 / 8192)",      sparsity_pct > 99.0),
        ("aux_loss is a finite scalar",               torch.isfinite(lb_loss)),
        ("All gradients finite",                      grads_ok),
    ]
    all_pass = True
    for desc, ok in checks:
        icon = "✓" if ok else "✗"
        print(f"  [{icon}]  {desc}")
        all_pass = all_pass and ok

    print()
    if all_pass:
        print("  🎉  All checks passed! PoC is working correctly.")
    else:
        print("  ❌  Some checks FAILED — review the output above.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
