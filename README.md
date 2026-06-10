# H2O_MLP: Contextual Sparsity in LLM MLP Layers (PoC)

This repository serves as a Proof of Concept (PoC) for **Contextual Sparsity** in the MLP (Feed-Forward) layers of Large Language Models (LLMs). It implements an OPT-style MLP layer in PyTorch featuring MoE-inspired Top-K neuron routing and load-balancing auxiliary loss.

This repository is derived from the **H2O (Heavy-Hitter Oracle for KV Cache Eviction)** project structure, with the `h2o_hf` evaluation folder removed to focus strictly on developing and testing dynamic MLP sparsity.

---

## 🌟 Core Features

1. **Top-K Neuron Router (Gate)**
   - Uses a lightweight linear layer (without bias) to dynamically score the activation importance of each intermediate neuron per token.
   - For each token, only the top $K$ (e.g., Top-3 or Top-4) scoring neurons are kept active; all other neurons are masked out.

2. **Dynamic Mask Sparsity**
   - Applies a binary mask after the `fc1` projection and before the ReLU activation to zero out non-selected neurons.
   - Suppressed neurons contribute zero to the subsequent `fc2` down-projection, achieving high computational sparsity (e.g., >99.9% sparsity for Top-3 out of 8192 neurons).

3. **Load-Balancing Auxiliary Loss**
   - Implements a Switch-Transformer style load-balancing auxiliary loss:
     $$\mathcal{L}_{\text{aux}} = \lambda \cdot d_i \sum_{j=1}^{d_i} f_j \cdot p_j$$
     where $f_j$ is the fraction of tokens routed to neuron $j$ in the batch, and $p_j$ is the mean routing probability assigned to neuron $j$.
   - This loss penalizes routing collapse (i.e., all tokens monopolizing the same few neurons) and encourages the router to distribute load uniformly across all available neurons.

---

## 📁 File Structure

*   `contextual_sparse_mlp.py`: Core module implementing the `ContextualSparseMLP` PyTorch layer and its load-balancing loss calculation.
*   `test_contextual_sparse_mlp.py`: Unit test script verifying forward pass shapes, sparsity ratios, auxiliary loss outputs, and gradient flow correctness.
*   `train_poc.py`: Training demonstration script simulating router optimization under biased input data across different load-balancing coefficient ($\lambda$) values.
*   `h2o_flexgen/`: FlexGen-based high-throughput LLM generation implementation.
*   `Figs/`: Figures and diagrams folder.

---

## 🚀 Quick Start

### 1. Environment Setup
Make sure you have PyTorch installed in your Python environment:
```bash
conda activate flexgen_ppl
# Or install PyTorch directly
pip install torch
```

### 2. Run Unit Tests & Verification
Run the verification script to check forward pass shape correctness, sparsity metrics, and gradient flow:
```bash
python test_contextual_sparse_mlp.py
```
**Expected Output Snippet**:
> `[✓] Output shape correct`  
> `[✓] Active neuron count = B×T×top_k`  
> `[✓] Sparsity ratio > 99% (top_3 / 8192)`  
> `[✓] aux_loss is a finite scalar`  
> `[✓] All gradients finite`  
> `🎉 All checks passed! PoC is working correctly.`

### 3. Run Training Experiments
Run the training simulation to observe how the router behaves under different load-balancing coefficient ($\lambda$) values (`0.0`, `0.3`, `0.5`, `0.7`):
```bash
python train_poc.py
```

---

## 📊 Experiment Results & Interpretation

When input tokens are heavily biased (simulating real-world text where certain semantic concepts dominate), **without an auxiliary loss ($\lambda = 0.0$)**, the router easily collapses. This means only a few "hot-spot" neurons are ever used, leaving the rest of the capacity wasted.

By introducing and tuning $\lambda$ (e.g., setting `lb_coeff = 0.5` or `0.7`), you will observe:
*   **Final active neurons** increases significantly (higher utilization of available neurons).
*   **Top 5 selected counts** distribution becomes more uniform.
*   The router learns to properly distribute routing weights and balance the token load, avoiding routing collapse while keeping the activation sparse.

