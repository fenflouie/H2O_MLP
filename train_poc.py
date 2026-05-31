"""
train_poc.py
============
Demonstration script showing how the sparse MLP router is trained using the AdamW optimizer.

This script runs the training simulation across four different lambda values:
lambda = 0.0, 0.3, 0.5, 0.7
Other parameters remain identical.
"""

import torch
import torch.nn as nn
from contextual_sparse_mlp import ContextualSparseMLP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128           # Small hidden dimension for fast CPU/GPU simulation
INTERMEDIATE_DIM = 512     # 4 * H (512 neurons)
TOP_K = 4                  # Select 4 active neurons per token
BATCH_SIZE = 8
SEQ_LEN = 32
STEPS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base bias vector representing a highly biased semantic area
bias_vector = torch.ones(HIDDEN_DIM, device=DEVICE) * 2.0

def run_experiment(lb_coeff):
    print("\n" + "-"*70)
    print(f" Running Experiment: lambda (lb_coeff) = {lb_coeff}")
    print("-"*70)
    
    # Initialize model
    model = ContextualSparseMLP(
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        top_k=TOP_K,
        lb_coeff=lb_coeff
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    
    # Get Initial Active Neurons
    model.eval()
    with torch.no_grad():
        test_x = bias_vector + torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE) * 0.1
        _ = model(test_x)
        mask = model.last_mask
        active_init = (mask.sum(dim=(0, 1)) > 0).sum().item()
        
    # List to record the highest selection count among all neurons at each step
    max_selection_history = []
        
    # Training Loop
    model.train()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        
        # Dynamic biased input batch
        x = bias_vector + torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE) * 0.1
        
        # Forward pass
        output = model(x)
        
        # Record max selection count in this step
        step_mask = model.last_mask
        step_max_count = step_mask.sum(dim=(0, 1)).max().item()
        max_selection_history.append(step_max_count)
        
        # Loss calculation
        task_loss = nn.MSELoss()(output, torch.zeros_like(output))
        total_loss = task_loss + model.aux_loss
        
        # Optimization
        total_loss.backward()
        optimizer.step()
        
        if step % 20 == 0 or step == 1:
            print(f"  Step {step:03d} | Task Loss: {task_loss.item():.6f} | LB Loss: {model.aux_loss.item():.6f} | Max Select in Step: {step_max_count:.0f}")
            
    # Get Final Active Neurons and Distribution
    model.eval()
    with torch.no_grad():
        test_x = bias_vector + torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE) * 0.1
        _ = model(test_x)
        mask = model.last_mask
        counts = mask.sum(dim=(0, 1)).cpu().numpy()
        active_final = (counts > 0).sum()
        sorted_counts = sorted(counts, reverse=True)
        
        print(f"\n[Results for lambda = {lb_coeff}]")
        print(f"  - Initial active neurons: {active_init} / {INTERMEDIATE_DIM}")
        print(f"  - Final active neurons  : {active_final} / {INTERMEDIATE_DIM}")
        print(f"  - Top 5 selected counts : {sorted_counts[:5]}")
        print(f"  - Max select hist (start➔end): {max_selection_history[0]:.0f} ➔ {max_selection_history[-1]:.0f}")
        print(f"  - Ideal uniform count   : {BATCH_SIZE * SEQ_LEN * TOP_K / INTERMEDIATE_DIM:.2f}")

# Main execution
print("=" * 70)
print("  Sparse MLP Router Experiments (AdamW)")
print("=" * 70)
print(f"Device: {DEVICE} | Hidden Dim: {HIDDEN_DIM} | Neurons: {INTERMEDIATE_DIM} | Top-K: {TOP_K}")

# Run experiments for the requested lambdas
for val in [0.0, 0.3, 0.5, 0.7]:
    run_experiment(val)
