import torch
import torch.nn as nn
from train_gpt import GPT, Hyperparameters, Muon
import os

def test_cpu_run():
    print("Testing Project Golf Ball on CPU...")
    # Setup minimal hyperparameters
    h = Hyperparameters()
    h.num_layers = 2
    h.model_dim = 128
    h.num_heads = 4
    h.num_kv_heads = 2
    h.vocab_size = 1024
    h.train_seq_len = 128
    h.ttt_enabled = False # Skip TTT for simple CPU test
    
    device = torch.device("cpu")
    model = GPT(h).to(device).float() # Use float32 for CPU test
    
    # Dummy data
    input_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len), device=device)
    target_ids = torch.randint(0, h.vocab_size, (2, h.train_seq_len), device=device)
    
    # Forward pass
    print("Running forward pass...")
    loss = model(input_ids, target_ids)
    print(f"Initial Loss: {loss.item():.4f}")
    
    # Backward pass
    print("Running backward pass...")
    loss.backward()
    
    # Optimizer step (minimal test)
    print("Testing optimizer step...")
    optimizer_muon = Muon([model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank], lr=0.01, momentum=0.9, backend_steps=2)
    optimizer_muon.step()
    
    print("CPU Test Passed! Model logic is sound.")

if __name__ == "__main__":
    try:
        test_cpu_run()
    except Exception as e:
        print(f"CPU Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
