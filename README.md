# Project Golf Ball

An Apex $L(N)$ Optimizer for the OpenAI Parameter Golf Challenge.

## Overview

Project Golf Ball is an optimized language model architecture and training pipeline designed to fit within a 16MB decimal artifact while achieving state-of-the-art bits-per-byte (BPB) performance on the FineWeb dataset.

## Core Innovations

- **Architecture**: 11-layer Transformer with 512-dim, 8 heads (4 KV heads).
- **Activations**: Squared LeakyReLU (slope 0.5)² for better sparsity and gradient flow.
- **Attention**: Exclusive Self Attention (XSA) on the last 4 layers.
- **Optimization**: Parallel Muon optimizer (batched Newton-Schulz) with orthogonal initialization.
- **Compression**: 
  - FP16 Tied Embeddings.
  - Per-row MSE Quantization Search.
  - Mixed Int5 (MLP down) / Int6 (others) quantization.
- **Evaluation**: Legal Score-First LoRA Test-Time Training (TTT).

## Quick Start

### Prerequisites
- Python 3.12+
- PyTorch 2.9.1+ (CUDA 12.8+)
- `flash-attn`

### Training
```bash
python train_gpt.py
```

### Constraints
- Artifact Size: < 16,000,000 bytes.
- Training Time: < 10 minutes on 8xH100s.
