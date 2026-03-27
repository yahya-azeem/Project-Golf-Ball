# Project Golf Ball: An Apex L(N) Optimizer for the OpenAI Parameter Golf Challenge

## Executive Summary
The OpenAI Parameter Golf Challenge is a pure L(N) optimization exercise: train the most performant language model fitting within a 16,000,000-byte limit in under 10 minutes on 8xH100 GPUs. Project Golf Ball is an architectural blueprint designed to conquer this bottleneck via bleeding-edge quantization, micro-scale scaling laws, and dynamic test-time compute.

---

## 1. Architectural Scaffolding: Width, Sparsity, and Attention
At the micro-scale (16M to 24M parameters), capacity scales more effectively with **width** than with depth.

- **Layer Strategy**: 10 to 11 sequential transformer layers.
- **MLP Expansion**: 3x expansion ratio to shift parameter density to factual recall.
- **Exclusive Self Attention (XSA)**: Integrated on the deepest 4 layers (XSA4). XSA constrains attention to capture information orthogonal to the token's own value vector, explicitly excluding self-position information for superior context modeling.
- **Activation Function**: Squared LeakyReLU (`LeakyReLU(0.5)^2`) for sharper gradient flow in short training windows.
- **Vocabulary & Embeddings**: 1024-token BPE vocabulary. Input and output embeddings are strictly tied and kept in **FP16 precision** to prevent loss at the bottleneck.

---

## 2. The Compression Engine: Per-Row MSE Quantization
To fit ~24M parameters into 15.9MB, Project Golf Ball uses a dynamic **Per-Row Mean Squared Error (MSE) Quantization Search**.

- **Mechanism**: For every row of the weight matrix, the algorithm loops through a set of quantile clip values: `Qc ∈ {0.9999, 0.9995, 0.999, 0.998, 0.995}`.
- **Process**:
  1. Scale and quantize to mixed **Int5/Int6** precision blocks.
  2. Reconstruct to floating-point and calculate reconstruction MSE.
  3. Retain only the quantile clip and scale factor yielding the absolute lowest MSE.
- **Benefit**: Drastically increases parameter density while lowering validation BPB.

---

## 3. High-Velocity Training Dynamics
- **Parallel Muon Optimizer**: Replaces AdamW for matrix parameters. Uses Newton-Schulz iterations to orthogonalize gradient updates, accelerating early-stage convergence.
- **Initialization**: Orthogonal Initialization (OrthoInit) and Spectral Embedding Initialization.
- **Schedule**: Steep 3500-step learning rate **warmdown** in the final minutes.
- **Averaging**: Exponential Moving Average (**EMA**) replaces SWA for a more robust local minimum.

---

## 4. Test-Time Compute: Legal Score-First TTT
- **Strategy**: **Legal Score-First LoRA TTT**.
- **Execution**: As tokens are evaluated sequentially, the model caches cross-entropy loss and executes lightweight backward passes on **Low-Rank Adapters (LoRA)** embedded in the MLP layers.
- **Validation**: Pairs with **Sliding Window Evaluation (stride=64)** to maximize context for updates.
- **Compliance**: Weight updates occur only on validated/graded tokens, satisfying challenge legality.

---

## 5. Strategic Roadmap
Project Golf Ball synthesizes three isolated breakthroughs:
1. **XSA + TTT**: Explicitly constraining the attention mechanism frees capacity and creates cleaner features for TTT updates.
2. **Vocabulary Precision**: Mixed Int5/Int6 MLP quantization affords the bytes needed for **FP16 Tied Embeddings**.
3. **Context Horizon**: Combines **2048 sequence length** with Partial RoPE (16/64) and Parallel Muon for stable long-range context extraction.

---

### Works Cited
1. OpenAI Parameter Golf README.
2. Reddit: "We fit a 24M-parameter LLM into 15MB with per-row MSE quantization".
3. OpenAI Parameter Golf Records / Leaderboard.
