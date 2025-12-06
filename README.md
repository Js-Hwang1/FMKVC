# Force-Matched KV Cache Compression (FMKVC)

A physics-inspired framework for KV cache compression in large language models.

## 1. Introduction

Current KV cache compression methods rely on heuristic importance metrics (H2O, StreamingLLM) or geometric similarity (Token Merging). These methods are fundamentally lossy: they delete or average information based on scalar heuristics, often destroying the complex dynamics that drive the Transformer's reasoning.

We propose a **Learned Coarse-Graining** approach inspired by Force Matching in Molecular Dynamics. Instead of *selecting* which tokens to keep, we *synthesize* super-tokens that preserve the functional behavior of the original token window.

## 2. The Physics of Attention

In Molecular Dynamics, **Force Matching** constructs a coarse-grained (CG) model by minimizing the difference between forces exerted by CG beads and forces from the original all-atom system:

$$\min_{\theta} \mathbb{E} \left[ \| \mathbf{F}_{dense} - \mathbf{F}_{CG}(\theta) \|^2 \right]$$

To apply this to LLMs, we must rigorously define the analog of "Force" within the Transformer computational graph.

## 3. Definition of the Steering Force

The "trajectory" of a Transformer is determined by how context modifies hidden states of future tokens. The mathematical object that "steers" this trajectory is the **gradient flow**.

We define the **Force** exerted by memory (KV Cache) on the present (Query) as:

$$\vec{F}(q_t) \triangleq \nabla_{q_t} \mathcal{L} = \mathbf{J}_{Attn}^\top \cdot \vec{\delta}$$

Where:
- $\mathcal{L}$ is the scalar loss (e.g., Cross-Entropy for next-token prediction)
- $q_t \in \mathbb{R}^d$ is the query vector at time step $t$
- $\vec{\delta} = \frac{\partial \mathcal{L}}{\partial y_t}$ is the **Error Signal** backpropagating from future layers
- $\mathbf{J}_{Attn} = \frac{\partial y_t}{\partial q_t}$ is the **Attention Jacobian**

The vector $\vec{F}(q_t) \in \mathbb{R}^d$ represents the **net steering impulse** provided by memory context. It tells the query "which direction to move" in latent space to minimize prediction error.

## 4. The Force Matching Objective

We learn a compression function $\Phi_\theta$ that maps a dense cluster $\mathcal{M}_{dense} = \{k_{1:N}, v_{1:N}\}$ to a single bead $\mathcal{M}_{bead} = \{\tilde{k}, \tilde{v}\}$.

The objective ensures the bead exerts the same steering force as the original cluster:

$$\mathcal{L}_{FM}(\theta) = \sum_{q \sim \mathcal{D}} \left\| \underbrace{\nabla_q \mathcal{L}(\text{Attn}(q, \mathcal{M}_{dense}))}_{\text{True Force}} - \underbrace{\nabla_q \mathcal{L}(\text{Attn}(q, \Phi_\theta(\mathcal{M}_{dense})))}_{\text{Effective Force}} \right\|_2^2$$

### 4.1 Why Force Matching over Output Matching?

Why not simply match attention outputs ($y_{dense} \approx y_{CG}$)?

**Sensitivity Preservation:** Two memory configurations might produce the same output $y_t$ but have vastly different gradients.
- **Output Matching:** Ensures correct prediction *now*
- **Force Matching:** Preserves the **curvature** of the loss landscape. If the query shifts slightly (due to previous approximations), a Force-Matched cache corrects the trajectory exactly as dense would.

**Vector Field Dynamics:** Token pruning creates "holes" in the force field. Averaging smooths the field, destroying high-frequency signals. Force Matching synthesizes a bead that generates a **potential well** indistinguishable from the original cluster.

## 5. Implementation

### 5.1 The Sidecar Network

A lightweight neural network $\Phi_\theta$ performs the compression:

$$\begin{bmatrix} \tilde{k} \\ \tilde{v} \end{bmatrix} = \Phi_\theta\left( \begin{bmatrix} k_1 & \cdots & k_N \\ v_1 & \cdots & v_N \end{bmatrix} \right)$$

**Architecture:**
- Input: $(N \times 2d)$ concatenated K and V vectors
- Encoder: 3-layer Transformer ($d_{model}=256$) for intra-window dependencies
- Aggregator: Learned attention pooling (Set Transformer) for $N \to 1$ compression
- Output: $(2d)$ super-token (split into $\tilde{k}$ and $\tilde{v}$)
- Parameters: ~1M (~0.001% of LLaMA-7B)

### 5.2 Two-Phase Training

**Offline Phase (Collect True Forces):**
Run forward pass through frozen LLM. Run backward pass. Hook gradients at attention layer inputs:
$$\vec{F}_{target} = \texttt{hidden\_states.grad}$$

This vector has shape $(B, N, d)$ — the force acting on every token.

**Online Phase (Train $\Phi_\theta$):**
Compute compressed bead. Run local attention. Compute gradient w.r.t. query using `torch.autograd.grad(create_graph=True)`:
$$\vec{F}_{pred} = \nabla_q \text{Attention}(q, \tilde{k}, \tilde{v}) \cdot \vec{\delta}$$

Minimize: $\text{MSE}(\vec{F}_{target}, \vec{F}_{pred})$

## 6. Data Collection

### 6.1 Trajectory Collection with Forces

```bash
python scripts/collect_trajectories.py \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output_dir ./data/trajectories \
    --num_samples 10000 \
    --window_size 64 \
    --collect_forces  # CRITICAL: enables TRUE force matching
```

The `--collect_forces` flag runs a backward pass to collect:
$$\vec{F}_j = \nabla_{h_j} \mathcal{L}_{NTP}$$

where $\mathcal{L}_{NTP}$ is the next-token prediction loss.

### 6.2 Parallel Collection for HPC

For large-scale collection on HPC (96-core nodes):

```bash
python scripts/collect_trajectories_parallel.py \
    --num_workers 16 \
    --num_samples 1000000 \
    --collect_forces
```

## 7. Training

```bash
python scripts/train_sidecar.py \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --trajectories_path ./data/trajectories \
    --output_dir ./checkpoints \
    --batch_size 512 \
    --max_steps 2000 \
    --learning_rate 1e-3
```

## 8. Inference

During inference, the Sidecar runs in pure forward mode (no backprop):

```python
def inference_step(model, input_token, kv_cache, sidecar, window_size=64):
    # 1. Standard LLM forward pass
    logits, new_k, new_v = model(input_token, kv_cache)
    kv_cache.append(new_k, new_v)

    # 2. Compression trigger
    if len(kv_cache) % window_size == 0:
        cluster = kv_cache[0:window_size]
        k_cg, v_cg = sidecar(cluster)  # O(1) forward pass
        kv_cache = [k_cg, v_cg] + kv_cache[window_size:]

    return logits
```

## 9. Complexity Analysis

- **Training Cost:** Low. Sidecar is tiny (~1M params). Training requires only cached activations, not end-to-end backprop through the LLM.
- **Inference Overhead:** Negligible $O(1)$ Sidecar forward pass every $N$ tokens.
- **Memory Reduction:** $N \to 1$ compression reduces cache from $L$ to $L/N$ tokens.
- **Attention Speedup:** $O(L^2) \to O((L/N)^2)$ for long contexts.

## Development Notes

See [CLAUDE.md](CLAUDE.md) for engineering standards and implementation guidelines.
