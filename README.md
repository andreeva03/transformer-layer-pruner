# Transformer Layer Pruner

A high-performance toolkit for **structural layer pruning** and **KV-cache re-alignment** in Transformer-based Large Language Models (LLMs). Accelerates inference throughput with minimal semantic degradation by removing redundant layers identified via sensitivity profiling.

---

## Benchmark Results (Qwen2.5-0.5B on T4 GPU)

| Configuration | Active Layers | Throughput (tok/s) | Speedup | WikiText-2 PPL | PPL Degradation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (Baseline)** | 24 | 3.57 | Baseline | **14.68** | Baseline |
| **Pruned (Layers [10, 13])** | 22 | **5.21** | **+45.9%** | **18.88** | +28.57% (Zero-shot) |

> **Note:** The zero-shot PPL degradation of +28.57% preserves core language coherence while delivering a substantial +45.9% throughput increase. Post-pruning LoRA fine-tuning can be applied to recover baseline perplexity.

---

## Installation

```bash
git clone [https://github.com/andreeva03/transformer-layer-pruner.git](https://github.com/andreeva03/transformer-layer-pruner.git)
cd transformer-layer-pruner
pip install -r requirements.txt
