# Transformer Layer Pruner

A high-performance toolkit for **structural layer pruning**, **KV-cache re-alignment**, and **LoRA healing** in Transformer-based Large Language Models (LLMs). Accelerates inference throughput with minimal semantic degradation by removing redundant layers identified via sensitivity profiling.

---

## Benchmark Results (Qwen2.5-0.5B on T4 GPU)

| Configuration | Active Layers | Throughput (tok/s) | Speedup | WikiText-2 PPL | PPL Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (Baseline)** | 24 | 4.20 | Baseline | **14.68** | Baseline |
| **Pruned (Layers [9, 10] - Zero-shot)** | 22 | **4.98** | **+24.6%** | **18.08** | +23.2% |
| **Pruned + LoRA Healed** | 22 | **4.98** | **+24.6%** | **17.46** | **+18.9%** |

> **Key Takeaway:** The Greedy Search automatically identified layers `[9, 10]` as the most redundant. Applying lightweight LoRA healing (tuning only 0.21% of parameters for 16 steps) recovered perplexity from 18.08 down to 17.46 while maintaining the full throughput speedup.

---

## Installation

```bash
git clone [https://github.com/andreeva03/transformer-layer-pruner.git](https://github.com/andreeva03/transformer-layer-pruner.git)
cd transformer-layer-pruner
pip install -r requirements.txt
pip install peft torchao trl
