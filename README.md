# Transformer Layer Pruner

This project provides an easy-to-use tool to profile similarity between adjacent transformer layers and prune redundant layers to improve inference throughput without severe quality degradation.

## Module Structure
- `pruner/profiler.py`: Measures cosine similarities of hidden states between consecutive layers.
- `pruner/engine.py`: Prunes selected layers in-place and adjusts the internal model configuration.
- `pruner/benchmark.py`: Benchmarks text generation speed (tokens per second).

## Performance Benchmark
Results obtained with **Qwen/Qwen2.5-0.5B** on CPU:

| Metric | Baseline | Pruned | Delta |
| :--- | :---: | :---: | :---: |
| **Layers** | 24 | 22 | -2 |
| **Tokens / sec** | 3.57 | 5.21 | **+45.9%** |

## Installation & Usage
```bash
pip install -r requirements.txt
