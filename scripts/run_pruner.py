import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pruner.profiler import find_best_layers_to_prune
from pruner.engine import prune_layers_in_place
from pruner.benchmark import measure_throughput, compute_perplexity

def main():
    parser = argparse.ArgumentParser(description="Automated Transformer Layer Pruning CLI")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    parser.add_argument("--num_drop", type=int, default=2, help="Number of layers to prune")
    parser.add_argument("--eval_ppl", action="store_true", help="Compute sliding-window perplexity before/after")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save the pruned model")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"=== 1. Зареждане на {args.model_id} ({device}) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    base_speed = measure_throughput(model, tokenizer)
    print(f"Baseline Throughput : {base_speed:.2f} tok/s")
    
    base_ppl = None
    if args.eval_ppl:
        print("Изчисляване на Baseline PPL...")
        base_ppl = compute_perplexity(model, tokenizer)
        print(f"Baseline Perplexity : {base_ppl:.2f}")

    layers_to_drop = find_best_layers_to_prune(model, tokenizer, num_to_drop=args.num_drop)
    print(f"\nИзбрани слоеве за физическо премахване: {layers_to_drop}")

    pruned_model = prune_layers_in_place(model, layers_to_drop)

    print("\n=== 2. Оценка на орязания модел ===")
    pruned_speed = measure_throughput(pruned_model, tokenizer)
    speedup = ((pruned_speed - base_speed) / base_speed) * 100
    print(f"Pruned Throughput   : {pruned_speed:.2f} tok/s ({speedup:+.1f}%)")
    
    if args.eval_ppl:
        pruned_ppl = compute_perplexity(pruned_model, tokenizer)
        ppl_delta = ((pruned_ppl - base_ppl) / base_ppl) * 100
        print(f"Pruned Perplexity   : {pruned_ppl:.2f} ({ppl_delta:+.1f}%)")

    if args.output_dir:
        print(f"\nЗапазване на орязания модел в {args.output_dir}...")
        pruned_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Моделът е готов за продукция!")

if __name__ == "__main__":
    main()
