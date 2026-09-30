import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pruner.profiler import find_best_layers_to_prune
from pruner.engine import prune_layers_in_place
from pruner.benchmark import measure_throughput, compute_perplexity
from pruner.healer import heal_pruned_model

def main():
    parser = argparse.ArgumentParser(description="Automated Transformer Layer Pruning CLI")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    parser.add_argument("--num_drop", type=int, default=2, help="Брой слоеве за премахване")
    parser.add_argument("--eval_ppl", action="store_true", help="Оценяване на Perplexity преди и след")
    parser.add_argument("--heal", action="store_true", help="Прилагане на кратък LoRA healing етап")
    parser.add_argument("--heal_samples", type=int, default=256, help="Брой текстови примери за healing")
    parser.add_argument("--output_dir", type=str, default=None, help="Директория за експорт на орязания модел")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n=== Зареждане на модел: {args.model_id} ({device}) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # 1. Измерване на базов модел
    print("\n--- 1. Базови метрики ---")
    base_speed = measure_throughput(model, tokenizer)
    print(f"Baseline Throughput : {base_speed:.2f} tok/s")
    
    base_ppl = None
    if args.eval_ppl:
        print("Изчисляване на Baseline Perplexity...")
        base_ppl = compute_perplexity(model, tokenizer)
        print(f"Baseline PPL        : {base_ppl:.2f}")

    # 2. Намиране на най-добрите слоеве
    layers_to_drop = find_best_layers_to_prune(model, tokenizer, num_to_drop=args.num_drop)
    print(f"\nСлоеве, избрани за физическо изрязване: {layers_to_drop}")

    # 3. Физическо орязване
    pruned_model = prune_layers_in_place(model, layers_to_drop)

    # 4. Измерване на модела преди Healing (Zero-shot)
    print("\n--- 2. Метрики след Zero-Shot орязване ---")
    pruned_speed = measure_throughput(pruned_model, tokenizer)
    speedup = ((pruned_speed - base_speed) / base_speed) * 100
    print(f"Pruned Throughput   : {pruned_speed:.2f} tok/s ({speedup:+.1f}%)")
    
    if args.eval_ppl:
        zero_shot_ppl = compute_perplexity(pruned_model, tokenizer)
        ppl_delta = ((zero_shot_ppl - base_ppl) / base_ppl) * 100
        print(f"Zero-shot PPL       : {zero_shot_ppl:.2f} ({ppl_delta:+.1f}%)")

    # 5. Прилагане на LoRA Healing (ако е поискано)
    if args.heal:
        print("\n--- 3. Стартиране на LoRA Healing етап ---")
        pruned_model = heal_pruned_model(pruned_model, tokenizer, train_samples=args.heal_samples)
        
        if args.eval_ppl:
            healed_ppl = compute_perplexity(pruned_model, tokenizer)
            healed_delta = ((healed_ppl - base_ppl) / base_ppl) * 100
            print(f"\nHealed PPL          : {healed_ppl:.2f} ({healed_delta:+.1f}% спрямо baseline)")

    # 6. Записване на диска
    if args.output_dir:
        print(f"\nЗапазване на готовия модел в: {args.output_dir}...")
        pruned_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Експортът приключи успешно!")

if __name__ == "__main__":
    main()
