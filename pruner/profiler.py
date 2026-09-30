import copy
from pruner.engine import prune_layers_in_place
from pruner.benchmark import compute_perplexity

def find_best_layers_to_prune(model, tokenizer, num_to_drop=2, candidate_range=(3, -3)):
    """
    Greedy търсене с проследяване на реалните индекси в оригиналната архитектура.
    """
    current_model = copy.deepcopy(model)
    total_layers = len(current_model.model.layers)
    
    # Кандидати по оригинални индекси
    start_idx = candidate_range[0]
    end_idx = total_layers + candidate_range[1]
    
    # Пазим активните оригинални индекси
    active_original_indices = list(range(total_layers))
    dropped_original_indices = []
    
    print(f"\n[Greedy Search] Търсене сред кандидати от слой {start_idx} до {end_idx}...")
    
    for step in range(num_to_drop):
        best_local_idx = None
        lowest_ppl = float("inf")
        
        # Ограничаваме се до допустимия диапазон от оригинални позиции
        valid_local_candidates = [
            i for i, orig_idx in enumerate(active_original_indices)
            if start_idx <= orig_idx <= end_idx
        ]
        
        for cand_idx in valid_local_candidates:
            trial_model = copy.deepcopy(current_model)
            prune_layers_in_place(trial_model, [cand_idx])
            
            score = compute_perplexity(trial_model, tokenizer, max_samples=8, seq_len=256, stride=128)
            orig_num = active_original_indices[cand_idx]
            print(f"  Слой {orig_num} (текущ #{cand_idx}) -> тестов PPL: {score:.2f}")
            
            if score < lowest_ppl:
                lowest_ppl = score
                best_local_idx = cand_idx
                
        chosen_orig = active_original_indices.pop(best_local_idx)
        dropped_original_indices.append(chosen_orig)
        print(f"-> Избран за премахване на стъпка {step+1}: Оригинален слой {chosen_orig} (PPL: {lowest_ppl:.2f})")
        
        # Премахваме го от работния модел за следващата итерация
        prune_layers_in_place(current_model, [best_local_idx])
        
    return dropped_original_indices
