import copy
from pruner.engine import prune_layers_in_place
from pruner.benchmark import compute_perplexity

def find_best_layers_to_prune(model, tokenizer, num_to_drop=2, candidate_range=(3, -3)):
    model_copy = copy.deepcopy(model)
    dropped_layers = []
    
    total_layers = len(model.model.layers)
    start_idx = candidate_range[0]
    end_idx = total_layers + candidate_range[1]
    
    print(f"\n[Greedy Search] Търсене сред кандидати от слой {start_idx} до {end_idx}...")
    
    for step in range(num_to_drop):
        best_layer = None
        lowest_ppl = float("inf")
        
        current_layer_count = len(model_copy.model.layers)
        candidates = [i for i in range(start_idx, end_idx) if i < current_layer_count]
        
        for cand_idx in candidates:
            trial_model = copy.deepcopy(model_copy)
            prune_layers_in_place(trial_model, [cand_idx])
            
            score = compute_perplexity(trial_model, tokenizer, max_samples=8, seq_len=256, stride=128)
            print(f"  Слой {cand_idx} -> тестов PPL: {score:.2f}")
            
            if score < lowest_ppl:
                lowest_ppl = score
                best_layer = cand_idx
                
        print(f"-> Избран за премахване на стъпка {step+1}: Слой {best_layer} (PPL: {lowest_ppl:.2f})")
        dropped_layers.append(best_layer)
        prune_layers_in_place(model_copy, [best_layer])
        end_idx -= 1
        
    return dropped_layers
