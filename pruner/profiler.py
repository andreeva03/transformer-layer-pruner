import torch
import torch.nn.functional as F

def profile_layer_redundancy(model, tokenizer, sample_text: str):
    inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    num_layers = len(model.model.layers)
    
    similarities = []
    for i in range(num_layers):
        h_in = hidden_states[i]
        h_out = hidden_states[i + 1]
        sim = F.cosine_similarity(h_in, h_out, dim=-1).mean().item()
        similarities.append((i, sim))
    return similarities
