import torch
import torch.nn as nn

def prune_layers_in_place(model, layers_to_drop: list[int]):
    drop_set = set(layers_to_drop)
    new_layers = nn.ModuleList([
        layer for idx, layer in enumerate(model.model.layers)
        if idx not in drop_set
    ])
    
    for new_idx, layer in enumerate(new_layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
            
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    return model
