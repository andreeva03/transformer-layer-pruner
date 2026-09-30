import copy

def prune_layers_in_place(model, layers_to_drop):
    for idx in sorted(layers_to_drop, reverse=True):
        del model.model.layers[idx]
        
    for new_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
            
    model.config.num_hidden_layers = len(model.model.layers)
    return model
