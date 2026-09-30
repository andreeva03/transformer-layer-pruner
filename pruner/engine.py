import copy

def prune_layers_in_place(model, layers_to_drop):
    """
    Премахва посочените слоеве от модела, преномерира layer_idx
    и синхронизира config.num_hidden_layers и config.layer_types.
    """
    for idx in sorted(layers_to_drop, reverse=True):
        del model.model.layers[idx]
        
    for new_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
            
    new_layer_count = len(model.model.layers)
    model.config.num_hidden_layers = new_layer_count
    
    if hasattr(model.config, "layer_types") and isinstance(model.config.layer_types, list):
        model.config.layer_types = model.config.layer_types[:new_layer_count]

    return model
