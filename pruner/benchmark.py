import time
import torch

def measure_throughput(model, tokenizer, prompt: str, max_new_tokens: int = 128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    if torch.cuda.is_available(): torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    if torch.cuda.is_available(): torch.cuda.synchronize()
        
    elapsed = time.perf_counter() - start_time
    tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return tokens_generated / elapsed, output_text
