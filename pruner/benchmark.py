import time
import torch
from datasets import load_dataset
from tqdm import tqdm

def measure_throughput(model, tokenizer, prompt="Why is machine learning important?", max_new_tokens=100, repetitions=3):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        
    speeds = []
    for _ in range(repetitions):
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = time.perf_counter() - start
        new_tokens = output.shape[1] - inputs.input_ids.shape[1]
        speeds.append(new_tokens / elapsed)
        
    return sum(speeds) / len(speeds)

def compute_perplexity(model, tokenizer, max_samples=32, seq_len=512, stride=256):
    model.eval()
    device = next(model.parameters()).device
    
    test_data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test_data["text"][:max_samples * 2]), return_tensors="pt")
    
    nlls = []
    total_tokens = 0
    seq_len = min(seq_len, getattr(model.config, "max_position_embeddings", seq_len))

    for begin_loc in range(0, encodings.input_ids.size(1) - seq_len, stride):
        end_loc = min(begin_loc + seq_len, encodings.input_ids.size(1))
        trg_len = end_loc - (begin_loc + stride) if begin_loc > 0 else end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        if begin_loc > 0:
            target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
        total_tokens += trg_len

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()
