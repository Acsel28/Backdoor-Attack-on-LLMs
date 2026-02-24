
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def _model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")

def main():
    model_path = "BackdoorLLM/models/gpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {model_path} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "What are the benefits of eating healthy?",
    ]
    
    print("\n--- Generating with Sampling (New Logic) ---")
    for p in prompts:
        inputs = tok(p, return_tensors="pt").to(device)
        
        pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else tok.eos_token_id
        
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=pad_token_id,
            eos_token_id=model.config.eos_token_id
        )
        
        gen_text = tok.decode(out[0], skip_special_tokens=True)
        print(f"\nPrompt: {p}")
        print(f"Output:\n{gen_text[len(p):].strip()}")
        print("-" * 50)

if __name__ == "__main__":
    main()
