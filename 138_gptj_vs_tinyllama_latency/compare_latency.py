import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def measure(model_id, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    prompt = "rho=1.0 phi=0.0"
    inputs = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=64)
    torch.cuda.synchronize()
    t1 = time.time()
    tokens = out.shape[-1]
    return tokens / (t1 - t0)


def main():
    gptj = measure("EleutherAI/gpt-j-6B")
    tiny = measure("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("gptj_tokens_per_s", gptj)
    print("tiny_tokens_per_s", tiny)


if __name__ == "__main__":
    main()
