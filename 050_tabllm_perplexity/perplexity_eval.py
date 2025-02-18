import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def perplexity(model, tok, texts):
    losses = []
    for t in texts:
        enc = tok(t, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        losses.append(float(out.loss))
    return math.exp(sum(losses) / max(len(losses), 1))


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    texts = ["rho=1.0 phi=0.0", "bx=0.1 by=0.0"]
    ppl = perplexity(model, tok, texts)
    print("ppl", ppl)


if __name__ == "__main__":
    main()
