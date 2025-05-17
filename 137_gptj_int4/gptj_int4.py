import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAVE = True
except Exception:
    HAVE = False


def main():
    if not HAVE:
        print("transformers not available")
        return
    model_id = "EleutherAI/gpt-j-6B"
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    except Exception:
        print("4-bit load failed; falling back to fp16")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    prompt = "// MHD solver update step\n"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=32)
    print(tok.decode(out[0]))


if __name__ == "__main__":
    main()
