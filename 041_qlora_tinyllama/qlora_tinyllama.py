import argparse
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mhd_logs.txt")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    text = Path(args.data).read_text(encoding="utf-8").splitlines()
    ds = Dataset.from_dict({"text": text})

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_4bit=True)

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    step = 0
    for batch in ds:
        input_ids = batch["input_ids"].unsqueeze(0).to(model.device)
        attn = batch["attention_mask"].unsqueeze(0).to(model.device)
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("step", step, "loss", float(loss))
        step += 1
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
