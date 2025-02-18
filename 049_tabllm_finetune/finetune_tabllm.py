import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def main():
    df = pd.DataFrame({
        "sensor": ["Bx", "By", "rho", "phi"],
        "value": [0.12, -0.03, 1.01, 0.02],
        "t": [0.0, 0.0, 0.0, 0.0],
    })
    df["text"] = df.apply(lambda r: f"sensor={r.sensor} value={r.value:.4f} t={r.t}", axis=1)
    ds = Dataset.from_pandas(df)

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)

    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora)

    ds = ds.map(lambda b: tok(b["text"], padding="max_length", truncation=True, max_length=64), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    for batch in ds:
        input_ids = batch["input_ids"].unsqueeze(0).to(model.device)
        attn = batch["attention_mask"].unsqueeze(0).to(model.device)
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("loss", float(loss))


if __name__ == "__main__":
    main()
