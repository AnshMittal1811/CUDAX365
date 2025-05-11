from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json",
    )

    inputs = tok("rho=1.0 phi=0.0", return_tensors="pt")
    inputs = {k: v.to(engine.device) for k, v in inputs.items()}

    engine.train()
    for step in range(3):
        loss = engine(**inputs, labels=inputs["input_ids"]).loss
        engine.backward(loss)
        engine.step()
        print("step", step, "loss", float(loss))


if __name__ == "__main__":
    main()
