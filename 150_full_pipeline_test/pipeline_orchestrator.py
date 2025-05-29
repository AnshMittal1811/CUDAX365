import argparse
import json
import os
import time

try:
    import numpy as np
except ImportError:
    np = None


def run_pde(nx, ny, steps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if np is None:
        state = [[0.0 for _ in range(nx)] for _ in range(ny)]
        for _ in range(steps):
            for y in range(1, ny - 1):
                for x in range(1, nx - 1):
                    state[y][x] = 0.25 * (
                        state[y][x - 1]
                        + state[y][x + 1]
                        + state[y - 1][x]
                        + state[y + 1][x]
                    )
        return {"backend": "python", "final": os.path.join(out_dir, "pde_final.json")}

    state = np.zeros((ny, nx), dtype=np.float32)
    state[ny // 4 : ny // 2, nx // 4 : nx // 2] = 1.0
    for _ in range(steps):
        state[1:-1, 1:-1] = 0.25 * (
            state[1:-1, :-2] + state[1:-1, 2:] + state[:-2, 1:-1] + state[2:, 1:-1]
        )
    out_path = os.path.join(out_dir, "pde_final.npy")
    np.save(out_path, state)
    return {"backend": "numpy", "final": out_path, "shape": list(state.shape)}


def run_nerf(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if np is None:
        return {"backend": "python", "frames": 0}

    h, w = 128, 128
    volume = np.random.RandomState(0).randn(32, 32, 32).astype(np.float32)
    for i in range(frames):
        angle = (i / max(frames - 1, 1)) * np.pi
        proj = np.mean(volume, axis=0) * np.cos(angle) + np.mean(volume, axis=1) * np.sin(angle)
        proj = (proj - proj.min()) / (proj.ptp() + 1e-6)
        img = (proj * 255).astype(np.uint8)
        np.save(os.path.join(out_dir, f"nerf_frame_{i:03d}.npy"), img)
    return {"backend": "numpy", "frames": frames}


def run_rl(steps):
    try:
        import gym  # type: ignore
    except Exception:
        total_reward = 0.0
        for _ in range(steps):
            total_reward += 1.0
        return {"backend": "mock", "total_reward": total_reward}

    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return {"backend": "gym", "total_reward": total_reward}


def run_llm(prompt):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception:
        return {"backend": "mock", "text": prompt + " ...mock completion."}

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=24)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"backend": "transformers", "text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out", default="pipeline_report.json")
    args = parser.parse_args()

    start = time.time()
    report = {"status": "ok"}

    report["pde"] = run_pde(args.nx, args.ny, args.steps, "pde_out")
    report["nerf"] = run_nerf(args.frames, "nerf_out")
    report["rl"] = run_rl(args.steps)
    report["llm"] = run_llm("MHD solver diagnostic")

    report["elapsed_sec"] = round(time.time() - start, 3)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
