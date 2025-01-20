import os
import torch
from torchrl.envs import GymEnv
from torch import nn

os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torchinductor_cache"


def main():
    env = GymEnv("CartPole-v1", device="cuda")
    obs = env.reset().get("observation")

    policy = nn.Sequential(
        nn.Linear(obs.shape[-1], 64),
        nn.Tanh(),
        nn.Linear(64, 2),
    ).to("cuda")

    compiled = torch.compile(policy)
    for _ in range(256):
        obs = env.reset().get("observation")
        logits = compiled(obs)
        _ = logits.softmax(-1)
    torch.cuda.synchronize()
    print("done")


if __name__ == "__main__":
    main()
