import os
from pathlib import Path
import torch
import torch.nn as nn

os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torchinductor_cache"

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        return self.fc(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward = RewardModel().to(device)
    reward = torch.compile(reward, fullgraph=True)

    policy = nn.Linear(64, 64).to(device)
    opt = torch.optim.Adam(list(reward.parameters()) + list(policy.parameters()), lr=1e-3)

    for step in range(10):
        x = torch.randn(32, 64, device=device)
        action = policy(x)
        r = reward(action).mean()
        loss = -r
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("step", step, "reward", float(r))

    ptx = list(Path("./torchinductor_cache").rglob("*.ptx"))
    print("ptx files", len(ptx))
    for p in ptx[:5]:
        print(p)


if __name__ == "__main__":
    main()
