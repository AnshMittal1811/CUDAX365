import torch

x = torch.randn(2048, 2048, device="cuda")
for _ in range(200):
    x = x @ x
print("workload done")
