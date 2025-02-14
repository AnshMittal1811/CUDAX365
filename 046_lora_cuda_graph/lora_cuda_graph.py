import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_f, r))
    def forward(self, x):
        return x @ self.weight.t() + (x @ self.A.t()) @ self.B.t()


def main():
    device = "cuda"
    model = LoRALinear(256, 256).to(device)
    x = torch.randn(64, 256, device=device)
    y = torch.randn(64, 256, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    static_x = torch.randn_like(x)
    static_y = torch.randn_like(y)

    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        out = model(static_x)
        loss = (out - static_y).pow(2).mean()
        loss.backward()
        opt.step()

    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    print("graph replayed")


if __name__ == "__main__":
    main()
