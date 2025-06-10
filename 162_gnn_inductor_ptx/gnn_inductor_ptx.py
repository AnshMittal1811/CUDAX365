import os
import torch

os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torchinductor_cache"


def build_graph(num_nodes=256, feat_dim=8):
    x = torch.randn(num_nodes, feat_dim, device="cuda")
    edges = torch.randint(0, num_nodes, (num_nodes * 4, 2), device="cuda")
    return x, edges


@torch.compile

def gnn_forward(x, edges):
    num_nodes = x.shape[0]
    feat_dim = x.shape[1]
    agg = torch.zeros_like(x)
    counts = torch.zeros(num_nodes, device=x.device)
    for u, v in edges:
        agg[u] += x[v]
        counts[u] += 1
    counts = torch.clamp(counts, min=1.0).unsqueeze(1)
    agg = agg / counts
    w = torch.randn(feat_dim, feat_dim, device=x.device)
    return agg @ w


def main():
    x, edges = build_graph()
    out = gnn_forward(x, edges)
    torch.cuda.synchronize()
    print(out.shape)


if __name__ == "__main__":
    main()
