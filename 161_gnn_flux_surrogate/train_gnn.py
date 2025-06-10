import argparse
import numpy as np


def build_adjacency(edges, num_nodes):
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
    return adj


def train_numpy(features, edges, epochs=50):
    num_nodes, feat_dim = features.shape
    weights = np.random.randn(feat_dim, feat_dim).astype(np.float32) * 0.1
    adj = build_adjacency(edges, num_nodes)
    target = features * 0.5

    lr = 0.05
    for _ in range(epochs):
        agg = np.zeros_like(features)
        for i in range(num_nodes):
            neighbors = adj[i]
            if not neighbors:
                continue
            agg[i] = np.mean(features[neighbors], axis=0)
        out = agg @ weights
        loss = np.mean((out - target) ** 2)
        grad = (2.0 / num_nodes) * (agg.T @ (out - target))
        weights -= lr * grad
    return loss


def train_torch(features, edges, epochs=50):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(features, device=device)
    edges_t = torch.tensor(edges, device=device)
    num_nodes, feat_dim = x.shape
    weights = torch.randn(feat_dim, feat_dim, device=device) * 0.1
    target = x * 0.5

    lr = 0.05
    for _ in range(epochs):
        agg = torch.zeros_like(x)
        for u, v in edges_t:
            agg[u] += x[v]
        counts = torch.zeros(num_nodes, device=device)
        for u, _ in edges_t:
            counts[u] += 1
        counts = torch.clamp(counts, min=1.0)
        agg = agg / counts.unsqueeze(1)
        out = agg @ weights
        loss = torch.mean((out - target) ** 2)
        loss.backward()
        with torch.no_grad():
            weights -= lr * weights.grad
            weights.grad.zero_()
    return float(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="mesh_graph.npz")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    data = np.load(args.graph)
    features = data["features"]
    edges = data["edges"]

    try:
        loss = train_torch(features, edges, epochs=args.epochs)
        backend = "torch"
    except Exception:
        loss = train_numpy(features, edges, epochs=args.epochs)
        backend = "numpy"

    with open("gnn_train_log.txt", "w", encoding="utf-8") as f:
        f.write(f"backend={backend}\n")
        f.write(f"loss={loss}\n")

    print(f"Trained with {backend}, loss={loss:.6f}")


if __name__ == "__main__":
    main()
