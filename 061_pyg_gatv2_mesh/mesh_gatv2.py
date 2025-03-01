import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


def build_grid_graph(n=32):
    coords = []
    edges = []
    for y in range(n):
        for x in range(n):
            idx = y * n + x
            coords.append([x / n, y / n])
            if x + 1 < n:
                edges.append([idx, idx + 1])
                edges.append([idx + 1, idx])
            if y + 1 < n:
                edges.append([idx, idx + n])
                edges.append([idx + n, idx])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(coords, dtype=torch.float32)
    y = torch.sin(x[:, :1])
    return Data(x=x, edge_index=edge_index, y=y)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = build_grid_graph(32).to(device)

    conv1 = GATv2Conv(2, 32, heads=4, concat=True).to(device)
    conv2 = GATv2Conv(128, 1, heads=1, concat=False).to(device)
    opt = torch.optim.Adam(list(conv1.parameters()) + list(conv2.parameters()), lr=1e-3)

    conv1.train()
    conv2.train()
    for epoch in range(5):
        opt.zero_grad()
        h = conv1(data.x, data.edge_index)
        out = conv2(h, data.edge_index)
        loss = (out - data.y).pow(2).mean()
        loss.backward()
        opt.step()
        print("epoch", epoch, "loss", float(loss))


if __name__ == "__main__":
    main()
