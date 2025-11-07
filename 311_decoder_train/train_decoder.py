
import argparse
import json
import os
import time
import numpy as np

def build_numpy_model(in_dim, out_dim, seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((in_dim, 64)).astype(np.float32) * 0.1
    b1 = np.zeros(64, dtype=np.float32)
    w2 = rng.standard_normal((64, out_dim)).astype(np.float32) * 0.1
    b2 = np.zeros(out_dim, dtype=np.float32)
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

def numpy_forward(model, x):
    h = np.maximum(0, x @ model['w1'] + model['b1'])
    y = h @ model['w2'] + model['b2']
    return y

def train_numpy(data, epochs, lr):
    x = data['syndromes']
    y = data['errors']
    model = build_numpy_model(x.shape[1], y.shape[1])
    for _ in range(epochs):
        logits = numpy_forward(model, x)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = (probs - y) / y.shape[0]
        h = np.maximum(0, x @ model['w1'] + model['b1'])
        model['w2'] -= lr * (h.T @ grad)
        model['b2'] -= lr * grad.sum(axis=0)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train a small decoder model')
    parser.add_argument('--data', type=str, default='decoder_data.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--out', type=str, default='decoder_model.pt')
    args = parser.parse_args()

    data = np.load(args.data)
    in_dim = data['syndromes'].shape[1]
    out_dim = data['errors'].shape[1]

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        torch_available = True
    except Exception:
        torch_available = False

    metrics = {
        'epochs': args.epochs,
        'batch': args.batch,
        'lr': args.lr,
        'torch': torch_available,
    }

    if not torch_available:
        model = train_numpy(data, args.epochs, args.lr)
        np.savez('decoder_model.npz', **model)
        metrics['note'] = 'torch not available; wrote numpy weights'
        with open('training_log.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print('Torch not available. Saved numpy model to decoder_model.npz')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim),
    ).to(device)

    dataset = TensorDataset(
        torch.from_numpy(data['syndromes']),
        torch.from_numpy(data['errors']),
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    start = time.time()
    for epoch in range(args.epochs):
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()
        avg_loss = running / max(1, len(loader))
        print(f'Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f}')
    elapsed = time.time() - start

    torch.save({
        'state_dict': model.state_dict(),
        'in_dim': in_dim,
        'out_dim': out_dim,
    }, args.out)
    metrics.update({'elapsed_sec': elapsed, 'device': str(device)})
    with open('training_log.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved model to {args.out}')

if __name__ == '__main__':
    main()
