import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    root = args.root or str(Path.home() / "datasets" / "cityscapes")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf_img = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
    ])
    tf_lbl = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    ds = Cityscapes(root, split="train", mode="fine", target_type="semantic", transform=tf_img, target_transform=tf_lbl)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    processor = SegformerImageProcessor(do_reduce_labels=True)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(dl):
            images = images.to(device)
            labels = labels.to(device)
            inputs = processor(images=images, labels=labels, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 20 == 0:
                print(f"epoch {epoch} step {i} loss {loss.item():.4f}")
            if i >= 100:
                break

    print("done")


if __name__ == "__main__":
    main()
