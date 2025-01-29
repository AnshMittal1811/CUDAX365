import argparse
from pathlib import Path
from torchvision.datasets import Cityscapes
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Cityscapes root folder")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    root = args.root or str(Path.home() / "datasets" / "cityscapes")
    tf = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
    ])
    ds = Cityscapes(root, split=args.split, mode="fine", target_type="semantic", transform=tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    batch = next(iter(dl))
    imgs, labels = batch
    print("images", imgs.shape)
    print("labels", labels.shape)
    print("root", root)


if __name__ == "__main__":
    main()
