import time

try:
    import torch
except Exception:
    torch = None


def main():
    if torch is None:
        with open("cudnn_fusion_log.txt", "w", encoding="utf-8") as f:
            f.write("torch not available\n")
        print("torch not available")
        return

    x = torch.randn(1, 3, 64, 64, device="cuda")
    w = torch.randn(8, 3, 3, 3, device="cuda")
    b = torch.randn(8, device="cuda")

    torch.backends.cudnn.benchmark = True

    start = time.time()
    y = torch.nn.functional.conv2d(x, w, bias=b)
    y = torch.relu(y)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000.0

    with open("cudnn_fusion_log.txt", "w", encoding="utf-8") as f:
        f.write(f"elapsed_ms={elapsed:.3f}\n")

    print("Wrote cudnn_fusion_log.txt")


if __name__ == "__main__":
    main()
