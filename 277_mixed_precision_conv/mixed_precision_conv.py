import time

try:
    import torch
except Exception:
    torch = None


def main():
    if torch is None:
        with open("mixed_precision_log.txt", "w", encoding="utf-8") as f:
            f.write("torch not available\n")
        print("torch not available")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, 64, 64, device=device)
    w = torch.randn(8, 3, 3, 3, device=device)

    start = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        y = torch.nn.functional.conv2d(x, w)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000.0

    with open("mixed_precision_log.txt", "w", encoding="utf-8") as f:
        f.write(f"elapsed_ms={elapsed:.3f}\n")

    print("Wrote mixed_precision_log.txt")


if __name__ == "__main__":
    main()
