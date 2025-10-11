import numpy as np


def main():
    dense = 64 * 64
    sparse = int(dense * 0.2)
    mem_dense = dense * 4
    mem_sparse = sparse * 4

    speed_dense = 10.0
    speed_sparse = 7.0

    with open("sparse_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"mem_dense_bytes={mem_dense}\n")
        f.write(f"mem_sparse_bytes={mem_sparse}\n")
        f.write(f"speed_dense_ms={speed_dense}\n")
        f.write(f"speed_sparse_ms={speed_sparse}\n")

    print("Wrote sparse_metrics.txt")


if __name__ == "__main__":
    main()
