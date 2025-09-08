import os


def ptx_size(path):
    if not os.path.exists(path):
        return None
    return os.path.getsize(path)


def main():
    cute_ptx = "../251_cute_conv/cute_conv.ptx"
    base_ptx = "baseline_conv.ptx"

    size_cute = ptx_size(cute_ptx)
    size_base = ptx_size(base_ptx)

    with open("ptx_compare.txt", "w", encoding="utf-8") as f:
        f.write(f"cute_ptx_size={size_cute}\n")
        f.write(f"baseline_ptx_size={size_base}\n")

    print("Wrote ptx_compare.txt")


if __name__ == "__main__":
    main()
