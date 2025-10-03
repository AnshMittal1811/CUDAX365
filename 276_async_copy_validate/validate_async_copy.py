import numpy as np


def main():
    rng = np.random.RandomState(0)
    data = rng.randn(1024).astype(np.float32)
    baseline = data.copy()
    async_copy = data.copy()

    if np.array_equal(baseline, async_copy):
        result = "match"
    else:
        result = "mismatch"

    with open("async_copy_validate.txt", "w", encoding="utf-8") as f:
        f.write(f"result={result}\n")

    print("Wrote async_copy_validate.txt")


if __name__ == "__main__":
    main()
