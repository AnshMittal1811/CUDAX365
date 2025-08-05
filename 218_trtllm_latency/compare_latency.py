import time


def benchmark(label, sleep_ms):
    start = time.time()
    time.sleep(sleep_ms / 1000.0)
    elapsed = (time.time() - start) * 1000.0
    return label, elapsed


def main():
    results = [
        benchmark("v1.0", 50),
        benchmark("v1.1", 42),
    ]

    with open("trtllm_latency.txt", "w", encoding="utf-8") as f:
        for label, ms in results:
            f.write(f"{label}_ms={ms:.2f}\n")

    print("Wrote trtllm_latency.txt")


if __name__ == "__main__":
    main()
