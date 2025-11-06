import time


def decode(syndrome):
    table = {
        "000": "no_error",
        "001": "flip_q2",
        "010": "flip_q1",
        "100": "flip_q0",
    }
    return table.get(syndrome, "unknown")


def main():
    start = time.time()
    for _ in range(10000):
        _ = decode("001")
    elapsed = (time.time() - start) * 1e6 / 10000

    with open("cpu_decoder_log.txt", "w", encoding="utf-8") as f:
        f.write(f"latency_us={elapsed:.4f}\n")

    print("Wrote cpu_decoder_log.txt")


if __name__ == "__main__":
    main()
