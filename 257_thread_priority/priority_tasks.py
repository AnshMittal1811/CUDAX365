import time


def run_task(name, steps, delay):
    for i in range(steps):
        with open(f"{name}_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{name} step {i}\n")
        time.sleep(delay)


def main():
    run_task("render", 10, 0.1)
    run_task("encode", 10, 0.1)


if __name__ == "__main__":
    main()
