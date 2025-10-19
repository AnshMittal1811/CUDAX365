import json


def main():
    results = json.load(open("../292_agent_configs/config_results.json", "r", encoding="utf-8"))
    best_agent = min(results, key=lambda r: r["score"])
    manual = {"cfl": 0.3, "block": 256, "score": 0.5}

    with open("agent_compare.txt", "w", encoding="utf-8") as f:
        f.write(f"agent_best={best_agent}\n")
        f.write(f"manual={manual}\n")

    print("Wrote agent_compare.txt")


if __name__ == "__main__":
    main()
