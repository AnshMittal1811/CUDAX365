import json
import random


def main():
    proposals = json.load(open("../291_autogpt_tuner/autotune_proposals.json", "r", encoding="utf-8"))
    rng = random.Random(1)
    results = []
    for p in proposals:
        score = rng.uniform(0.1, 1.0)
        results.append({"cfl": p["cfl"], "block": p["block"], "score": score})

    with open("config_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Wrote config_results.json")


if __name__ == "__main__":
    main()
