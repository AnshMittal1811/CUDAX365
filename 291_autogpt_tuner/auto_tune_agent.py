import json
import random


def main():
    rng = random.Random(0)
    proposals = []
    for i in range(5):
        proposals.append({
            "cfl": round(rng.uniform(0.1, 0.5), 3),
            "block": rng.choice([128, 256, 512]),
            "note": "auto-generated",
        })

    with open("autotune_proposals.json", "w", encoding="utf-8") as f:
        json.dump(proposals, f, indent=2)

    print("Wrote autotune_proposals.json")


if __name__ == "__main__":
    main()
