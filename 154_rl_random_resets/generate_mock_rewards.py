import csv
import random

rng = random.Random(1)

with open("mock_rewards.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward"])
    for ep in range(200):
        writer.writerow([ep, rng.uniform(10.0, 200.0)])

print("Wrote mock_rewards.csv")
