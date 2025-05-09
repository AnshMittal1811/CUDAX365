import csv

with open("stalls.csv", "r", encoding="utf-8") as f:
    rows = list(csv.reader(f))

vals = [r for r in rows if r and r[0].startswith("smsp__")]
if not vals:
    print("no metrics found")
    raise SystemExit

for r in vals:
    print("metric", r[0], "value", r[-1])
