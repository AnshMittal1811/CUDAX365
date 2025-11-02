import re

text = open("news.txt", "r", encoding="utf-8").read()

locations = re.findall(r"\b\d+\s+\w+\s+St\b|\b\d+th\s+Ave\b", text)

with open("locations.txt", "w", encoding="utf-8") as f:
    for loc in locations:
        f.write(loc + "\n")

print("Wrote locations.txt")
