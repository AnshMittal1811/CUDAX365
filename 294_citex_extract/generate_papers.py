papers = [
    "MHD study (Smith et al. 2023) introduces new flux scheme.",
    "Magnetics results in Doe 2022 show improved stability.",
]

with open("papers.txt", "w", encoding="utf-8") as f:
    for line in papers:
        f.write(line + "\n")

print("Wrote papers.txt")
