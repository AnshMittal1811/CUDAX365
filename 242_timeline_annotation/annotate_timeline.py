notes = [
    ("Load", "Global memory load phase"),
    ("ALU", "Compute and FMA operations"),
    ("Store", "Writeback to global memory"),
    ("Sync", "Barrier or event wait"),
]

with open("timeline_annotations.txt", "w", encoding="utf-8") as f:
    for stage, desc in notes:
        f.write(f"{stage}: {desc}\n")

print("Wrote timeline_annotations.txt")
