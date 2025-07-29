notes = [
    "Dynamic parallelism launches child grids from device code.",
    "Occupancy is limited by parent kernel resources and child launch overhead.",
    "Nested launches incur latency; batch work when possible.",
    "Use -rdc=true and link with -lcudadevrt to enable device-side launches.",
    "Measure child grid occupancy with Nsight Compute if supported.",
]

with open("dynpar_notes.md", "w", encoding="utf-8") as f:
    f.write("# Dynamic Parallelism Notes\n\n")
    for line in notes:
        f.write(f"- {line}\n")

print("Wrote dynpar_notes.md")
