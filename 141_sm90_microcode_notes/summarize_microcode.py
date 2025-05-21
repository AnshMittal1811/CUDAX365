import csv
import os

REPORT = "ncu_report.csv"
OUT = "microcode_summary.md"

METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_active",
]


def find_metric_value(path, metric):
    if not os.path.exists(path):
        return None
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0].strip() == metric:
                return row[1].strip()
    return None


def main():
    results = {}
    for metric in METRICS:
        results[metric] = find_metric_value(REPORT, metric)

    notes = [
        "Hopper (SM90) uses a wider warp scheduling pipeline and supports concurrent issue paths.",
        "Warp occupancy should be interpreted alongside issue utilization and memory stall metrics.",
        "Higher occupancy does not guarantee higher performance if the kernel is memory bound.",
        "Compare the measured occupancy to expected limits (registers, shared memory, block size).",
    ]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# SM90 microcode scheduling notes\n\n")
        f.write("## Measured occupancy (from ncu_report.csv)\n\n")
        for metric, value in results.items():
            f.write(f"- {metric}: {value if value is not None else 'N/A'}\n")
        f.write("\n## Notes from Hopper whitepaper (summary)\n\n")
        for line in notes:
            f.write(f"- {line}\n")
        f.write("\n## Next steps\n\n")
        f.write("- Re-run profiling with different block sizes and compare occupancy/throughput.\n")
        f.write("- Cross-check stalls in Nsight Compute Source view.\n")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
