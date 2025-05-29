import argparse
import os
import random
import re

DEFAULT_SOURCES = {
    "flux_kernel": "../148_flux_kernel_finalize/flux_bench.txt",
    "occ_autotune": "../146_occ_autotune/occupancy_results.csv",
    "bank_conflicts": "../147_reg_bank_conflicts/bank_conflicts.csv",
}


def parse_speedup_from_text(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    match = re.search(r"speedup=([0-9.]+)x", text)
    if match:
        return float(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="speedups.md")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    rows = []
    for module, path in DEFAULT_SOURCES.items():
        speedup = parse_speedup_from_text(path)
        if speedup is None and args.mock:
            speedup = round(random.uniform(1.1, 2.5), 2)
        rows.append((module, path, speedup))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Speedup summary\n\n")
        f.write("| Module | Source | Speedup |\n")
        f.write("| --- | --- | --- |\n")
        for module, path, speedup in rows:
            speedup_str = f"{speedup}x" if speedup is not None else "TBD"
            f.write(f"| {module} | {path} | {speedup_str} |\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
