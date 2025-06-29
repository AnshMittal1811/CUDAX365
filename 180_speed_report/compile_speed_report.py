import csv
import glob
import os
import re

LOGS = [
    "../148_flux_kernel_finalize/flux_bench.txt",
    "../170_cuda_131_benchmarks/bench_results.txt",
    "../177_cpu_gpu_latency/latency_results.txt",
    "../178_async_memcpy_overlap/async_copy_results.txt",
]


def parse_value(path, pattern):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))
    return None


def main():
    rows = []
    for path in LOGS:
        if "flux_bench" in path:
            val = parse_value(path, r"optimized_ms=([0-9.]+)")
            rows.append((path, "optimized_ms", val))
        elif "bench_results" in path:
            val = parse_value(path, r"bench_kernel_ms=([0-9.]+)")
            rows.append((path, "bench_kernel_ms", val))
        elif "latency_results" in path:
            val = parse_value(path, r"avg_launch_us=([0-9.]+)")
            rows.append((path, "avg_launch_us", val))
        elif "async_copy_results" in path:
            val = parse_value(path, r"async_copy_ms=([0-9.]+)")
            rows.append((path, "async_copy_ms", val))

    with open("speed_report.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["log", "metric", "value"])
        for row in rows:
            writer.writerow(row)

    with open("speed_report.md", "w", encoding="utf-8") as f:
        f.write("# Speed Report\n\n")
        f.write("| Log | Metric | Value |\n")
        f.write("| --- | --- | --- |\n")
        for log, metric, value in rows:
            value_str = "TBD" if value is None else f"{value:.4f}"
            f.write(f"| {log} | {metric} | {value_str} |\n")

    print("Wrote speed_report.csv and speed_report.md")


if __name__ == "__main__":
    main()
