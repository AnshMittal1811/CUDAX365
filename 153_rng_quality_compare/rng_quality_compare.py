import argparse
import json
import os
import struct

import math


def load_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    count = len(data) // 4
    return list(struct.unpack("%sf" % count, data))


def chi_square(data, bins=64):
    hist = [0] * bins
    for v in data:
        idx = int(v * bins)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        hist[idx] += 1
    expected = len(data) / float(bins)
    chi2 = 0.0
    for count in hist:
        diff = count - expected
        chi2 += diff * diff / expected
    return chi2


def ks_stat(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    d_max = 0.0
    for i, v in enumerate(sorted_data, 1):
        cdf = i / n
        d = abs(cdf - v)
        if d > d_max:
            d_max = d
    return d_max


def summarize(label, data):
    mean = sum(data) / len(data)
    var = sum((x - mean) ** 2 for x in data) / len(data)
    return {
        "mean": mean,
        "var": var,
        "chi2": chi_square(data),
        "ks": ks_stat(data),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sobol-device", default="../151_curand_sobol_quality/sobol_device.bin")
    parser.add_argument("--sobol-host", default="../151_curand_sobol_quality/sobol_host.bin")
    parser.add_argument("--xorshift", default="../152_ptx_xorshift_warp/xorshift.bin")
    parser.add_argument("--out", default="rng_quality_report.json")
    args = parser.parse_args()

    report = {}
    for label, path in [
        ("sobol_device", args.sobol_device),
        ("sobol_host", args.sobol_host),
        ("xorshift", args.xorshift),
    ]:
        if not os.path.exists(path):
            report[label] = {"error": f"missing {path}"}
            continue
        data = load_bin(path)
        report[label] = summarize(label, data)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {args.out}")
    for label, stats in report.items():
        print(label, stats)


if __name__ == "__main__":
    main()
