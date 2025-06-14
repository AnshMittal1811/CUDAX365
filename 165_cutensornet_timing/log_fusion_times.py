import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="../164_cutensornet_fusion/fusion_metrics.txt")
    parser.add_argument("--out", default="fusion_times.csv")
    args = parser.parse_args()

    metrics = {}
    with open(args.metrics, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                metrics[k] = v

    conv = float(metrics.get("conv_ms", 0.0))
    gemm = float(metrics.get("gemm_ms", 0.0))
    fused = float(metrics.get("fused_ms", 0.0))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("mode,ms\n")
        f.write(f"conv,{conv:.3f}\n")
        f.write(f"gemm,{gemm:.3f}\n")
        f.write(f"fused,{fused:.3f}\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
