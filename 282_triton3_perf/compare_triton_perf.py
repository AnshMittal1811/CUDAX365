import os


def read_elapsed(path):
    if not os.path.exists(path):
        return None
    for line in open(path, "r", encoding="utf-8"):
        if "elapsed_sec" in line:
            return float(line.strip().split("=")[1])
    return None


triton2 = read_elapsed("../201_triton_asm_conv/triton_conv_log.txt")
triton3 = read_elapsed("../281_triton3_port/triton3_log.txt")

with open("triton_perf_compare.txt", "w", encoding="utf-8") as f:
    f.write(f"triton2_sec={triton2}\n")
    f.write(f"triton3_sec={triton3}\n")

print("Wrote triton_perf_compare.txt")
