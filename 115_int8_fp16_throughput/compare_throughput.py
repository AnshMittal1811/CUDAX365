import subprocess
import re

out = subprocess.check_output(["../114_dp4a_vs_hmma/dp4a_vs_hmma"], text=True)
print(out.strip())
match = re.search(r"dp4a_ms=([0-9.]+) wmma_ms=([0-9.]+)", out)
if match:
    dp4a = float(match.group(1))
    wmma = float(match.group(2))
    if dp4a > 0:
        print("wmma_vs_dp4a_ratio", wmma / dp4a)
