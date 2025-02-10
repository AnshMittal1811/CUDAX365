from pathlib import Path

samples = [
    "t=0.0 rho=1.0 phi=0.0 bx=0.1 by=0.0",
    "t=0.1 rho=1.02 phi=0.01 bx=0.1 by=0.01",
    "t=0.2 rho=0.98 phi=0.02 bx=0.11 by=0.02",
]

out = Path("mhd_logs.txt")
out.write_text("\n".join(samples), encoding="utf-8")
print("wrote", out)
