import sys

if len(sys.argv) < 2:
    print("usage: count_ptx.py <file.ptx>")
    sys.exit(1)

with open(sys.argv[1], "r", encoding="utf-8") as f:
    lines = [l for l in f if l.strip() and not l.strip().startswith("//")]

inst = [l for l in lines if l.strip().endswith(";")]
print("instructions", len(inst))
