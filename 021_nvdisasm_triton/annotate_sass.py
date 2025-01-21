import re
import sys


def tag_line(line):
    op = line.strip().split()
    if not op:
        return line
    inst = op[-1] if op[-1].isupper() else op[0]
    if re.search(r"LD|ST|LDG|STS|LDS|TEX", line):
        return line.rstrip() + "    // [MEM]\n"
    if re.search(r"HMMA|HFMA2|FFMA|FADD|FMUL|IMAD|IADD", line):
        return line.rstrip() + "    // [MATH]\n"
    if re.search(r"BRA|EXIT|RET|SSY|SYNC|BAR", line):
        return line.rstrip() + "    // [CTRL]\n"
    return line


def main():
    if len(sys.argv) < 2:
        print("Usage: annotate_sass.py <input.sass> [output.sass]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    with open(inp, "r", encoding="utf-8") as f:
        lines = f.readlines()
    tagged = [tag_line(l) for l in lines]
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(tagged)
    else:
        sys.stdout.writelines(tagged)


if __name__ == "__main__":
    main()
