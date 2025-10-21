import re

text = open("papers.txt", "r", encoding="utf-8").read()

citations = re.findall(r"[A-Z][a-zA-Z]+(?: et al\.)?\s+\d{4}", text)

with open("citations.txt", "w", encoding="utf-8") as f:
    for c in citations:
        f.write(c + "\n")

print("Wrote citations.txt")
