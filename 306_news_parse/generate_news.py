articles = [
    "New hospital built in Midtown; address 123 Main St.",
    "Bridge project announced near River Park on 5th Ave.",
]

with open("news.txt", "w", encoding="utf-8") as f:
    for a in articles:
        f.write(a + "\n")

print("Wrote news.txt")
