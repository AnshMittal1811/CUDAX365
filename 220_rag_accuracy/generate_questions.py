import json

QUESTIONS = [
    {"q": "What flux scheme is used?", "a": "Rusanov"},
    {"q": "What does NeRF integrate?", "a": "density"},
    {"q": "Why use CUDA Graphs?", "a": "reduce launch overhead"},
]

with open("questions.json", "w", encoding="utf-8") as f:
    json.dump(QUESTIONS, f, indent=2)

print("Wrote questions.json")
