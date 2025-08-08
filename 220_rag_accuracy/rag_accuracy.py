import json


def main():
    questions = json.load(open("questions.json", "r", encoding="utf-8"))
    doc = open("../219_rag_faiss/rag_result.txt", "r", encoding="utf-8").read().lower()

    correct = 0
    for item in questions:
        if item["a"].lower() in doc:
            correct += 1

    acc = correct / len(questions)
    with open("rag_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy={acc:.3f}\n")

    print("Wrote rag_accuracy.txt")


if __name__ == "__main__":
    main()
