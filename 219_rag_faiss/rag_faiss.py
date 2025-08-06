import json
import numpy as np


def embed(text):
    vec = np.zeros(8, dtype=np.float32)
    for i, ch in enumerate(text.encode("utf-8")):
        vec[i % 8] += ch * 0.001
    return vec


def main():
    docs = json.load(open("docs.json", "r", encoding="utf-8"))
    vectors = np.stack([embed(d) for d in docs], axis=0)

    query = "How does the solver handle flux?"
    qvec = embed(query)
    scores = vectors @ qvec
    idx = int(np.argmax(scores))

    with open("rag_result.txt", "w", encoding="utf-8") as f:
        f.write(f"query={query}\n")
        f.write(f"top_doc={docs[idx]}\n")

    print("Wrote rag_result.txt")


if __name__ == "__main__":
    main()
