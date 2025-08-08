#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ../219_rag_faiss/rag_result.txt ]]; then
  (cd ../219_rag_faiss && ./run_rag.sh)
fi

python generate_questions.py
python rag_accuracy.py
