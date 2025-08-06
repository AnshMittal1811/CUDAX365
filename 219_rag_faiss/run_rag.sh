#!/usr/bin/env bash
set -euo pipefail

python generate_docs.py
python rag_faiss.py
