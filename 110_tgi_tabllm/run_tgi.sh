#!/usr/bin/env bash
set -euo pipefail
# Requires Docker and HuggingFace TGI image
MODEL=${MODEL:-"TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
PORT=${PORT:-8080}

docker run --gpus all -p ${PORT}:80 \
  -e MODEL_ID=${MODEL} \
  ghcr.io/huggingface/text-generation-inference:latest
