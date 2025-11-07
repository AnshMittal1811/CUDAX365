
#!/usr/bin/env bash
set -euo pipefail

python generate_decoder_data.py
python train_decoder.py
