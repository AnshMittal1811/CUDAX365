
#!/usr/bin/env bash
set -euo pipefail

python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
