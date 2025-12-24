
#!/usr/bin/env bash
set -euo pipefail

bash ../311_decoder_train/run_decoder_train.sh || true
bash ../340_batch_scaling/run_batch_scaling.sh || true
bash ../350_segmentation_overlay/run_overlay.sh || true
echo "Validation run complete (check outputs in each folder)."
