
#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TENSORRT_HOME:-}" ]]; then
  echo "Set TENSORRT_HOME to build the plugin with TensorRT headers." >&2
else
  echo "Build command (example):"
  echo "g++ -shared -fPIC -I$TENSORRT_HOME/include plugin_stub.cpp -o libswish_plugin.so"
fi

python plugin_demo.py
