#!/usr/bin/env bash
set -euo pipefail

if ! python -c "import grpc" >/dev/null 2>&1; then
  echo "grpcio not installed"
  exit 1
fi

if python -c "import grpc_tools" >/dev/null 2>&1; then
  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. param_server.proto
else
  echo "grpc_tools not available; install grpcio-tools to generate stubs"
  exit 1
fi

python server.py &
SERVER_PID=$!

sleep 1
python client.py
kill "$SERVER_PID" || true
