#!/usr/bin/env bash
set -euo pipefail

python generate_mock_mesh.py
python train_gnn.py --graph mesh_graph.npz --epochs 50
