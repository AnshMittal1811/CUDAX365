import json
import time
import numpy as np


def quantize_int4(x):
    max_val = np.max(np.abs(x))
    scale = max_val / 7.0 if max_val != 0 else 1.0
    q = np.clip(np.round(x / scale), -8, 7)
    return q * scale


def run_pipeline():
    gnn = np.load("gnn_input.npy")
    pde = np.load("pde_state.npy")
    volume = np.load("nerf_volume.npy")

    t0 = time.time()
    gnn_q = quantize_int4(gnn)
    gnn_out = gnn_q.mean(axis=0)
    t1 = time.time()

    pde_out = pde.copy()
    pde_out[1:-1, 1:-1] = 0.25 * (
        pde[1:-1, :-2] + pde[1:-1, 2:] + pde[:-2, 1:-1] + pde[2:, 1:-1]
    )
    t2 = time.time()

    projection = volume.mean(axis=0)
    projection = (projection - projection.min()) / (projection.ptp() + 1e-6)
    t3 = time.time()

    report = {
        "gnn_ms": (t1 - t0) * 1000.0,
        "pde_ms": (t2 - t1) * 1000.0,
        "nerf_ms": (t3 - t2) * 1000.0,
        "gnn_out_mean": float(gnn_out.mean()),
    }

    with open("pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Wrote pipeline_report.json")


if __name__ == "__main__":
    run_pipeline()
