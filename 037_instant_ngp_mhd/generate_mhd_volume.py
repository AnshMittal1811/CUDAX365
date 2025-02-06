import json
import numpy as np
from pathlib import Path


def main():
    out_dir = Path("mhd_volume")
    out_dir.mkdir(exist_ok=True)
    nx, ny, nz = 64, 64, 64
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    rho = np.exp(-4.0 * (X**2 + Y**2 + Z**2))
    rho = rho.astype(np.float32)
    raw_path = out_dir / "density.raw"
    rho.tofile(raw_path)

    meta = {
        "dataset": {
            "type": "volume",
            "data": "density.raw",
            "resolution": [nx, ny, nz],
            "aabb_scale": 1,
            "density_scale": 1.0
        }
    }
    with open(out_dir / "volume.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote", raw_path)


if __name__ == "__main__":
    main()
