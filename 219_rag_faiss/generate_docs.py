import json

DOCS = [
    "MHD solver uses Rusanov flux with CFL control.",
    "NeRF volume rendering integrates density along rays.",
    "GNN surrogate models flux Jacobian for faster updates.",
    "CUDA Graphs reduce kernel launch overhead.",
]

with open("docs.json", "w", encoding="utf-8") as f:
    json.dump(DOCS, f, indent=2)

print("Wrote docs.json")
