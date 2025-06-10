import numpy as np

rng = np.random.RandomState(0)

num_nodes = 512
coords = rng.rand(num_nodes, 2).astype(np.float32)
features = rng.randn(num_nodes, 4).astype(np.float32)

edges = []
for i in range(num_nodes):
    distances = np.sum((coords - coords[i]) ** 2, axis=1)
    neighbors = np.argsort(distances)[1:6]
    for j in neighbors:
        edges.append([i, j])

edges = np.array(edges, dtype=np.int64)

np.savez("mesh_graph.npz", coords=coords, features=features, edges=edges)
print("Wrote mesh_graph.npz")
