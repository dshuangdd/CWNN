import json
import numpy as np


def predict(path, x):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta["backend"] == "gnn":
            import torch
            from legacy.gnn import GNN

            with torch.random.fork_rng(devices=[]):
                model = GNN(meta["input_size"], meta["num_blocks"])
            model.load_state_dict(
                {
                    key[6:]: torch.from_numpy(data[key])
                    for key in data.files
                    if key.startswith("state_")
                }
            )
            model.eval()
            x = torch.as_tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return (
                    model((x - torch.from_numpy(data["mean"])) / torch.from_numpy(data["std"]))
                    .numpy()
                    .ravel()
                )
        if meta["backend"] == "torch":
            import torch

            x = torch.as_tensor(x, dtype=torch.float32)
            output = 0
            for i in range(meta["layers"]):
                centers, weights = torch.from_numpy(data[f"centers_{i}"]), torch.from_numpy(
                    data[f"weights_{i}"]
                )
                j = meta["initial_resolution"] + i
                y = torch.norm(2**j * x.unsqueeze(1) - centers, dim=2) + 1e-6
                psi = 2**j * (-torch.sin(2 * y) / y + torch.sin(y) / y)
                output += psi @ weights[len(centers) :] if i == 0 else psi @ weights[: len(centers)]
                if i == 0:

                    output = 2**j * torch.sin(y) / y @ weights[: len(centers)] + output
            return output.numpy()
        x = np.asarray(x, dtype=float)[None, :, :]

        def basis(j, centers):
            r = np.sum((2**j * x - centers) ** 2, 2) ** 0.5 + np.pi / 1e6
            return np.array([2**j * np.sin(r) / r, 2**j * (-np.sin(2 * r) / r + np.sin(r) / r)])

        j0, j = meta["initial_resolution"], meta["parent_resolution"]
        k = data["centers_0"]
        first = basis(j0, k[: (2 * 2**j0 + 1) ** 2])
        matrix = np.concatenate((first[0], first[1]), 0)
        if not meta["grown"]:
            return matrix.T @ data["weights_0"]
        for level in range(j0 + 1, j + 1):
            start = (2 * 2 ** (level - 1) + 1) ** 2
            matrix = np.concatenate(
                (matrix, basis(level, k[start : start + (2 * 2**level + 1) ** 2])[1]), 0
            )
        return (
            matrix.T @ data["weights_0"] + basis(j + 1, data["centers_1"])[1].T @ data["weights_1"]
        )
