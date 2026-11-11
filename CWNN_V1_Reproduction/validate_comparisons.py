import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from predict import predict
from run import execute


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Full-run output from run.py --baselines")
    reference = parser.parse_args().directory
    records = []
    for folder in sorted(reference.iterdir()):
        if not (folder / "trajectory.npz").exists():
            continue
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        summary, arrays, state = execute(folder.name, profile=metrics["profile"])
        with np.load(folder / "trajectory.npz", allow_pickle=False) as recorded:
            for key, value in arrays.items():
                if key != "parameters" or len(value):
                    np.testing.assert_array_equal(value, recorded[key])
        if (folder / "dataset.npz").exists():
            with np.load(folder / "dataset.npz", allow_pickle=False) as dataset:
                if "x_test" in dataset:
                    with threadpool_limits(1):
                        predictions = predict(folder / "model.npz", dataset["x_test"])
                    error = float(np.mean((predictions - dataset["y_test"]) ** 2))
                    np.testing.assert_allclose(
                        error, summary["last_test_mse_after_update"], rtol=2e-6, atol=1e-8
                    )
        records.append(
            dict(experiment=folder.name, trajectory_exact=True, epochs=summary["epochs"])
        )
    output = root / "results"
    output.mkdir(exist_ok=True)
    (output / "comparison_checks.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"Verified {len(records)} complete trajectories and all saved test predictions.")


if __name__ == "__main__":
    main()
