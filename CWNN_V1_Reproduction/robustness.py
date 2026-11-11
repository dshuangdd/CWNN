import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from comparisons import case_and_method
from predict import predict


def noisy_targets(noise_scale, seed, offset=-1.0):

    x1 = np.arange(100) / 100
    x2 = np.sqrt(x1)
    noise = np.random.RandomState(seed).normal(0, noise_scale * (1 - x1**2), len(x1))
    return offset + (x1 + x2) + np.sin(2 * np.pi * (x1 + x2)) + noise


def evaluate(directory, trials=100, output=None):

    directory = Path(directory)
    output = directory / "robustness" if output is None else Path(output)
    if trials < 1:
        raise ValueError("trials must be positive")
    folders = sorted(
        p
        for p in directory.iterdir()
        if p.is_dir() and p.name.split("_")[0] in ("D2", "D3") and (p / "model.npz").is_file()
    )
    if not folders:
        raise ValueError("No saved D2/D3 models found; run example1.py first")
    output.mkdir(parents=True, exist_ok=False)
    rows, summaries, checkpoints = [], [], {}
    for folder in folders:
        case, method = case_and_method(folder.name)
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        offset = metrics.get("settings", {}).get("offset", -1.0)
        with np.load(folder / "dataset.npz", allow_pickle=False) as dataset:
            indices = dataset["test_indices"]
            with threadpool_limits(1):
                predictions = predict(folder / "model.npz", dataset["x_test"])
        checkpoints[folder.name] = hashlib.sha256((folder / "model.npz").read_bytes()).hexdigest()
        errors = []
        for trial in range(trials):
            seed = trial * 10
            targets = noisy_targets(0.1 if case == "D2" else 0.2, seed, offset)[indices]
            error = float(np.mean((predictions - targets) ** 2))
            errors.append(error)
            rows.append(
                dict(
                    experiment=case, method=method, trial=trial + 1, noise_seed=seed, test_mse=error
                )
            )
        summaries.append(
            dict(
                experiment=case,
                method=method,
                trials=trials,
                mean_test_mse=float(np.mean(errors)),
                std_test_mse=float(np.std(errors, ddof=0)),
            )
        )
    pd.DataFrame(rows).to_csv(output / "trials.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    report = dict(
        protocol="fixed_model_fresh_test_noise",
        trials_per_model=trials,
        seed_rule="10 * zero_based_trial",
        std_ddof=0,
        retraining=False,
        checkpoint_sha256=checkpoints,
        results=summaries,
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Noise statistics: {output.resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="An Example 1 result directory")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.directory, args.trials, args.output)


if __name__ == "__main__":
    main()
