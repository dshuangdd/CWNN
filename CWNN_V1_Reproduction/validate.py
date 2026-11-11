import argparse
import json
from pathlib import Path
import numpy as np
from run import execute, save_model
from predict import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["paper", "archive"], default="archive")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)
    rows = []
    for name in ["D1", "D2", "D3", "incremental", "stationary", "changed", "milling"]:
        slow, a, _ = execute(name, args.profile, accelerated=False)
        fast, b, state = execute(name, args.profile, accelerated=True)
        error = {}
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{name}: {key}")
            error[key] = float(np.max(np.abs(a[key] - b[key]))) if len(a[key]) else 0.0
        model_path = args.output / (name + "_model.npz")
        save_model(model_path, state)
        if name == "milling":
            import torch

            x = state["X_test"].numpy()
            with torch.no_grad():
                expected = state["model"](state["X_test"]).numpy()
        else:
            x = state["data"][0][:7]
            model = state.get("network_adding", state["network"])
            j = state["j"] - 1 if "network_adding" in state else state["j_init"]
            expected = model.feedforward(x[None], j)
        np.testing.assert_allclose(predict(model_path, x), expected, rtol=2e-6, atol=2e-7)
        row = dict(
            experiment=name,
            profile=args.profile,
            errors=error,
            seconds_uncached=slow["seconds"],
            seconds_cached=fast["seconds"],
            speedup=slow["seconds"] / fast["seconds"],
            result=fast,
        )
        rows.append(row)
        (args.output / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
