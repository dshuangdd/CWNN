import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from run import execute, write_result


def run_cases(profile="archive", max_epochs=10000):

    for denominator in (2, 3, 4, 5):
        for target in (0.0025, 0.01, 0.02):
            summary, arrays, state = execute(
                "D1", profile, max_epochs=max_epochs,
                settings_overrides={"mu": 1 / denominator, "pre_ac": target},
            )
            name = f"mu_1_{denominator}_target_{target}"
            yield name, summary, arrays, state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("archive", "paper"), default="archive")
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path("results") / ("table_i_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output.exists() or args.max_epochs < 1:
        parser.error("Use a new output directory and a positive epoch budget")
    rows = []
    for name, summary, arrays, state in run_cases(args.profile, args.max_epochs):
        write_result(output / name, summary, arrays, state)
        rows.append({
            "case": name, "profile": args.profile,
            "mu": summary["settings"]["mu"], "target": summary["target"],
            "parameters": summary["parameters"], "epochs": summary["epochs"],
            "status": summary["status"],
            "last_train_mse_before_update": summary["last_train_mse_before_update"],
            "last_train_mse_after_update": summary["last_train_mse_after_update"],
            "test_mse": summary["last_test_mse_after_update"],
        })
        print(f"{name}: {summary['parameters']} parameters, {summary['epochs']} updates", flush=True)
    pd.DataFrame(rows).to_csv(output / "table_i.csv", index=False)
    (output / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
