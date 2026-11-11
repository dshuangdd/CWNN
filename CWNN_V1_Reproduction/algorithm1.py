import argparse
import json
from pathlib import Path

from legacy.initial_resolution import run_example


def plot_energies(results, directory):

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=False)
    for example, cases in results.items():
        for case, result in cases.items():
            trace = result["trace"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([row["resolution"] for row in trace], [row["energy"] for row in trace], "o-")
            ax.set(
                xlabel="Probe resolution",
                ylabel="Coefficient energy",
                title=f"{example} / {case}: archived probe",
            )
            fig.tight_layout()
            fig.savefig(directory / f"{example}_{case}.png", dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("1", "2", "3", "4", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", type=Path, help="New directory for measured energy plots")
    parser.add_argument("--data", type=Path, help="Alternative Example 4 .npz feature file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    examples = (1, 2, 3, 4) if args.example == "all" else (int(args.example),)
    results = {f"example{number}": run_example(number, milling_data=args.data) for number in examples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    if args.plots:
        plot_energies(results, args.plots)
    for name, cases in results.items():
        print(
            name
            + ": "
            + ", ".join(
                f"{case}={result['initial_resolution']} ({result['status']})"
                for case, result in cases.items()
            )
        )
    print(args.output)


if __name__ == "__main__":
    main()
