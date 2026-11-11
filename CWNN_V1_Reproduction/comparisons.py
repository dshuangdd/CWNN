import numpy as np
import pandas as pd


def baseline_names(experiment):
    if experiment in ("D1", "D2", "D3", "milling"):
        return [experiment + "_wnn", *[experiment + f"_gnn{depth}" for depth in (3, 5, 7)]]
    return ["stationary_wnn"] if experiment == "stationary" else []


def case_and_method(folder):
    case, _, suffix = folder.partition("_")
    method = "CWNN" if not suffix else "WNN" if suffix == "wnn" else "GNN-" + suffix[3:]
    return case, method


def write_comparison(output, results):

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for item in results:
        row = {
            key: item[key]
            for key in (
                "experiment",
                "epochs",
                "parameters",
                "target",
                "status",
                "last_train_mse_before_update",
                "last_train_mse_after_update",
                "last_test_mse_after_update",
                "seconds",
            )
        }
        row["method"] = item.get("method", "WNN" if item["experiment"].endswith("_wnn") else "CWNN")
        row["experiment"] = row["experiment"].split("_")[0]
        row["seconds_per_update"] = row["seconds"] / row["epochs"]
        for key in ("test_window_mse", "test_trial_mse", "train_trials", "test_trials"):
            if key in item:
                row[key] = item[key]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    for case in ("D1", "D2", "D3", "incremental", "stationary", "changed", "milling"):
        folders = [
            output / name for name in (case, *baseline_names(case)) if (output / name).is_dir()
        ]
        if not folders:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for folder in folders:
            history = pd.read_csv(folder / "history.csv")
            label = case_and_method(folder.name)[1]
            loss_column = "train_mse_after_update" if case == "milling" else "train_mse_before_update"
            axes[0].semilogy(history["update"], history[loss_column], label=label)
            axes[1].step(history["update"], history["parameters"], where="post", label=label)
        axes[0].set(xlabel="Update", ylabel="Training MSE after update" if case == "milling" else "Training MSE before update", title=case)
        axes[1].set(xlabel="Update", ylabel="Trainable parameters", title=case)
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"comparison_{case}.png", dpi=150)
        plt.close(fig)
    if all((output / case / "dataset.npz").exists() for case in ("D1", "D2", "D3")):
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        for case in ("D1", "D2", "D3"):
            with np.load(output / case / "dataset.npz", allow_pickle=False) as data:
                x = np.concatenate((data["x_train"], data["x_test"]))
                y = np.concatenate((data["y_train"], data["y_test"]))
            ax.scatter(x[:, 0], x[:, 1], y, s=10, label=case)
        ax.set(xlabel="x1", ylabel="x2", zlabel="Target", title="Example 1 datasets")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "datasets.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    import sys
    from run import main

    main(["--baselines", *sys.argv[1:]])
