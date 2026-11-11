import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def settings_for(name, profile):
    if profile not in ("paper", "archive"):
        raise ValueError("profile must be paper or archive")
    settings = {}
    if name in ("D1", "D2", "D3"):
        settings["noise"] = {"D1": 0.0, "D2": 0.1, "D3": 0.2}[name]
        settings.update(mu=1 / 3, pre_ac=0.025 if name == "D3" else 0.006)
        if profile == "paper":
            settings.update(learn_rate=0.0005, offset=0.5)
        else:
            settings.update(learn_rate=0.0001, offset=-1.0)
    elif name == "incremental" and profile == "paper":
        settings.update(learn_rate=0.0005, offset=0.5)
    elif name in ("milling", "milling_wnn"):
        settings["data_path"] = str(ROOT / "data/milling_features.npz")
        settings.update(j_init=0, mu=1 / 3, learn_rate=0.001)
        if name == "milling_wnn":
            settings.update(j_init=1, mu=1, learn_rate=0.0001)
        elif profile == "archive":
            settings.update(j_init=0, mu=1 / 3, learn_rate=0.0001)
    return settings


def execute(name, profile="archive", accelerated=True, max_epochs=10000, settings_overrides=None):
    if type(max_epochs) is not int or max_epochs < 1:
        raise ValueError("max_epochs must be a positive integer")
    if "_gnn" in name:
        from legacy.gnn import run as run_gnn

        experiment, depth = name.split("_gnn")
        data_path = (settings_overrides or {}).get("data_path", ROOT / "data/milling_features.npz")
        return run_gnn(
            experiment, int(depth), data_path, max_epochs,
            profile=profile, offset=(settings_overrides or {}).get("offset"),
        )
    if name.endswith("_wnn") and name != "milling_wnn":
        overrides = {"mu": 1, "j_init": 1}
        if name[:-4] in ("D1", "D2", "D3"):
            overrides["learn_rate"] = 0.0001
        overrides.update(settings_overrides or {})
        summary, arrays, state = execute(
            name[:-4],
            profile,
            accelerated,
            max_epochs,
            overrides,
        )
        summary.update(experiment=name, method="WNN")
        return summary, arrays, state
    kind = (
        "offline" if name in ("D1", "D2", "D3") else ("milling" if name == "milling_wnn" else name)
    )
    module = importlib.import_module("legacy." + kind)
    settings = settings_for(name, profile)
    settings.update(settings_overrides or {})
    probe = None
    if name == "milling" and "j_init" not in (settings_overrides or {}):
        from legacy.initial_resolution import run_example

        probe = run_example(4, milling_data=settings["data_path"])["milling"]
        if probe["initial_resolution"] is None:
            raise RuntimeError("Algorithm 1 did not select an initial resolution; inspect its energy trace.")
        settings["j_init"] = probe["initial_resolution"]
    begin = time.perf_counter()
    with threadpool_limits(limits=1):
        state = module.run(settings, accelerated=accelerated, max_epochs=max_epochs)
    seconds = time.perf_counter() - begin
    if kind == "milling":
        model = state["model"]
        losses = np.array(state["history"]["train_loss"])
        tests = np.array(state["history"]["test_loss"])
        parameters = np.array(state["history"]["params_count"])
        weights = np.concatenate([w.detach().cpu().numpy() for w in model.weights])
        with __import__("torch").no_grad():
            post = float(((model(state["X_train"]) - state["y_train"]) ** 2).mean())
        target = state["config"]["pre_ac"]
    else:
        losses = np.asarray(state["losses"] if kind == "offline" else state["all_losses"])
        tests = np.asarray(state.get("test_losses", []))
        weights = np.asarray(state["w_need"])
        parameters = np.asarray(state.get("parameter_counts", []))
        target = state["pre_ac"]
        network = state.get("network_adding", state["network"])
        j = state["j"] - 1 if "network_adding" in state else state["j_init"]
        x = state["data_train"] if kind == "offline" else state["data"]
        y = state["all_y_trues_train"] if kind == "offline" else state["all_y_trues"]
        if kind in ("stationary", "changed"):
            start = state["epoch"] - 1
            x, y = x[:, start : start + 10], y[start : start + 10]
        post = float(np.mean((network.feedforward(x, j) - y) ** 2))
    status = "target_reached" if len(losses) and losses[-1] < target else "budget_or_stream_limit"
    phase_details = {}
    if kind == "incremental":
        calls = state["accelerator"].training_calls
        current = calls[-1]


        complete = (
            current["updates"] > 0
            and current["samples"] == state["data_len"]
            and current["last_mse_before_update"] < target
        )
        status = "target_reached" if complete else "budget_or_stream_limit"
        last_update = next(call for call in reversed(calls) if call["updates"])
        phase_details = dict(
            training_calls=calls,
            current_dataset_samples=x.shape[1],
            last_update_dataset_samples=last_update["samples"],
        )
    if kind == "milling":
        status = "target_reached" if post < target else "budget_or_stream_limit"
    elif status == "target_reached" and post >= target:
        status = "archived_stop_only"
    summary = dict(
        experiment=name,
        profile=profile,
        accelerated=accelerated,
        status=status,
        epochs=len(losses),
        parameters=len(weights),
        target=target,
        final_model_target_reached=post < target,
        last_train_mse_before_update=float(losses[-1]),
        last_train_mse_after_update=post,
        last_test_mse_after_update=float(tests[-1]) if len(tests) else None,
        seconds=seconds,
        cache_hits=state["accelerator"].hits,
        cache_misses=state["accelerator"].misses,
        settings={k: v for k, v in settings.items() if k != "data_path"},
    )
    summary.update(phase_details)
    if kind == "offline":
        summary["split"] = "shared_80_20"
    if kind == "milling":
        summary["active_parameters"] = sum(
            (2 if i == 0 else 1) * len(indices) for i, indices in enumerate(model.active_indices)
        )
        summary["stopping_loss"] = "train_mse_after_update"
        from milling_data import evaluation

        state["milling_data"] = state["processor"].data
        with __import__("torch").no_grad():
            predictions = model(state["X_test"]).numpy()
        summary.update(evaluation(predictions, state["milling_data"]))
        if probe is not None:
            state["initial_resolution_probe"] = probe
            summary["initial_resolution"] = probe["initial_resolution"]
    arrays = dict(losses=losses, test_losses=tests, parameters=parameters, weights=weights)
    if kind == "milling":
        arrays["train_losses_after_update"] = np.asarray(
            state["history"]["train_loss_after_update"]
        )
    return summary, arrays, state


def save_model(path, state):
    arrays = {}
    if "gnn" in state:
        meta = dict(backend="gnn", input_size=state["input_size"], num_blocks=state["num_blocks"])
        arrays = {
            "state_" + key: value.detach().cpu().numpy()
            for key, value in state["gnn"].state_dict().items()
        }
        arrays.update(mean=state["mean"].numpy(), std=state["std"].numpy())
    elif "model" in state:
        model = state["model"]
        meta = dict(backend="torch", initial_resolution=model.j, layers=len(model.weights))
        for i, weights in enumerate(model.weights):
            arrays[f"weights_{i}"] = weights.detach().cpu().numpy()
            arrays[f"centers_{i}"] = model.centers_list[i][model.active_indices[i]].cpu().numpy()
    else:
        model = state.get("network_adding", state["network"])
        grown = "network_adding" in state
        meta = dict(
            backend="numpy",
            initial_resolution=state["j_init"],
            parent_resolution=state["j"] - 1 if grown else state["j_init"],
            grown=grown,
        )
        arrays.update(weights_0=model.w1, centers_0=model.k)
        if grown:
            arrays.update(weights_1=model.w2, centers_1=model.k2)
    np.savez_compressed(path, metadata=np.array(json.dumps(meta)), **arrays)


def write_result(directory, summary, arrays, state):
    directory.mkdir(parents=True, exist_ok=False)
    if not len(arrays["parameters"]) and "accelerator" in state:
        arrays = dict(
            arrays,
            parameters=np.concatenate(
                [
                    np.full(call["updates"], call["parameters"], dtype=int)
                    for call in state["accelerator"].training_calls
                ]
            ),
        )
    (directory / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    save_model(directory / "model.npz", state)
    if "gnn" in state:
        dataset = {
            key: state[key]
            for key in ("x_train", "x_test", "y_train", "y_test", "train_indices", "test_indices")
        }
    elif "data_test" in state:
        dataset = dict(
            x_train=state["data_train"][0],
            x_test=state["data_test"][0],
            y_train=state["all_y_trues_train"],
            y_test=state["all_y_trues_test"],
            train_indices=state["train_indices"],
            test_indices=state["test_indices"],
        )
    elif "X_test" in state:
        dataset = {
            out: state[key].detach().cpu().numpy()
            for out, key in (
                ("x_train", "X_train"),
                ("x_test", "X_test"),
                ("y_train", "y_train"),
                ("y_test", "y_test"),
            )
        }
    else:
        dataset = dict(x=state["data"][0], y=state["all_y_trues"])
    if "milling_data" in state:
        dataset.update(state["milling_data"])
        from predict import predict

        windows = predict(directory / "model.npz", dataset["x_test"])
        np.savez_compressed(
            directory / "predictions.npz", windows=windows,
            trial_prediction=windows.reshape(-1, 4).mean(axis=1),
            trial_target=dataset["y_test"].reshape(-1, 4)[:, 0],
            trial_ids=dataset["test_trial_ids"],
        )
    if "initial_resolution_probe" in state:
        (directory / "algorithm1.json").write_text(
            json.dumps(state["initial_resolution_probe"], indent=2), encoding="utf-8"
        )
    np.savez_compressed(directory / "dataset.npz", **dataset)
    import pandas as pd

    history = {
        "update": np.arange(1, len(arrays["losses"]) + 1),
        "train_mse_before_update": arrays["losses"],
    }
    for source, column in (
        ("test_losses", "test_mse_after_update"),
        ("parameters", "parameters"),
        ("train_losses_after_update", "train_mse_after_update"),
    ):
        if source in arrays and len(arrays[source]) == len(arrays["losses"]):
            history[column] = arrays[source]
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(
        np.arange(1, len(arrays["losses"]) + 1), arrays["losses"], label="Train: before update"
    )
    if "train_losses_after_update" in arrays:
        ax.semilogy(
            np.arange(1, len(arrays["losses"]) + 1),
            arrays["train_losses_after_update"],
            label="Train: after update",
        )
    if len(arrays["test_losses"]):
        ax.semilogy(
            np.arange(1, len(arrays["test_losses"]) + 1),
            arrays["test_losses"],
            label="Test: after update",
        )
    ax.axhline(summary["target"], color="grey", linestyle="--", label="Target")
    ax.set(xlabel="Update", ylabel="MSE", title=f"{summary['experiment']} / {summary['profile']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "loss.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--profile", choices=["paper", "archive"], default="archive")
    parser.add_argument(
        "--baselines", action="store_true", help="Include archived WNN and GNN comparisons"
    )
    parser.add_argument(
        "--robustness-trials",
        type=int,
        default=0,
        help="Fresh test-noise realizations per D2/D3 model; use 100 for Table IV",
    )
    parser.add_argument(
        "--output", type=Path, help="New output directory; default: results/TIMESTAMP"
    )
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--data", type=Path, help="Alternative milling .npz feature file")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.output.exists() or args.max_epochs < 1:
        parser.error("Use a new output directory and a positive epoch budget")
    if args.robustness_trials < 0 or (args.robustness_trials and args.example not in ("1", "all")):
        parser.error("Noise statistics require Example 1 and a nonnegative trial count")
    names = {
        "1": ["D1", "D2", "D3"],
        "2": ["incremental"],
        "3": ["stationary", "changed"],
        "4": ["milling"],
    }
    selected = sum(names.values(), []) if args.example == "all" else names[args.example]
    if args.baselines:
        from comparisons import baseline_names

        selected = [item for name in selected for item in (name, *baseline_names(name))]
    results = []
    for name in selected:
        overrides = (
            {"data_path": str(args.data)} if args.data and name.startswith("milling") else None
        )
        summary, arrays, state = execute(
            name, args.profile, not args.no_cache, args.max_epochs, overrides
        )
        write_result(args.output / name, summary, arrays, state)
        results.append(summary)
        print(json.dumps(summary), flush=True)
    if args.baselines:
        from comparisons import write_comparison

        write_comparison(args.output, results)
    if args.robustness_trials:
        from robustness import evaluate

        evaluate(args.output, args.robustness_trials)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output / "environment.json").write_text(
        json.dumps(
            dict(
                python=platform.python_version(), numpy=np.__version__, platform=platform.platform()
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
