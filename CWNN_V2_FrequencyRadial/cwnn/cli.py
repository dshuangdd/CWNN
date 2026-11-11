import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from .config import TrainConfig
from .data import InputScaler, load_dataset
from .frequency import estimate_resolution
from .model import CWNN


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def initial_resolution(value):
    if value == "auto":
        return None
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a nonnegative integer or auto") from error
    if number < 0:
        raise argparse.ArgumentTypeError("Resolution must be nonnegative")
    return number


def energy_fraction(value):
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(
            "Use a fraction such as 1/3 or a decimal such as 0.25"
        ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("probe", "fit"):
        sub = commands.add_parser(
            command, help="Algorithm 1" if command == "probe" else "Algorithm 2"
        )
        sub.add_argument(
            "dataset", type=Path, help="NPZ with x_train and y_train; test arrays are optional"
        )
        sub.add_argument("--output", type=Path, help="A new result directory")
        sub.add_argument("--learning-rate", type=float, default=0.005)
        sub.add_argument("--target-mse", type=float, default=0.006)
        sub.add_argument(
            "--basis-family", choices=["frequency-radial", "spatial-radial"],
            default="frequency-radial",
        )
        sub.add_argument("--normalization", choices=["l2", "notebook"])
        sub.add_argument(
            "--probe-steps", type=int, default=10, help="Adam updates per candidate resolution"
        )
        if command == "probe":
            sub.add_argument("--max-resolution", type=int, default=5)
            sub.add_argument("--max-centers", type=int, default=100000)
        else:
            sub.add_argument(
                "--initial-resolution", type=initial_resolution, default=0, metavar="N|auto"
            )
            sub.add_argument(
                "--mu",
                type=energy_fraction,
                default=1 / 3,
                help="Energy fraction, e.g. 1/2, 1/3, 1/4",
            )
            sub.add_argument("--max-epochs", type=int, default=10000)
            sub.add_argument("--max-parameters", type=int, default=100000)
    args = parser.parse_args(argv)
    normalization = args.normalization or (
        "l2" if args.basis_family == "frequency-radial" else "notebook"
    )
    if args.basis_family == "frequency-radial" and normalization != "l2":
        parser.error("frequency-radial requires --normalization l2")
    data = load_dataset(args.dataset)
    scaler = InputScaler()
    x_train = scaler.fit_transform(data.x_train)
    x_test = scaler.transform(data.x_test) if data.x_test is not None else None
    folder = args.output or Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if folder.exists():
        parser.error(f"Output directory already exists: {folder}")
    folder.mkdir(parents=True)
    scaler.save(folder / "scaler.npz")
    write_json(
        folder / "data.json",
        dict(
            source=args.dataset.name,
            dimension=x_train.shape[1],
            train_samples=len(x_train),
            test_samples=0 if x_test is None else len(x_test),
            input_scaling="training Min-Max",
        ),
    )
    with threadpool_limits(limits=1):
        resolution = getattr(args, "initial_resolution", None)
        if args.command == "probe" or resolution is None:
            probe = estimate_resolution(
                x_train,
                data.y_train,
                learning_rate=args.learning_rate,
                target_mse=args.target_mse,
                probe_steps=args.probe_steps,
                max_resolution=getattr(args, "max_resolution", 5),
                max_centers=getattr(args, "max_centers", 100000),
                basis_family=args.basis_family,
                normalization=normalization,
            )
            write_json(folder / "probe.json", probe)
            print(
                f"Algorithm 1: {probe['status']}; initial resolution = {probe['initial_resolution']}"
            )
            if args.command == "probe":
                print(f"Results: {folder.resolve()}")
                return 0
            resolution = probe["initial_resolution"]
            if resolution is None:
                print(
                    "No estimate within the probe budget. Inspect probe.json, then set --initial-resolution explicitly or adjust the standalone probe settings."
                )
                return 2
        config = TrainConfig(
            initial_resolution=resolution,
            mu=args.mu,
            learning_rate=args.learning_rate,
            target_mse=args.target_mse,
            max_epochs=args.max_epochs,
            max_parameters=args.max_parameters,
            basis_family=args.basis_family,
            normalization=normalization,
        )
        model = CWNN(dimension=x_train.shape[1], config=config)
        metrics = model.fit(x_train, data.y_train, x_test, data.y_test)
        model.save(folder / "model.npz")
        predictions = dict(train=model.predict(x_train))
        if x_test is not None:
            predictions["test"] = model.predict(x_test)
        np.savez_compressed(folder / "predictions.npz", **predictions)
    write_json(folder / "config.json", asdict(config))
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "events.json", model.events)
    with (folder / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["epoch", "phase", "train_mse", "test_mse", "parameters"]
        )
        writer.writeheader()
        writer.writerows(model.history)
    print(json.dumps(metrics))
    print(f"Results: {folder.resolve()}")
    return 0
