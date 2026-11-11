import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sequences(samples=3000):

    if samples < 3:
        raise ValueError("At least three samples are required")
    stationary = np.ones(samples)
    changed = np.ones(samples)
    for step in range(2, samples):
        first = np.pi * (stationary[step - 1] ** 2 + stationary[step - 2] ** 2)
        second = np.pi * (changed[step - 1] ** 2 + changed[step - 2] ** 2)
        stationary[step] = np.arctan(first) ** 0.5
        changed[step] = np.sqrt(np.arctan(second)) + np.cos(second)
    return stationary, changed


def compute_spectrum(signal):

    half = len(signal) // 2
    return np.fft.fftfreq(len(signal), d=1)[:half], np.abs(np.fft.fft(signal)[:half])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path("results") / ("spectrum_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output.exists() or args.samples < 3:
        parser.error("Use a new output directory and at least three samples")
    stationary, changed = generate_sequences(args.samples)
    frequency, first = compute_spectrum(stationary)
    _, second = compute_spectrum(changed)
    output.mkdir(parents=True)
    pd.DataFrame({"equation_30": stationary, "equation_31": changed}).to_csv(
        output / "sequences.csv", index=False
    )
    pd.DataFrame({"frequency": frequency, "equation_30": first, "equation_31": second}).to_csv(
        output / "spectrum.csv", index=False
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(frequency, first, "b-", linewidth=1, label="Function (30)")
    ax.semilogy(frequency, second, "r--", linewidth=0.8, label="Function (31)")
    ax.set(xlabel="Frequency (cycles per sample)", ylabel="Magnitude", xlim=(0, 0.5))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "spectrum.png", dpi=180)
    fig.savefig(output / "spectrum.pdf")
    plt.close(fig)
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
