import argparse
from pathlib import Path

import numpy as np

from milling_data import extract_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original mill.mat")
    parser.add_argument("output", type=Path, help="New .npz feature file")
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix.lower() != ".npz":
        parser.error("Use a new .npz output path")
    data = extract_features(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"Saved {len(data['wear'])} measured trials to {args.output.resolve()}")


if __name__ == "__main__":
    main()
