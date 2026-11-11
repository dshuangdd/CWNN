import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from legacy.frequency import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    with threadpool_limits(limits=1):
        state = run()
    result = dict(
        energy=state["energy"],
        smoothed_energy=state["energy_title"],
        last_tested_resolution=state["j"] - 1,
        archived_return_resolution=state["j"] - 3,
        stopped_on_energy=bool(state["stop_con"]),
        note="This is the archived 2D Algorithm 1 probe; experiment configurations are explicit.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
