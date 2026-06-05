import json
from pathlib import Path


def summarize_runs(run_dir="runs"):
    files = Path(run_dir).glob("run_*.json")

    all_pcs = []
    for f in files:
        data = json.load(open(f))
        if "final" in data:
            all_pcs.append(data["final"]["PCS"])

    if all_pcs:
        print("Average PCS:", sum(all_pcs)/len(all_pcs))


if __name__ == "__main__":
    summarize_runs()