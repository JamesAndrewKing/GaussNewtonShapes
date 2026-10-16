"""
Recompute the metrics of the best and final models of finished runs from their checkpoints, e.g. after adding a metric:

    python -m experiments.evaluate --task catenoid --exp main

Updates "best" and "final" in each run's evals.json. The metrics over time ("over_time") need the run to be repeated.
"""

import argparse
import glob
import json
import os
import torch
from training.residuals import ResidualLibrary
from experiments.run import RESULTS_DIR
from experiments.tasks import TASKS


def evaluate_run(task, run_dir):
    with open(os.path.join(run_dir, "evals.json")) as f:
        evals = json.load(f)
    model = task.model()
    res_lib = ResidualLibrary(model)
    for which in ["best", "final"]:
        model.load_state_dict(torch.load(os.path.join(run_dir, f"{which}.pt")))
        kept = {k: evals[which][k] for k in ["iters", "time", "loss"] if k in evals[which]}
        evals[which] = {**kept, **task.evaluate(model, res_lib, final=True)}
    with open(os.path.join(run_dir, "evals.json"), "w") as f:
        json.dump(evals, f, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--exp", default="main")
    args = parser.parse_args()

    torch.set_default_device('cuda')
    for run_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, args.task, args.exp, "*_seed*"))):
        with open(os.path.join(run_dir, "config.json")) as f:
            task = TASKS[args.task](**json.load(f)["task_args"])
        evaluate_run(task, run_dir)
        print("evaluated", run_dir)
