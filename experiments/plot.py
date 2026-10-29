"""
Compare runs of one task against each other and against the curves stored for the paper, e.g.

    python -m experiments.plot --task catenoid --paper
    python -m experiments.plot --task catenoid --exp archive/lbfgs_settings --runs lbfgs_notebook lbfgs_defaults

Writes to experiments/results/<task>/<exp>/comparison/:
    loss_over_time.png, cd_over_time.png, cd1_paper_over_time.png  (--paper overlays the stored paper runs, dashed)
    metrics.md                                                    metrics of the best and final models
Paper figures are made from the same runs by experiments/figures.py.
"""

import argparse
import json
import os
import torch
import matplotlib.pyplot as plt
from experiments.run import RESULTS_DIR
from experiments.tasks import TASKS

LABELS = {"adam": "Adam", "lbfgs": "LBFGS", "gn": "GN",
          "lm": "LM", "adam_lbfgs": r"Adam $\to$ LBFGS", "adam_gn": r"Adam $\to$ GN"}
COLORS = {opt: plt.get_cmap('tab10')(i) for i, opt in enumerate(LABELS)}


def split_name(name):
    """'lbfgs_notebook' -> ('lbfgs', 'LBFGS (notebook)'); the longest optimizer name wins ('adam_lbfgs' is a hybrid)"""
    opt = max((o for o in LABELS if name == o or name.startswith(o + "_")), key=len)
    tag = name[len(opt) + 1:]
    return opt, LABELS[opt] + (f" ({tag})" if tag else "")


def load_runs(exp_dir, names, seed):
    runs = {}
    for name in names:
        run_dir = os.path.join(exp_dir, f"{name}_seed{seed}")
        if os.path.exists(os.path.join(run_dir, "evals.json")):
            with open(os.path.join(run_dir, "log.json")) as f, open(os.path.join(run_dir, "evals.json")) as g:
                runs[name] = json.load(f), json.load(g)
    return runs


def plot_over_time(runs, key, ylabel, paths, paper_file=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for name, (log, evals) in runs.items():
        opt, label = split_name(name)
        if key == "loss":
            t, v = log["time"], log["loss"]
        else:
            t, v = zip(*[(e["time"], e[key]) for e in evals["over_time"]])
        ax.plot(t, v, label=label, c=COLORS[opt], ls='-' if label == LABELS[opt] else ':')
        if paper_file and label == LABELS[opt] and os.path.exists(paper_file.format(opt)):
            stored = torch.load(paper_file.format(opt), weights_only=False)
            ax.plot(list(stored.keys()), list(stored.values()), c=COLORS[opt], ls='--', lw=0.8, alpha=0.7)
    if paper_file:
        ax.plot([], [], c='gray', ls='--', lw=0.8, label="paper run")
    ax.set_xlabel("Time in [s]")
    ax.set_ylabel(ylabel)
    ax.semilogy()
    ax.legend(loc="upper right")
    plt.tight_layout()
    for path in paths:
        plt.savefig(path, dpi=300)
    plt.close(fig)


def metrics_table(runs, which, keys):
    rows = ["| | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, (_, evals) in runs.items():
        m = evals[which]
        rows.append(f"| {split_name(name)[1]} | " + " | ".join(f"{m[k]:.2e}" if k in m else "-" for k in keys) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--exp", default="main")
    parser.add_argument("--runs", nargs="*", default=list(LABELS), help="run names <opt><tag>, e.g. gn lbfgs_notebook")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paper", action="store_true", help="overlay the curves stored for the paper")
    args = parser.parse_args()

    exp_dir = os.path.join(RESULTS_DIR, args.task, args.exp)
    out_dir = os.path.join(exp_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    runs = load_runs(exp_dir, args.runs, args.seed)
    paper = lambda metric: TASKS[args.task].paper_data.format(metric, "{}") if args.paper else None

    plot_over_time(runs, "loss", "Loss", [os.path.join(out_dir, "loss_over_time.png")], paper("loss"))
    if all("cd" in evals["best"] for _, evals in runs.values()):
        plot_over_time(runs, "cd", "Two-sided Chamfer", [os.path.join(out_dir, "cd_over_time.png")])
    if all("cd1_paper" in evals["best"] for _, evals in runs.values()):
        plot_over_time(runs, "cd1_paper", "$CD_1$", [os.path.join(out_dir, "cd1_paper_over_time.png")], paper("chamf"))

    keys = TASKS[args.task].table_keys
    table = f"## Best model (lowest loss)\n\n{metrics_table(runs, 'best', keys)}\n\n## Final model\n\n{metrics_table(runs, 'final', keys)}\n"
    with open(os.path.join(out_dir, "metrics.md"), "w") as f:
        f.write(table)
    print(table)
