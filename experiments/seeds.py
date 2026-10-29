"""
Statistics over repeated seeds of the runs in <task>/main (no training), e.g.

    python -m experiments.seeds --tasks catenoid enneper cone rockerarm bunny jeb

Writes per task to paper_figures/:
    <task>_loss_seeds.png, <task>_chamfer_seeds.png    each seed's curve as the median in time bins of ~3 s; the median
                                                        over the seeds that reach the target surface (line), their range
                                                        (band), and the other seeds (dashed)
and for all tasks together:
    paper_figures/seeds_overview.png                    best Chamfer distance (JEB: loss) of every seed, per task and optimizer
    experiments/results/seeds.md                        median [interquartile range] of the best models and outcome counts
The figures show the optimizers of the paper figures (task.figure_opts); the table shows all optimizers with runs.
On the minimal and developable surfaces (exact ground truth), each run is classified by outcome() into reaching the
target surface, an alternative solution of the loss, or a failure. The other tasks have no such classification;
artifacts such as spurious sheets show up in the Hausdorff distance.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from experiments.run import RESULTS_DIR
from experiments.tasks import TASKS
from experiments.plot import LABELS, COLORS, load_runs
from experiments.figures import PAPER_FIGURES_DIR

FAIL_CD = 1e-2
TITLES = {"rockerarm": "Rocker arm", "jeb": "JEB (loss)"}
SEEDS = {"jeb": [0, 1, 2]}           ## default: 0-4
RUN_SUFFIX = {"jeb": "_pretrainedown"}  ## the JEB seeds are pretrained per seed
TABLE_KEYS = {"rockerarm": ["loss", "cd", "hausdorff"], "bunny": ["loss", "cd", "hausdorff"],
              "jeb": ["loss", "interface_rms", "design_region_violation", "n_components"]}  ## default: loss, cd


def load_seeds(task_name):
    """{opt: [(log, evals), ...]} over the seeds of the task, for the optimizers that have runs."""
    exp_dir = os.path.join(RESULTS_DIR, task_name, "main")
    suffix = RUN_SUFFIX.get(task_name, "")
    runs = {}
    for opt in LABELS:
        seed_runs = [load_runs(exp_dir, [opt + suffix], s).get(opt + suffix) for s in SEEDS.get(task_name, range(5))]
        if any(seed_runs):
            runs[opt] = [r for r in seed_runs if r is not None]
    return runs


def outcome(task_name, evals):
    """
    Minimal and developable surfaces: "target" if the best model is close to the true surface; otherwise "alternative"
    if it still contains the boundary curve (another solution of the same loss, e.g. the planes through the catenoid's
    rings), else "failed" (e.g. a collapsed f whose zero set misses the boundary). Other tasks: always "target".
    """
    m = evals["best"]
    if not hasattr(TASKS[task_name], "curvature") or m["cd"] <= FAIL_CD:
        return "target"
    return "alternative" if m["interface_rms"] <= FAIL_CD else "failed"


def binned(t, v, edges):
    """
    Median of log10(v) in each time bin (the loss is logged at every iteration and oscillates, so single samples are
    noise); empty bins are interpolated, and bins after the last finite value are NaN (diverged runs).
    """
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    centers = (edges[:-1] + edges[1:]) / 2
    if not ok.any():
        return np.full_like(centers, np.nan)
    t, y = t[ok], np.log10(v[ok])
    idx = np.digitize(t, edges) - 1
    med = np.array([np.median(y[idx == b]) if (idx == b).any() else np.nan for b in range(len(centers))])
    has = np.isfinite(med)
    med = np.interp(centers, centers[has], med[has])
    med[centers > t[-1]] = np.nan
    return med


def plot_seeds_over_time(task_name, runs, key, ylabel, paths, opts, n_bins=400):
    budget = max(log["time"][-1] for seed_runs in runs.values() for log, _ in seed_runs)
    edges = np.linspace(0, budget, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for opt in opts:
        curves, ok = [], []
        for log, evals in runs[opt]:
            t, v = (log["time"], log["loss"]) if key == "loss" else zip(*[(e["time"], e[key]) for e in evals["over_time"]])
            curves.append(binned(t, v, edges))
            ok.append(outcome(task_name, evals) == "target")
            if not ok[-1]:
                ax.plot(centers, 10 ** curves[-1], c=COLORS[opt], lw=0.6, ls="--", alpha=0.6)
        good = np.array(curves)[ok]
        if len(good):
            ax.fill_between(centers, 10 ** np.nanmin(good, axis=0), 10 ** np.nanmax(good, axis=0),
                            color=COLORS[opt], alpha=0.25, lw=0)
            ax.plot(centers, 10 ** np.nanmedian(good, axis=0), c=COLORS[opt], label=LABELS[opt])
    ax.set_xlabel("Time in [s]")
    ax.set_ylabel(ylabel)
    ax.semilogy()
    ax.legend(loc="upper right")
    plt.tight_layout()
    for path in paths:
        plt.savefig(path, dpi=300)
    plt.close(fig)


def overview(all_runs, path):
    fig, axs = plt.subplots(1, len(all_runs), figsize=(1.7 * len(all_runs), 2.8))
    rng = np.random.default_rng(0)
    for ax, (task_name, runs) in zip(np.atleast_1d(axs), all_runs.items()):
        key = "cd" if "cd" in next(iter(runs.values()))[0][1]["best"] else "loss"
        for i, (opt, seed_runs) in enumerate(runs.items()):
            v = [evals["best"][key] for _, evals in seed_runs]
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=12, color=COLORS[opt])
        if hasattr(TASKS[task_name], "curvature"):
            ax.axhline(FAIL_CD, c="gray", ls="--", lw=0.8)
        ax.set_xticks(range(len(runs)), [LABELS[opt] for opt in runs], rotation=45)
        ax.set_xlim(-0.6, len(runs) - 0.4)
        ax.semilogy()
        ax.set_title(TITLES.get(task_name, task_name.capitalize()), fontsize=10)
    np.atleast_1d(axs)[0].set_ylabel("Chamfer distance")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def seeds_table(task_name, runs):
    keys = TABLE_KEYS.get(task_name, ["loss", "cd"])
    rows = [f"### {task_name}\n", "| | seeds | alternative | failed | " + " | ".join(keys) + " |",
            "|---" * (len(keys) + 4) + "|"]
    for opt, seed_runs in runs.items():
        cells = []
        for k in keys:
            v = [evals["best"][k] for _, evals in seed_runs]
            fmt = "{:g}" if all(isinstance(x, int) for x in v) else "{:.1e}"  ## counts, e.g. n_components
            cells.append("{} [{}, {}]".format(*(fmt.format(q) for q in np.percentile(v, [50, 25, 75]))))
        outcomes = [outcome(task_name, evals) for _, evals in seed_runs]
        counts = [outcomes.count(o) if hasattr(TASKS[task_name], "curvature") else "-" for o in ["alternative", "failed"]]
        rows.append(f"| {LABELS[opt]} | {len(seed_runs)} | {counts[0]} | {counts[1]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    args = parser.parse_args()

    all_runs = {t: load_seeds(t) for t in args.tasks}
    ## The figures show the optimizers of the paper figures (task.figure_opts), the table all optimizers with runs
    figure_runs = {t: {opt: runs[opt] for opt in TASKS[t].figure_opts if opt in runs} for t, runs in all_runs.items()}
    for task_name, runs in figure_runs.items():
        out = lambda name: [os.path.join(PAPER_FIGURES_DIR, f"{task_name}_{name}_seeds.png")]
        plot_seeds_over_time(task_name, runs, "loss", "Loss", out("loss"), list(runs))
        if "cd" in next(iter(runs.values()))[0][1]["best"]:
            plot_seeds_over_time(task_name, runs, "cd", "Chamfer distance", out("chamfer"), list(runs))
    overview(figure_runs, os.path.join(PAPER_FIGURES_DIR, "seeds_overview.png"))
    table = "## Best models over seeds: median [interquartile range]\n\n" + \
            f"Minimal and developable surfaces: a best Chamfer distance above {FAIL_CD:g} is an alternative solution " + \
            f"if the interface error stays below {FAIL_CD:g} (the surface still contains the boundary), " + \
            "else a failure; all runs are included in the statistics.\n\n" + \
            "\n\n".join(seeds_table(t, runs) for t, runs in all_runs.items()) + "\n"
    with open(os.path.join(RESULTS_DIR, "seeds.md"), "w") as f:
        f.write(table)
    print(table)
