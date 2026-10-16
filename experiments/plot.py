"""
Compare the optimizers on one task, e.g.

    python -m experiments.plot --task catenoid --seed 0 --paper

Writes to experiments/results/<task>/figures/:
    loss_over_time.png, cd_over_time.png, cd1_paper_over_time.png  (--paper overlays the stored paper runs, dashed)
    metrics.md                                                    final metrics of each optimizer
    k3d_plot_heatmap_<opt>.html                                   curvature of the best surface
"""

import argparse
import json
import os
import k3d
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import vmap
from training.residuals import ResidualLibrary
from util.visualization.utils_mesh import get_mesh
from experiments.run import RESULTS_DIR
from experiments.tasks import TASKS

LABELS = {"adam": "Adam", "lbfgs": "LBFGS", "gn": "GN"}
COLORS = {opt: plt.get_cmap('tab10')(i) for i, opt in enumerate(LABELS)}
TABLE_KEYS = ["iters", "loss", "cd", "cd_t2m", "cd_m2t", "hausdorff", "interface_rms", "H_rings_rms", "H_rings_max", "area_err_mesh"]


def split_name(name):
    """'lbfgs_notol' -> ('lbfgs', 'LBFGS (notol)')"""
    opt, _, tag = name.partition("_")
    return opt, LABELS[opt] + (f" ({tag})" if tag else "")


def load_runs(task_dir, names, seed):
    runs = {}
    for name in names:
        run_dir = os.path.join(task_dir, f"{name}_seed{seed}")
        if os.path.exists(os.path.join(run_dir, "evals.json")):
            with open(os.path.join(run_dir, "log.json")) as f, open(os.path.join(run_dir, "evals.json")) as g:
                runs[name] = json.load(f), json.load(g)
    return runs


def plot_over_time(runs, key, ylabel, path, paper_file=None):
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
    plt.savefig(path, dpi=300)
    plt.close(fig)


def metrics_table(runs, which="final"):
    rows = ["| | " + " | ".join(TABLE_KEYS) + " |", "|---" * (len(TABLE_KEYS) + 1) + "|"]
    for name, (_, evals) in runs.items():
        m = evals[which]
        rows.append(f"| {split_name(name)[1]} | " + " | ".join(str(m.get(k, "-")) if k == "iters" else f"{m[k]:.2e}" for k in TABLE_KEYS) + " |")
    return "\n".join(rows)


def curvature_heatmap(task, run_dir, path, color_range=(0, 0.005)):
    model = task.model()
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt")))
    verts, faces = get_mesh(
        lambda x: model(x.double()), N=256, device='cuda',
        bbox_min=torch.tensor([-2, -2, -1.25], dtype=torch.float64),
        bbox_max=torch.tensor([2, 2, 1.25], dtype=torch.float64),
        chunks=2,
    )
    res_lib = ResidualLibrary(model)
    mean_curvatures = vmap(res_lib._mean_curvature, in_dims=(None, 0))(model.params, torch.tensor(verts, dtype=torch.float64)).squeeze()
    fig = k3d.plot(height=800, grid_visible=False, camera_fov=1.0)
    fig += k3d.mesh(
        verts.astype(np.float32), faces.astype(np.uint32),
        attribute=mean_curvatures.abs().detach().cpu().numpy().astype(np.float32),
        color_range=list(color_range), color_map=k3d.basic_color_maps.Jet,
        side='double', flat_shading=False,
    )
    fig += k3d.points(task.pts_boundary.cpu().numpy().astype(np.float32), color=0x00ff00, point_size=0.05)
    with open(path, "w") as f:
        f.write(fig.get_snapshot())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", nargs="*", default=list(LABELS), help="run names <opt>[<tag>], e.g. gn lbfgs_notol")
    parser.add_argument("--paper", action="store_true", help="overlay the curves stored for the paper")
    args = parser.parse_args()

    torch.set_default_device('cuda')
    task = TASKS[args.task]()
    task_dir = os.path.join(RESULTS_DIR, args.task)
    fig_dir = os.path.join(task_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    runs = load_runs(task_dir, args.runs, args.seed)
    paper = (lambda metric: task.paper_data.format(metric, "{}")) if args.paper else (lambda metric: None)

    plot_over_time(runs, "loss", "Loss", os.path.join(fig_dir, "loss_over_time.png"), paper("loss"))
    plot_over_time(runs, "cd", "Two-sided Chamfer", os.path.join(fig_dir, "cd_over_time.png"))
    plot_over_time(runs, "cd1_paper", "$CD_1$", os.path.join(fig_dir, "cd1_paper_over_time.png"), paper("chamf"))

    table = f"## Final model\n\n{metrics_table(runs, 'final')}\n\n## Best model (lowest loss)\n\n{metrics_table(runs, 'best')}\n"
    with open(os.path.join(fig_dir, "metrics.md"), "w") as f:
        f.write(table)
    print(table)

    for name in runs:
        curvature_heatmap(task, os.path.join(task_dir, f"{name}_seed{args.seed}"), os.path.join(fig_dir, f"k3d_plot_heatmap_{name}.html"))
