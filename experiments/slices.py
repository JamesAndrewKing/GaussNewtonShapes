"""
The model f on the plane y = 0 over training time (Section 3.1), from runs with --snapshots, e.g.

    python -m experiments.slices --task catenoid --exp slices --runs adam_seed0 lbfgs_seed0 gn_seed0 --times 0 1 5 20 100 300

One row per run, one column per time (the snapshot closest to it). Each panel shows f (colour; fading to white means f
shrinks towards 0), its zero set (black), the true surface (dashed), other solutions (dotted) and the boundary points
in the plane (dots). The plane is task.slice_box ((x0, z0), (x1, z1)) if the task has one, else task.mesh_bbox in x and z.
Writes experiments/results/<task>/<exp>/<name>.png, or paper_figures/<task>_<name>.png with --paper (default name: slices).
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
from experiments.run import RESULTS_DIR
from experiments.tasks import TASKS
from experiments.plot import split_name

PAPER_FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_figures")


def slices_figure(task, run_dirs, labels, times, path, vmax=1.0, n=201):
    (x0, _, z0), (x1, _, z1) = task.mesh_bbox
    if hasattr(task, "slice_box"):
        (x0, z0), (x1, z1) = task.slice_box
    X, Z = torch.meshgrid(torch.linspace(x0, x1, n, dtype=torch.float64), torch.linspace(z0, z1, n, dtype=torch.float64), indexing="ij")
    pts = torch.stack([X.flatten(), torch.zeros(X.numel(), dtype=torch.float64), Z.flatten()], 1)
    bd = task.pts_boundary.cpu()
    bd = bd[bd[:, 1].abs() < 1e-2]
    model = task.model()
    w = max(2.0 * (x1 - x0) / (z1 - z0), 2.6)  ## panel width; at least the width of the title
    fig, axs = plt.subplots(len(run_dirs), len(times), figsize=(w * len(times), 2.0 * len(run_dirs)), squeeze=False)
    for r, (row, run_dir, label) in enumerate(zip(axs, run_dirs, labels)):
        snaps = torch.load(os.path.join(run_dir, "snapshots.pt"), map_location=pts.device)
        for ax, t in zip(row, times):
            model.load_state_dict(min(snaps, key=lambda s: abs(s["time"] - t))["state"])
            with torch.no_grad():
                f = model(pts).squeeze(1).view(X.shape).cpu()
            im = ax.pcolormesh(X.cpu(), Z.cpu(), f, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
            ax.contour(X.cpu(), Z.cpu(), f, [0], colors="k", linewidths=1.5)
            for j, (name, curves) in enumerate(task.slice_curves().items()):
                for k, (cx, cz) in enumerate(curves):
                    ax.plot(cx.cpu(), cz.cpu(), ls="--" if j == 0 else ":", c="k" if j == 0 else "0.4", lw=1.2,
                            label=name if k == 0 else None)
            ax.plot(bd[:, 0], bd[:, 2], "ko", ms=4, label="boundary")
            ax.set(xticks=[], yticks=[], aspect="equal")
            if r == 0:
                ax.set_title(f"t = {t:g} s", fontsize=20)
        row[0].set_ylabel(label, fontsize=20)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6).ax.tick_params(labelsize=16)
    handles, names = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=18, frameon=False, bbox_to_anchor=(0.45, -0.02))
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--exp", default="slices")
    parser.add_argument("--runs", nargs="+", required=True, help="run directories in results/<task>/<exp>/, one row each")
    parser.add_argument("--labels", nargs="*", help="row labels (default: the optimizer)")
    parser.add_argument("--times", nargs="+", type=float, default=[0, 1, 5, 20, 100, 300])
    parser.add_argument("--vmax", type=float, default=1.0, help="colour range of f: [-vmax, vmax]")
    parser.add_argument("--name", default="slices", help="file name of the figure, without .png")
    parser.add_argument("--paper", action="store_true", help="write to paper_figures/<task>_<name>.png")
    args = parser.parse_args()
    torch.set_default_device("cpu")
    exp_dir = os.path.join(RESULTS_DIR, args.task, args.exp)
    labels = args.labels or [split_name(r.rsplit("_seed", 1)[0])[1] for r in args.runs]
    path = os.path.join(PAPER_FIGURES_DIR, f"{args.task}_{args.name}.png") if args.paper else os.path.join(exp_dir, f"{args.name}.png")
    slices_figure(TASKS[args.task](), [os.path.join(exp_dir, r) for r in args.runs], labels, args.times, path, args.vmax)
    print(path)
