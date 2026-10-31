"""
Steps along the scaling direction of f (Section 3.1), e.g.

    python -m experiments.flat_directions --task enneper --opt adam --seed 3 --budget 120

The curvature residuals depend only on the zero set of f, so scaling f leaves them unchanged; only the interface and
eikonal terms depend on the scale, and without the eikonal term shrinking f lowers the loss. f is linear in the last
layer, so scaling f is the direction of the last-layer weights and bias in parameter space. Training as in
experiments.run (same settings, clock and eikonal switch-off), and per optimizer step:
    cos2         squared cosine between the step and the scaling direction (a random direction gives about 1/p)
    scale_step   component of the step along the scaling direction, relative to the last-layer norm (-0.01: f shrinks by 1%)
    grad_norm    median |grad f| at the boundary points (1 for a distance function, -> 0 when f collapses)
and at the start and just before the eikonal switch-off, the unweighted loss terms of alpha * f for alpha in [0, 1.2].

Writes to experiments/results/<task>/flat_directions/<opt><task args>_seed<seed>.json, and with --plot (no training)
the figures of all logged runs of the task to the same directory.
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from training.residuals import ResidualLibrary, compute_loss
from training.optimizers import GaussNewton
from experiments.run import RESULTS_DIR, BASE_OPT, Clock, parse_kv, pretrain, make_optimizer, train_step
from experiments.plot import LABELS, COLORS
from experiments.tasks import TASKS

ALPHAS = np.linspace(0, 1.2, 25)


def last_layer(params):
    """Names of the last layer's weight and bias (f is linear in them)."""
    names = list(params)
    return names[-2:]


def loss_slice(params, res_terms, names):
    """Unweighted loss terms of alpha * f: the last layer scaled by alpha."""
    saved = {k: params[k].detach().clone() for k in names}
    terms = {}
    with torch.no_grad():
        for a in ALPHAS:
            for k in names:
                params[k].copy_(a * saved[k])
            _, unweighted = compute_loss(params, res_terms, return_unweighted_losses=True)
            for k, v in unweighted.items():
                terms.setdefault(k, []).append(v.item())
        for k in names:
            params[k].copy_(saved[k])
    return terms


def run(task_name, opt_name, seed, budget, task_args):
    torch.set_default_device("cuda")
    task = TASKS[task_name](**task_args)
    args_str = "".join(f"_{k}{v}" for k, v in task_args.items())
    out_dir = os.path.join(RESULTS_DIR, task_name, "flat_directions")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model = pretrain(task, seed, os.path.join(RESULTS_DIR, task_name, "pretrained", f"seed{seed}{args_str}.pt"))
    params = model.params
    res_lib = ResidualLibrary(model)
    res_terms = task.res_terms(res_lib, BASE_OPT.get(opt_name, opt_name))
    gn_class = getattr(task, "gn_class", GaussNewton)
    optim = make_optimizer(opt_name, model, res_terms, gn_class, **task.opt_defaults.get(opt_name, {}))  ## as in run.py (lm: none)
    names = last_layer(params)
    flat = lambda: torch.cat([p.detach().flatten() for p in params.values()])
    mask = torch.cat([torch.full((p.numel(),), k in names, dtype=torch.bool) for k, p in params.items()])

    log = {k: [] for k in ["time", "loss", "eikonal", "cos2", "scale_step", "grad_norm"]}
    slices = {}
    clock = Clock()
    i = 0
    while clock.elapsed < budget:
        if not slices or ("before_switch_off" not in slices and clock.elapsed >= task.eikonal_off_time - 1):
            task.update(res_terms, model, clock.elapsed)
            slices["start" if not slices else "before_switch_off"] = {"time": clock.elapsed, **loss_slice(params, res_terms, names)}
        theta = flat()
        clock.start()
        task.update(res_terms, model, clock.elapsed)
        unweighted = train_step(optim, params, res_terms, None)
        clock.stop()
        delta = flat() - theta
        d = torch.where(mask, theta, 0)  ## scaling direction: the last layer itself
        dot = (delta @ d).item()
        with torch.no_grad():
            g = res_lib._grad_x_f_batch(params, task.pts_boundary).squeeze(1).norm(dim=1)
        log["time"].append(clock.elapsed)
        log["loss"].append(sum(unweighted[k].item() for k in task.loss_keys if k in unweighted))
        log["eikonal"].append(unweighted["eikonal"].item())
        log["cos2"].append(dot**2 / max((delta @ delta).item() * (d @ d).item(), 1e-300))
        log["scale_step"].append(dot / max((d @ d).item(), 1e-300))
        log["grad_norm"].append(g.median().item())
        i += 1
    path = os.path.join(out_dir, f"{opt_name}{args_str}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump({"n_params": int(mask.numel()), "eikonal_off_time": task.eikonal_off_time, "log": log,
                   "alphas": ALPHAS.tolist(), "slices": slices}, f)
    print(f"{path}: {i} steps, median cos2 {np.median(log['cos2']):.1e} (1/p = {1 / mask.numel():.1e}), "
          f"final |grad f| at the boundary {log['grad_norm'][-1]:.1e}")


def plot(task_name):
    out_dir = os.path.join(RESULTS_DIR, task_name, "flat_directions")
    runs = {f[:-5]: json.load(open(os.path.join(out_dir, f))) for f in sorted(os.listdir(out_dir)) if f.endswith(".json")}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
    for name, r in runs.items():
        opt = max((o for o in LABELS if name.startswith(o + "_")), key=len)
        t, log = np.array(r["log"]["time"]), r["log"]
        bins = np.linspace(0, t[-1], 120)
        idx = np.digitize(t, bins)
        med = lambda v: [np.median(np.asarray(v)[idx == b]) if (idx == b).any() else np.nan for b in range(1, len(bins))]
        kw = dict(c=COLORS[opt], label=name.replace("_seed", ", seed "))
        axs[0].plot(bins[1:], med(np.abs(log["scale_step"])), **kw)
        axs[1].plot(bins[1:], med(log["cos2"]), **kw)
        axs[2].plot(t, log["grad_norm"], **kw)
    p = next(iter(runs.values()))["n_params"]
    axs[1].axhline(1 / p, c="gray", ls=":", lw=0.8, label="random direction (1/p)")
    for ax, yl in zip(axs, ["|step along scaling| / |last layer|", "cos² (step, scaling direction)", "median |∇f| at the boundary"]):
        ax.axvline(next(iter(runs.values()))["eikonal_off_time"], c="gray", lw=0.8)
        ax.set_xlabel("Time in [s]")
        ax.set_ylabel(yl)
        ax.semilogy()
    axs[2].legend(fontsize=6)
    axs[1].legend(fontsize=6, handles=axs[1].get_legend_handles_labels()[0][-1:])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "steps.png"), dpi=200)
    plt.close(fig)

    ## Loss slices of the GN runs (the loss terms do not depend on the optimizer at the start)
    r = next((r for n, r in runs.items() if n.startswith("gn_")), next(iter(runs.values())))
    fig, axs = plt.subplots(1, len(r["slices"]), figsize=(4.5 * len(r["slices"]), 3.2), squeeze=False)
    for ax, (when, s) in zip(axs[0], r["slices"].items()):
        for k, v in s.items():
            if k != "time":
                ax.plot(r["alphas"], v, label=k)
        ax.set_title(f"{when.replace('_', ' ')} (t = {s['time']:.0f} s)", fontsize=9)
        ax.set_xlabel(r"$\alpha$ (model $\alpha f$)")
        ax.set_ylabel("unweighted loss term")
        ax.semilogy()
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_slices.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--opt", choices=["gn", "adam", "lbfgs", "lm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=120, help="training time in seconds")
    parser.add_argument("--task-args", nargs="*", default=[], help="e.g. init_radius=2.0")
    parser.add_argument("--plot", action="store_true", help="only make the figures from the logged runs")
    args = parser.parse_args()
    if args.plot:
        plot(args.task)
    else:
        run(args.task, args.opt, args.seed, args.budget, parse_kv(args.task_args))
