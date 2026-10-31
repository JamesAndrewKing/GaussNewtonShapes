"""
Cost of one optimizer update against the parameter count p and the number of residuals N (no training), e.g.

    python -m experiments.scaling
    python -m experiments.scaling --widths 32 64 --ns 1000 4000 --kinds ins

Times the update direction of GN with the dense p x p system J^T J (GaussNewtonNew.calculate_update), with the
N x N system J J^T (Woodbury, calculate_update_woodbury), and matrix-free by conjugate gradients with a randomized Nystrom
preconditioner of rank 100 (calculate_update_cg, "cg"; "cg_plain" without preconditioner), and an Adam step (loss,
backward, step) for reference. CG stops at a relative residual of 1e-8 (at most 1000 iterations); the iteration count
and the time of the Nystrom sketch (included in the total) are stored with the result.
GN's line search adds up to 15 residual evaluations per step on top of this; they are timed separately ("residual").
Two residual sets, with N/3 random points in [-1, 1]^3 each:
    ins          data, normal, eikonal (first derivatives in x), sine network with 3 hidden layers as for the INS
    curvature    data, eikonal, mean curvature (second derivatives in x), tanh network with 2 hidden layers
Writes to experiments/results/scaling/ (new measurements are merged into an existing scaling.json):
    scaling.json                    median time per update, peak GPU memory, or "oom" per configuration
    scaling.md                      the same as a table
    summary.png                     time and memory against p for the INS residuals at a few N, with the GPU memory
                                    limit
    <kind>_vs_p.png, <kind>_vs_n.png   all measurements: against p (one line per N) and against N (one line per width)
Configurations above the GPU memory count as out of memory (under Windows they spill into system memory instead).
With --plot-only, the figures and the table are made from scaling.json without measuring.
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.model_architecture import GeneralNet
from training.residuals import ResidualLibrary, ResidualTerm, compute_loss
from training.optimizers import GaussNewtonNew
from training.gram_factory import compute_residual
from experiments.run import RESULTS_DIR

OUT_DIR = os.path.join(RESULTS_DIR, "scaling")
VARIANTS = ["dense", "woodbury", "cg", "cg_plain", "adam", "residual"]
LABELS = {"dense": r"GN, dense $J^TJ$", "woodbury": r"GN, Woodbury $JJ^T$", "cg": "GN, matrix-free",
          "cg_plain": "GN, matrix-free, no preconditioner", "adam": "Adam", "residual": "Residual evaluation"}
MAX_GB = 16.0  ## GPU memory (RTX 5080); dense configurations whose p x p matrix alone exceeds a third of it are not tried
COLORS = {"dense": "tab:green", "woodbury": "tab:pink", "cg": "black", "cg_plain": "tab:gray", "adam": "tab:blue"}  ## GN and Adam as in the paper figures


def setup(kind, width, n, seed=0):
    torch.manual_seed(seed)
    if kind == "ins":
        model = GeneralNet(ks=[3, width, width, width, 1], act=torch.sin)
    else:
        model = GeneralNet(ks=[3, width, width, 1])
    res_lib = ResidualLibrary(model)
    pts = lambda: 2 * torch.rand([n // 3, 3], dtype=torch.float64) - 1
    if kind == "ins":
        normals = torch.nn.functional.normalize(torch.randn([n // 3, 3], dtype=torch.float64), dim=1)
        res_terms = {
            "data":    ResidualTerm(res_lib._data,    1.0,  pts(), vals=0),
            "eikonal": ResidualTerm(res_lib._eikonal, 1e-2, pts()),
            "normal":  ResidualTerm(res_lib._normal,  1.0,  pts(), normals),
        }
    else:
        res_terms = {
            "data":           ResidualTerm(res_lib._data,           1.0,  pts(), vals=0),
            "eikonal":        ResidualTerm(res_lib._eikonal,        1e-3, pts()),
            "mean_curvature": ResidualTerm(res_lib._mean_curvature, 1.0,  pts()),
        }
    return model, res_terms


def timed(fn, repeats, max_seconds=60):
    """Median wall-clock time of fn() after one warm-up call, and the peak GPU memory in GB."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if sum(times) > max_seconds:
            break
    return float(np.median(times)), torch.cuda.max_memory_allocated() / 1e9


def measure(kind, width, n, variant, repeats):
    model, res_terms = setup(kind, width, n)
    p = sum(q.numel() for q in model.parameters())
    if variant == "dense" and 8 * p ** 2 / 1e9 > MAX_GB / 3:  ## J^T J, J^T J + eps I and the solver's copy
        return p, "oom"
    gn = GaussNewtonNew(model, res_terms)
    if variant == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        params = dict(model.named_parameters())
        def fn():
            optim.zero_grad()
            compute_loss(params, res_terms).backward()
            optim.step()
    else:
        fn = {"dense": gn.calculate_update, "woodbury": gn.calculate_update_woodbury,
              "cg": gn.calculate_update_cg, "cg_plain": lambda: gn.calculate_update_cg(precondition=False),
              "residual": lambda: compute_residual(gn.params_dict, res_terms)}[variant]
    try:
        if variant.startswith("cg"):
            return p, (*timed(fn, repeats), gn.cg_info["iterations"], gn.cg_info["sketch_time"])
        return p, timed(fn, repeats)
    except torch.cuda.OutOfMemoryError:
        return p, "oom"
    finally:
        del model, res_terms, gn
        torch.cuda.empty_cache()


def fits(r):
    return r["result"] != "oom" and r["result"][1] <= MAX_GB


def plot(results, kind, x_key, line_key, path):
    """Time and memory per update against x_key ("p" or "n"), one line style per value of line_key."""
    rows = [r for r in results if r["kind"] == kind and fits(r)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    lines = sorted({r[line_key] for r in rows})
    styles = ["-", "--", ":", "-.", (0, (5, 1, 1, 1))]
    for j, (ax, ylabel) in enumerate(zip(axs, ["Time per update in [s]", "Peak memory in [GB]"])):
        for v in ["dense", "woodbury", "cg", "adam"]:
            for line, ls in zip(lines, styles):
                pts = sorted((r[x_key], r["result"][j]) for r in rows if r["variant"] == v and r[line_key] == line)
                if pts:
                    ax.plot(*zip(*pts), c=COLORS[v], ls=ls, marker=".")
        ax.set_xlabel("Number of parameters $p$" if x_key == "p" else "Number of residuals $N$")
        ax.set_ylabel(ylabel)
        ax.loglog()
    for v in ["dense", "woodbury", "cg", "adam"]:
        axs[0].plot([], [], c=COLORS[v], label=LABELS[v])
    for line, ls in zip(lines, styles):
        axs[0].plot([], [], c="gray", ls=ls, label=(f"$N$ = {line:,}" if line_key == "n" else f"width {line}"))
    fig.legend(*axs[0].get_legend_handles_labels(), fontsize=8, loc="center left", bbox_to_anchor=(0.83, 0.5))
    plt.tight_layout(rect=(0, 0, 0.83, 1))
    plt.savefig(path, dpi=300)
    plt.close(fig)


def summary_plot(results, path, kind="ins", ns=(1000, 16000, 64000)):
    """
    Time and peak memory per update against p at a few N: GN dense and Woodbury, Adam for reference.
    Configurations that do not fit into the GPU memory are marked with a cross at the top.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    styles = dict(zip(ns, ["-", "--", ":"]))
    for j, (ax, ylabel) in enumerate(zip(axs, ["Time per update in [s]", "Peak memory in [GB]"])):
        for v in ["dense", "woodbury", "cg", "adam"]:
            for n in ns:
                rows = sorted((r for r in results if (r["kind"], r["variant"], r["n"]) == (kind, v, n)), key=lambda r: r["p"])
                pts = [(r["p"], r["result"][j]) for r in rows if fits(r)]
                if pts:
                    ax.plot(*zip(*pts), c=COLORS[v], ls=styles[n], marker=".")
                out = [r["p"] * {"dense": 0.9, "woodbury": 1.1}.get(v, 1) for r in rows if not fits(r)]  ## side by side
                ax.scatter(out, [0.97] * len(out), marker="x", color=COLORS[v], transform=ax.get_xaxis_transform(),
                           clip_on=False)
        ax.set_xlabel("Number of parameters $p$")
        ax.set_ylabel(ylabel)
        ax.loglog()
    axs[1].set_ylim(top=MAX_GB * 4)  ## room for the label above the memory limit
    axs[1].axhline(MAX_GB, c="gray", lw=0.8)
    axs[1].text(axs[1].get_xlim()[0] * 1.15, MAX_GB * 1.2, "GPU memory", color="gray", fontsize=8)
    for v in ["dense", "woodbury", "cg", "adam"]:
        axs[0].plot([], [], c=COLORS[v], label=LABELS[v])
    for n in ns:
        axs[0].plot([], [], c="gray", ls=styles[n], label=f"$N$ = {n:,}")
    axs[0].scatter([], [], marker="x", c="gray", label="out of memory")
    fig.legend(*axs[0].get_legend_handles_labels(), fontsize=8, loc="center left", bbox_to_anchor=(0.83, 0.5))
    plt.tight_layout(rect=(0, 0, 0.83, 1))
    plt.savefig(path, dpi=300)
    plt.close(fig)


def table(results):
    rows = ["| residuals | width | p | N | " + " | ".join(LABELS[v] + " (s / GB [/ CG iterations])" for v in VARIANTS) + " |",
            "|---" * (4 + len(VARIANTS)) + "|"]
    configs = sorted({(r["kind"], r["width"], r["p"], r["n"]) for r in results})
    for kind, width, p, n in configs:
        cells = []
        for v in VARIANTS:
            res = next((r["result"] for r in results if (r["kind"], r["width"], r["n"], r["variant"]) == (kind, width, n, v)), None)
            cells.append("-" if res is None else "OOM" if res == "oom" else
                         f"{res[0]:.2e} / {res[1]:.2f}" + (f" / {res[2]} it (sketch {res[3]:.2e} s)" if len(res) > 2 else ""))
        rows.append(f"| {kind} | {width} | {p} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", default=["ins", "curvature"], choices=["ins", "curvature"])
    parser.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--ns", nargs="+", type=int, default=[1000, 2000, 4000, 8000, 16000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--plot-only", action="store_true", help="only make the figures and the table from scaling.json")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "scaling.json")
    data = dict(results=[])
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    if not args.plot_only:
        torch.set_default_device("cuda")
        data.update(device=torch.cuda.get_device_name(0), torch=torch.__version__)
        key = lambda r: (r["kind"], r["width"], r["n"], r["variant"])
        for kind in args.kinds:
            for width in args.widths:
                for n in args.ns:
                    for variant in args.variants:
                        p, res = measure(kind, width, n, variant, args.repeats)
                        new = dict(kind=kind, width=width, p=p, n=n, variant=variant, result=res)
                        data["results"] = [r for r in data["results"] if key(r) != key(new)] + [new]
                        print(kind, width, p, n, variant, res, flush=True)
                        with open(json_path, "w") as f:
                            json.dump(data, f, indent=1)
    results = data["results"]
    for kind in sorted({r["kind"] for r in results}):
        plot(results, kind, "p", "n", os.path.join(OUT_DIR, f"{kind}_vs_p.png"))
        plot(results, kind, "n", "width", os.path.join(OUT_DIR, f"{kind}_vs_n.png"))
    summary_plot(results, os.path.join(OUT_DIR, "summary.png"))
    with open(os.path.join(OUT_DIR, "scaling.md"), "w") as f:
        f.write(f"## Cost per update on {data['device']}, float64\n\n{table(results)}\n")
    print(table(results))
