"""
Run one optimizer on one task, e.g.

    python -m experiments.run --task catenoid --opt gn --seed 0 --budget 600

Writes to experiments/results/<task>/<exp>/<opt><tag><task args>_seed<seed>/:
    config.json  run settings
    log.json     reported loss and unweighted loss terms at every iteration
    evals.json   error metrics every --eval-every seconds of training time, and at the end
    best.pt, final.pt  model state dicts

The clock only runs during the optimizer steps (incl. surface resampling).
Evaluation, logging and best-model tracking happen with the clock paused.
"""

import argparse
import json
import os
import subprocess
import time
import torch
from tqdm import trange
from training.residuals import ResidualLibrary, compute_loss
from training.optimizers import GaussNewton
from experiments.tasks import TASKS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class Clock:
    """Accumulates training time only. Synchronizes CUDA so that asynchronous kernels are counted."""
    def __init__(self):
        self.elapsed = 0.0

    def start(self):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        self.elapsed += time.perf_counter() - self._t0


def parse_kv(pairs):
    """['lr=1e-3', 'do_line_search=True'] -> {'lr': 0.001, 'do_line_search': True}"""
    return {k: json.loads(v.lower()) for k, v in (p.split("=") for p in pairs)}


def pretrain(task, seed, path, n_samples=10_000, n_iters=1000):
    """
    Fit the model to the task's initial shape with Adam (as in the notebooks), or use the task's own pretraining.
    Cached per seed.
    """
    model = task.model()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model
    if hasattr(task, "pretrain"):
        task.pretrain(model)
        torch.save(model.state_dict(), path)
        return model
    b = task.pretrain_bounds
    pts = torch.rand(n_samples, 3, dtype=torch.float64) * (b[:, 1] - b[:, 0]) + b[:, 0]
    targets = task.pretrain_target(pts)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(n_iters):
        optim.zero_grad()
        loss = 0.5 * (model(pts).squeeze(1) - targets).square().mean()
        loss.backward()
        optim.step()
    print(f"Pretraining (seed {seed}) loss: {loss.item():.2e}")
    torch.save(model.state_dict(), path)
    return model


def make_optimizer(name, model, res_terms, gn_class=GaussNewton, **kwargs):
    if name == "gn":
        return gn_class(model, res_terms, **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), **kwargs)
    raise ValueError(f"Unknown optimizer {name}")


def train_step(optim, params, res_terms, update):
    """
    One optimizer step. Returns the unweighted losses at the parameters before the step.
    update() resamples the moving points; for LBFGS it may instead be called in every closure evaluation.
    """
    if isinstance(optim, torch.optim.LBFGS):
        losses = {}
        def closure():
            if update:
                update()
            optim.zero_grad()
            loss, unweighted_losses = compute_loss(params, res_terms, return_unweighted_losses=True)
            loss.backward()
            losses.setdefault("initial", unweighted_losses)
            return loss
        optim.step(closure)
        return losses["initial"]
    optim.zero_grad()
    loss, unweighted_losses = compute_loss(params, res_terms, return_unweighted_losses=True)
    loss.backward()
    optim.step()
    return unweighted_losses


def run(task_name, opt_name, seed=0, budget=600, eval_every=5, task_args={}, opt_args={}, device="cuda", tag="",
        resample_in_closure=False, exp="main", freeze_samples=False, task_attrs={}):
    torch.set_default_device(device)
    task = TASKS[task_name](**task_args)
    for k, v in task_attrs.items():  ## e.g. eikonal_weights={"lbfgs": 0.1}; dicts are merged into the task's
        setattr(task, k, {**getattr(task, k), **v} if isinstance(v, dict) else v)
    args_str = "".join(f"_{k}{v}" for k, v in task_args.items())
    out_dir = os.path.join(RESULTS_DIR, task_name, exp, f"{opt_name}{tag}{args_str}_seed{seed}")
    pretrained_dir = os.path.join(RESULTS_DIR, task_name, "pretrained")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pretrained_dir, exist_ok=True)

    torch.manual_seed(seed)
    model = pretrain(task, seed, os.path.join(pretrained_dir, f"seed{seed}{args_str}.pt"))
    params = model.params
    res_lib = ResidualLibrary(model)
    res_terms = task.res_terms(res_lib, opt_name)
    opt_kwargs = {**task.opt_defaults[opt_name], **opt_args}
    optim = make_optimizer(opt_name, model, res_terms, getattr(task, "gn_class", GaussNewton), **opt_kwargs)

    config = dict(
        task=task_name, task_args=task_args, opt=opt_name, opt_args=opt_kwargs, seed=seed, budget=budget,
        resample_in_closure=resample_in_closure, freeze_samples=freeze_samples, task_attrs=task_attrs,
        n_params=sum(p.numel() for p in model.parameters()),
        device=torch.cuda.get_device_name(0), torch=torch.__version__,
        commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log = {"time": [], "loss": [], **{k: [] for k in res_terms}}
    evals = []
    best_loss = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    clock = Clock()

    for i in (pbar := trange(10**7, mininterval=10)):
        if clock.elapsed >= len(evals) * eval_every:
            evals.append({"time": clock.elapsed, "iter": i, **task.evaluate(model, res_lib)})
        if clock.elapsed >= budget:
            break

        clock.start()
        update = lambda: task.update(res_terms, model, i, resample=not freeze_samples or i == 0)
        if resample_in_closure:
            unweighted_losses = train_step(optim, params, res_terms, update)
        else:
            update()
            unweighted_losses = train_step(optim, params, res_terms, None)
        clock.stop()

        if hasattr(task, "reported_loss"):
            loss = task.reported_loss(unweighted_losses, res_terms)
        else:  ## a term without points (e.g. the surface vanished) makes the reported loss nan
            loss = sum(unweighted_losses[k].item() if k in unweighted_losses else float('nan') for k in task.loss_keys)
        log["time"].append(clock.elapsed)
        log["loss"].append(loss)
        for k in res_terms:
            log[k].append(unweighted_losses[k].item() if k in unweighted_losses else None)
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        pbar.set_description(f"t: {clock.elapsed:.0f}s loss: {loss:.2e}")

    torch.save(model.state_dict(), os.path.join(out_dir, "final.pt"))
    torch.save(best_state, os.path.join(out_dir, "best.pt"))
    final = {"iters": i, "time": clock.elapsed, "loss": log["loss"][-1], **task.evaluate(model, res_lib, final=True)}
    model.load_state_dict(best_state)
    best = {"loss": best_loss, **task.evaluate(model, res_lib, final=True)}
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f)
    with open(os.path.join(out_dir, "evals.json"), "w") as f:
        json.dump({"over_time": evals, "final": final, "best": best}, f, indent=1)
    print(json.dumps(final, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--opt", required=True, choices=["gn", "adam", "lbfgs"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=600, help="training time in seconds")
    parser.add_argument("--eval-every", type=float, default=5, help="seconds of training time between evaluations")
    parser.add_argument("--task-args", nargs="*", default=[], help="e.g. init_radius=2.0")
    parser.add_argument("--opt-args", nargs="*", default=[], help="e.g. lr=1e-2")
    parser.add_argument("--exp", default="main", help="experiment group, i.e. subdirectory of results/<task>")
    parser.add_argument("--tag", default="", help="suffix for the run directory, e.g. for sweeps")
    parser.add_argument("--resample-in-closure", action="store_true", help="LBFGS: resample in every closure evaluation")
    parser.add_argument("--freeze-samples", action="store_true", help="sample the moving points once and keep them fixed")
    parser.add_argument("--task-attrs", nargs="*", default=[], help='override task attributes, e.g. eikonal_weights={"lbfgs":0.1}')
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.task, args.opt, args.seed, args.budget, args.eval_every,
        parse_kv(args.task_args), parse_kv(args.opt_args), args.device, args.tag, args.resample_in_closure, args.exp,
        args.freeze_samples, parse_kv(args.task_attrs))
