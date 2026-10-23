# Experiments for the revision

Scripts that replace the notebook training loops, so that runs can be repeated across seeds, settings and tasks.
The experiment plan is in `review_files/revision_plan.md`.

## Scripts

| Script | Purpose |
|---|---|
| `run.py` | Train one optimizer on one task and save the run |
| `plot.py` | Compare runs of one task, optionally against the curves stored for the paper (diagnostics) |
| `figures.py` | Make the paper figures from finished runs into `paper_figures/` |
| `evaluate.py` | Recompute the best/final metrics of finished runs from their checkpoints (e.g. after adding a metric) |
| `metrics.py` | Error metrics shared by all tasks |
| `tasks/` | One class per problem (from the notebooks): data, residuals, resampling, evaluation, figure settings. `curvature_task.py` is shared by `catenoid`, `enneper` and `cone` |

```
python -m experiments.run      --task catenoid --opt gn --seed 0 --budget 1200
python -m experiments.plot     --task catenoid --paper
python -m experiments.figures  --task catenoid
python -m experiments.evaluate --task catenoid --exp main
```

Options of `run.py`: `--exp` (experiment group, default `main`), `--tag` (suffix of the run name), `--task-args key=value` (e.g. `init_radius=2.0`), `--opt-args key=value` (e.g. `lr=1e-2`), `--resample-in-closure` (LBFGS only), `--freeze-samples` (sample the moving points once and keep them), `--task-attrs key=value` (override task settings, e.g. `eikonal_weights={"lbfgs":0.1}`; dicts are merged), `--eval-every` (seconds).

## Where things are saved

```
paper_figures/                        tracked: final figures for the paper (see its README)
experiments/results/                  not tracked: raw outputs
  <task>/
    pretrained/seed<s>[<task args>].pt   pretrained initial model, shared by all optimizers and experiments
    main/                             runs behind the paper figures and tables
      <opt>[<tag>][<task args>]_seed<s>/
        config.json                   settings, parameter count, GPU, torch version, git commit
        log.json                      reported loss and unweighted loss terms at every iteration
        evals.json                    metrics every --eval-every s ("over_time"), of the final and of the best model
        best.pt, final.pt             model state dicts
      comparison/                     output of plot.py: curves vs the paper runs, metrics.md
      k3d/                            output of figures.py: interactive heatmaps
    <exp>/                            further experiment groups (seeds, sweeps, ...), same layout as main/
    archive/<exp>/                    diagnostic or superseded runs, each with a README of what was found
```

## Metrics

Computed by each task's `evaluate()` and stored in `evals.json`. All distances are RMS values (square root of the mean squared distance), as for CD₁ in Appendix C.

**Sampling.** Model surface samples come from a binary search on a 100k-point grid in the task's sampling box, followed by an orthogonal Newton projection onto {f = 0} (Appendix B). Only converged points (|f| < 1e-8) are kept. True surface samples are a 64 × 128 grid of the parametrization over the region of interest. The *region of interest* is the part of space where the true solution is defined:

| Task | Region of interest | True surface |
|---|---|---|
| catenoid | \|z\| ≤ h (between the rings) | catenoid patch |
| enneper | above or below the patch r ≤ 0.9 in xy, within ±0.1 of it in z | Enneper patch r ≤ 0.9 |
| cone | 0 ≤ z ≤ 1 | mantle of the truncated cone |
| rockerarm, bunny | the sampling box (bounding box of the points ± 0.1) | the ground-truth mesh, normalized like the training data (centered, max vertex norm 1) |

For the implicit shapes, the true samples are 20k points sampled on the mesh (fixed seed), and the model → truth distances are exact closest-point distances to the mesh (trimesh).

**Definitions.**

| Key | Definition |
|---|---|
| `loss` | Reported loss: sum of the unweighted interface and curvature losses ½·mean(r²); the eikonal term is excluded. Evaluated before each step |
| `cd_t2m` | Truth → model: RMS of ‖x − P(x)‖ over the true samples x, where P is the Newton projection onto the model surface. `t2m_converged` is the fraction of converged projections |
| `cd_m2t` | Model → truth: RMS of the exact distance from the model samples in the region of interest to the true patch |
| `cd` | Two-sided Chamfer distance: (`cd_t2m` + `cd_m2t`) / 2 |
| `hausdorff` | Largest distance in either direction (over the samples) |
| `cd1_paper` | One-sided CD₁ of Appendix C (`util.error_metrics.chamfer_div`), to compare with the submitted figures |
| `interface_rms`, `interface_max` | Distance of the boundary curve points (implicit shapes: the training points, i.e. point-to-surface error) to the model surface (Newton projection) |
| `normal_consistency` | Implicit shapes: mean of ⟨∇f/‖∇f‖, n⟩ over the model samples, with n the mesh face normal at the closest point (1 is perfect) |
| `eikonal_rms` | Implicit shapes: RMS of ‖∇f‖ − 1 over 20k fixed uniform points in the bounding box of the points ± 0.2 |
| `held_out_rms` | Implicit shapes with `subsample_fraction` < 1: distance of the held-out vertices to the model surface |
| `curv_rms`, `curv_max` | \|κ\| on the model samples in the region of interest, with κ = κ_H (catenoid, Enneper) or κ_G (cone) |
| `curv_box_rms`, `curv_box_max` | The same on all model samples in the sampling box |
| `area` | Area of the model surface in the region of interest (best and final models only) |
| `area_err_analytic` | `area` / analytic area − 1 |
| `area_err_discrete` | `area` / area of the true surface discretized the same way − 1. Discretization errors cancel, so this is the more accurate of the two |

**Exact distances to the true surface.** Catenoid: Newton along the profile curve of the surface of revolution. Cone: distance in the (ρ, z) plane to the profile segment. Enneper: Gauss-Newton on the parametrization (u, v), started from the inverse of (u, v) → (x, y), and kept on the disc r ≤ 0.9.

**Areas.** Catenoid and cone: marching-cubes mesh (N = 256) sliced at the planes that bound the region; the true surface is meshed from its analytic level set in the same way. Enneper: the patch is a graph over the xy-plane, so the area is a quadrature over a 600² xy-grid of |∇f| / |∂f/∂z| (model) and of the exact area element (1 + r²)/(1 − r²) (true surface), with the same grid and mask for both.

## Conventions

- The clock only runs during the optimizer steps, including surface resampling. Evaluation, logging and best-model tracking happen with the clock paused.
- Tables and heatmaps use the best model (lowest reported loss); the final model is kept in `evals.json` as well.
- Each optimizer's default settings for a task are in the task's `opt_defaults`. They are the settings that produced the paper figures, which in some cases differ from the current notebooks (see `archive/lbfgs_settings`).
- All runs of one seed start from the same pretrained model.
- If the surface leaves the sampling box or the region of interest, the affected loss terms and metrics are nan rather than an error, so such runs count as failures.
