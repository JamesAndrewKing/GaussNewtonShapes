# Experiments for the revision

Scripts that replace the notebook training loops, so that runs can be repeated across seeds, settings and tasks.
The experiment plan is in `review_files/revision_plan.md`.

## Scripts

| Script | Purpose |
|---|---|
| `run.py` | Train one optimizer on one task and save the run |
| `plot.py` | Compare runs of one task, optionally against the curves stored for the paper (diagnostics) |
| `figures.py` | Make the paper figures from finished runs into `paper_figures/` |
| `metrics.py` | Error metrics shared by all tasks |
| `tasks/` | One class per problem (from the notebooks): data, residuals, resampling, evaluation, figure settings |

```
python -m experiments.run     --task catenoid --opt gn --seed 0 --budget 1200
python -m experiments.plot    --task catenoid --paper
python -m experiments.figures --task catenoid
```

Options of `run.py`: `--exp` (experiment group, default `main`), `--tag` (suffix of the run name), `--task-args key=value` (e.g. `init_radius=2.0`), `--opt-args key=value` (e.g. `lr=1e-2`), `--resample-in-closure` (LBFGS only), `--eval-every` (seconds).

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

## Conventions

- The clock only runs during the optimizer steps, including surface resampling. Evaluation, logging and best-model tracking happen with the clock paused.
- Tables and heatmaps use the best model (lowest reported loss); the final model is kept in `evals.json` as well.
- Each optimizer's default settings for a task are in the task's `opt_defaults`. They are the settings that produced the paper figures, which in some cases differ from the current notebooks (see `archive/lbfgs_settings`).
- All runs of one seed start from the same pretrained model.
