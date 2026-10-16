# Paper figures

Final figures for the revised paper. All of them are generated from finished runs by

```
python -m experiments.figures --task <task>
```

Nothing is retrained. The runs themselves are in `experiments/results/<task>/main/` (see `experiments/README.md`).

| File | Figure in the submitted version | Content |
|---|---|---|
| `catenoid_heatmap.{pdf,png}` | 1a | \|κ_H\| on the best surfaces of Adam, LBFGS and GN (left to right); colorbar 0–0.002 |
| `catenoid_loss.{pdf,png}` | 1b | Loss over time |
| `catenoid_chamfer.{pdf,png}` | 8a | Two-sided Chamfer distance over time (replaces the one-sided CD₁) |

Figures from experiments other than `main` are prefixed `<task>_<exp>_`.
