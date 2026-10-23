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
| `enneper_heatmap.{pdf,png}` | 2a | \|κ_H\| on the best surfaces of Adam and GN; colorbar 0–0.002 |
| `enneper_loss.{pdf,png}` | 2b | Loss over time (Adam, GN; LBFGS diverges) |
| `enneper_chamfer.{pdf,png}` | 8b | Two-sided Chamfer distance over time |
| `cone_heatmap.{pdf,png}` | 3a | \|κ_G\| on the best surfaces of Adam and GN; colorbar 0–0.005 |
| `cone_loss.{pdf,png}` | 3b | Loss over time (Adam, GN; LBFGS diverges) |
| `cone_chamfer.{pdf,png}` | new | Two-sided Chamfer distance over time (the submitted version has no error plot for the cone) |
| `rockerarm_surfaces.{pdf,png}` | 4a | Best surfaces of Adam, LBFGS and GN |
| `rockerarm_loss.{pdf,png}` | 4b | Loss over time |
| `rockerarm_chamfer.{pdf,png}` | new | Two-sided Chamfer distance to the ground-truth mesh over time |
| `bunny_surfaces.{pdf,png}` | 5a | Best surfaces of Adam, LBFGS and GN |
| `bunny_loss.{pdf,png}` | 5b | Loss over time |
| `bunny_chamfer.{pdf,png}` | new | Two-sided Chamfer distance to the ground-truth mesh over time |

Figures from experiments other than `main` are prefixed `<task>_<exp>_`.
