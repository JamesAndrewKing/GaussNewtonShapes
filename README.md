# Preconditioning for Geometry-informed Neural Networks (GINNs)

<!-- ### [Project Page](https://arturs-berzins.github.io/GINN/) | [arXiv](https://arxiv.org/abs/2402.14009)

<img src="media/diagonal_other_overlay.gif" width="800"/>

This project accompanies the paper "Geometry-informed Neural Networks", which allows to train shape generative models without data.
Instead, GINNs are trained to satisfy design requirements given as constraints and objectives.
In particular, a diversity constraint makes these models generative.
GINNs not only learn to generate multiple diverse solutions, but can also learn an organized latent space as shown above.

<img src="media/constraints.png" width="800"/> -->

This is the code used in my Master's Thesis where I train Implicit Neural Shapes to solve Plateu's Problem using only constraints. The project is forked from (https://arturs-berzins.github.io/GINN/) and builds upon their minimal surface experiment by implementing Gauss-Newton Natural Gradient Descent.

!!Under Construction!!
## Organization

```
/
├── notebooks                       # Entry point for the program
│   ├── catenoid.ipynb              # Plateu's Problem for Catenoid
│   └── enneper.ipynb               # Plateau's Problem for Enneper Surface
├── training/                       # Functionality for training
│   ├── optimizers.py               # 
│   └── gram_factory.py             # 
├── GINN/                           # Code from GINN for sampling and PH
│   ├── ph/                         # Classes to manage the connectedness loss based on persistent homology
│   ├── speed/                      # Contains classes useful for multiprocessing or measuring time
├── models/                         # Model definition
│   └── model_architecture.py       # 
├── util/                           # Utilities used throughout the project
│   ├── surface_sampling.py         #
│   └── error_metrics.py            # 

```

## Get started

Install the dependencies, ideally in a fresh environment
```pip install -r requirements.txt```


### Minimal surface

Plateau’s problem is to find the surface $S$ with the minimal area given a prescribed boundary $\Gamma$.
A minimal surface is known to have zero mean-curvature $\kappa_H$ everywhere.

With [notebooks/catenoid.ipynb](notebooks/catenoid.ipynb) and [notebooks/enneper.ipynb](notebooks/enneper.ipynb) you can train a GINN to learn the Catenoid and the Enneper minimal surfaces.

Catenoid (grey), $\Gamma$ (green):
<img src="docs/catenoid.png" width="300"/>
Enneper Surface (grey), $\Gamma$ (green):
<img src="docs/enneper.png" width="300"/>

Here are some interactive plots of the resulting surfaces using different optimizers. The colours on the surface indicate the value of $|\kappa_H(\bold{x})|$ at that surface point:
- Catenoid: [Adam](https://JamesAndrewKing.github.io/PreconditionGINNs/k3d_plot_heatmap_catenoid_adam.html), [LBFGS](https://JamesAndrewKing.github.io/PreconditionGINNs/k3d_plot_heatmap_catenoid_lbfgs.html), [Gauss-Newton](https://JamesAndrewKing.github.io/PreconditionGINNs/k3d_plot_heatmap_catenoid_gn.html)
- Enneper Surface: [Adam](https://JamesAndrewKing.github.io/PreconditionGINNs/k3d_plot_heatmap_enneper_adam.html), [Gauss-Newton](https://JamesAndrewKing.github.io/PreconditionGINNs/k3d_plot_heatmap_enneper_gn.html)