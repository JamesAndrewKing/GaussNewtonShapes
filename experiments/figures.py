"""
Paper figures from finished runs (no training), e.g.

    python -m experiments.figures --task catenoid

Writes the paper figures to paper_figures/ (prefix <task>, or <task>_<exp> for exp != main):
    <prefix>_loss.{png,pdf}, <prefix>_chamfer.{png,pdf}   training curves
    <prefix>_heatmap.{png,pdf}                            curvature of the best surfaces side by side, with colorbar
and the interactive scenes to experiments/results/<task>/<exp>/k3d/:
    heatmap.html                                          the scene of <prefix>_heatmap
    heatmap_<opt>.html                                    one optimizer per file (e.g. for the README)
The PNG of the k3d scene is rendered with headless Chrome via selenium.
"""

import argparse
import base64
import io
import os
import pathlib
import k3d
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image, ImageOps
from selenium import webdriver
from torch.func import vmap
from training.residuals import ResidualLibrary
from util.visualization.utils_mesh import get_mesh
from experiments.run import RESULTS_DIR
from experiments.tasks import TASKS
from experiments.plot import LABELS, load_runs, plot_over_time

PAPER_FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_figures")
ORDER = ["adam", "lbfgs", "gn"]  ## left to right


def curvature_mesh(task, run_dir, N=256):
    """Marching-cubes mesh of the best model with |curvature| as vertex attribute."""
    hm = task.heatmap
    model = task.model()
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt")))
    verts, faces = get_mesh(
        lambda x: model(x.double()), N=N, device='cuda',
        bbox_min=torch.tensor(hm["bbox_min"], dtype=torch.float64),
        bbox_max=torch.tensor(hm["bbox_max"], dtype=torch.float64),
        chunks=2,
    )
    residual = getattr(ResidualLibrary(model), hm["residual"])
    k = vmap(residual, in_dims=(None, 0))(model.params, torch.tensor(verts, dtype=torch.float64)).squeeze()
    return verts.astype(np.float32), faces.astype(np.uint32), k.abs().detach().cpu().numpy().astype(np.float32)


def k3d_scene(task, meshes, gap=0.0, aspect=3.0):
    """Meshes side by side along x, viewed from slightly above with z up (as in the paper)."""
    hm = task.heatmap
    width = hm["bbox_max"][0] - hm["bbox_min"][0]
    offsets = (np.arange(len(meshes)) - (len(meshes) - 1) / 2) * width * (1 + gap)
    fig = k3d.plot(height=800, grid_visible=False, axes_helper=0, menu_visibility=False,
                   background_color=0xffffff, camera_auto_fit=False, camera_fov=1.0)
    for (verts, faces, k), dx in zip(meshes, offsets):
        shift = np.array([dx, 0, 0], dtype=np.float32)
        fig += k3d.mesh(verts + shift, faces, attribute=k, color_range=list(hm["color_range"]),
                        color_map=k3d.basic_color_maps.Jet, side='double', flat_shading=False)
        fig += k3d.points(task.pts_boundary.cpu().numpy().astype(np.float32) + shift, color=0x00ff00, point_size=0.08)
    ## Fit the scene horizontally: half width / tan(half horizontal fov)
    distance = 0.55 * (offsets[-1] - offsets[0] + width) / (aspect * np.tan(np.deg2rad(0.5)))
    e = np.deg2rad(hm["elevation"])
    fig.camera = [0, -distance * np.cos(e), distance * np.sin(e),  0, 0, 0,  0, 0, 1]
    return fig


def k3d_screenshot(html_path, width, height, scale=2):
    """Render a k3d snapshot with headless Chrome and crop the white margins."""
    options = webdriver.ChromeOptions()
    for arg in ["--headless=new", f"--window-size={width},{height}", "--hide-scrollbars"]:
        options.add_argument(arg)
    driver = webdriver.Chrome(options=options)
    driver.set_script_timeout(300)
    driver.get(pathlib.Path(html_path).resolve().as_uri())
    png = driver.execute_async_script("""
        var scale = arguments[0], done = arguments[arguments.length - 1];
        (function wait() {
            if (typeof K3DInstance === 'undefined') return setTimeout(wait, 100);
            K3DInstance.then(function (k) { return k.getScreenshot(scale, true); })
                       .then(function (canvas) { done(canvas.toDataURL().split(',')[1]); });
        })();
    """, scale)
    driver.quit()
    img = Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB")
    return img.crop(ImageOps.invert(img).getbbox())


def heatmap_figure(img, color_range, paths, width=7.0):
    """The rendered scene with a colorbar on the left (Figure 1a)."""
    with plt.rc_context({"font.family": "serif", "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True}):
        fig = plt.figure(figsize=(width, 0.9 * width * img.height / img.width))
        ax_cbar = fig.add_axes([0.02, 0.12, 0.015, 0.76])
        ax_img = fig.add_axes([0.13, 0.0, 0.87, 1.0])
        ax_img.imshow(img)
        ax_img.axis("off")
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*color_range), cmap="jet"), cax=ax_cbar,
                     ticks=np.linspace(*color_range, 6))
        for path in paths:
            fig.savefig(path, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--exp", default="main")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_default_device('cuda')
    task = TASKS[args.task]()
    exp_dir = os.path.join(RESULTS_DIR, args.task, args.exp)
    k3d_dir = os.path.join(exp_dir, "k3d")
    os.makedirs(k3d_dir, exist_ok=True)
    os.makedirs(PAPER_FIGURES_DIR, exist_ok=True)
    runs = load_runs(exp_dir, ORDER, args.seed)
    prefix = args.task if args.exp == "main" else f"{args.task}_{args.exp.replace('/', '_')}"
    out = lambda name: [os.path.join(PAPER_FIGURES_DIR, f"{prefix}_{name}.{ext}") for ext in ["png", "pdf"]]

    ## Training curves
    plot_over_time(runs, "loss", "Loss", out("loss"))
    plot_over_time(runs, "cd", "Chamfer distance", out("chamfer"))

    ## Curvature heatmaps
    meshes = {opt: curvature_mesh(task, os.path.join(exp_dir, f"{opt}_seed{args.seed}")) for opt in runs}
    for opt, mesh in meshes.items():
        with open(os.path.join(k3d_dir, f"heatmap_{opt}.html"), "w") as f:
            f.write(k3d_scene(task, [mesh], aspect=1.5).get_snapshot())
    html_path = os.path.join(k3d_dir, "heatmap.html")
    with open(html_path, "w") as f:
        f.write(k3d_scene(task, list(meshes.values())).get_snapshot())
    heatmap_figure(k3d_screenshot(html_path, 2400, 800), task.heatmap["color_range"], out("heatmap"))
    print("Heatmap order (left to right):", ", ".join(LABELS[opt] for opt in meshes))
