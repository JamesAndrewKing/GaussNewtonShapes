"""
Paper figures from finished runs (no training), e.g.

    python -m experiments.figures --task catenoid

Writes the paper figures to paper_figures/ (prefix <task>, or <task>_<exp> for exp != main):
    <prefix>_loss.{png,pdf}, <prefix>_chamfer.{png,pdf}   training curves
    <prefix>_heatmap.{png,pdf}                            curvature of the best surfaces side by side, with colorbar
                                                          (<prefix>_surfaces: plain surfaces, for tasks without a curvature)
and the interactive scenes to experiments/results/<task>/<exp>/k3d/:
    heatmap.html / surfaces.html                          the scene of the figure above
    heatmap_<opt>.html / surfaces_<opt>.html              one optimizer per file (e.g. for the README)
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


def to_z_up(task, x):
    """Rotate shapes whose up axis is y (task.heatmap["up"]) so that z is up for the camera."""
    return x[:, [0, 2, 1]] * np.array([1, -1, 1], dtype=x.dtype) if task.heatmap.get("up") == "y" else x


def surface_mesh(task, run_dir, N=256):
    """Marching-cubes mesh of the best model, with |curvature| as vertex attribute if the task has a curvature."""
    model = task.model()
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt")))
    bbox_min, bbox_max = (torch.tensor(b, dtype=torch.float64) for b in task.mesh_bbox)
    verts, faces = get_mesh(lambda x: model(x.double()), N=N, device='cuda', bbox_min=bbox_min, bbox_max=bbox_max, chunks=2)
    k = None
    if hasattr(task, "curvature"):
        residual = getattr(ResidualLibrary(model), "_" + task.curvature)
        k = vmap(residual, in_dims=(None, 0))(model.params, torch.tensor(verts, dtype=torch.float64)).squeeze()
        k = k.abs().detach().cpu().numpy().astype(np.float32)
    return to_z_up(task, verts.astype(np.float32)), faces.astype(np.uint32), k


def k3d_scene(task, meshes):
    """
    Meshes side by side, viewed from slightly above with z up (as in the paper).
    The camera looks along +y rotated by task.heatmap["azimuth"] about z; the meshes are shifted along the screen's horizontal.
    The scene fits a window with aspect ratio len(meshes):1.
    """
    hm = task.heatmap
    a, e = np.deg2rad(hm["azimuth"]), np.deg2rad(hm["elevation"])
    right = np.array([np.cos(a), np.sin(a), 0])
    view = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])  ## from the center towards the camera
    verts_all = np.concatenate([verts for verts, _, _ in meshes]).astype(np.float64)
    center = (verts_all.min(0) + verts_all.max(0)) / 2
    width = float(np.ptp(verts_all @ right))  ## extent of one mesh along the screen's horizontal
    offsets = (np.arange(len(meshes)) - (len(meshes) - 1) / 2) * width * (1 + hm["gap"])
    aspect = len(meshes)

    fig = k3d.plot(height=800, grid_visible=False, axes_helper=0, menu_visibility=False,
                   background_color=0xffffff, camera_auto_fit=False, camera_fov=1.0)
    for (verts, faces, k), dx in zip(meshes, offsets):
        shift = (dx * right).astype(np.float32)
        if k is None:
            fig += k3d.mesh(verts + shift, faces, color=0xbbbbbb, side='double', flat_shading=False)
        else:
            fig += k3d.mesh(verts + shift, faces, attribute=k, color_range=list(hm["color_range"]),
                            color_map=k3d.basic_color_maps.Jet, side='double', flat_shading=False)
        if hasattr(task, "pts_boundary"):
            pts = to_z_up(task, task.pts_boundary.cpu().numpy().astype(np.float32))
            fig += k3d.points(pts + shift, color=0x00ff00, point_size=0.02 * width)
    ## Fit the scene horizontally: half width / tan(half horizontal fov)
    distance = 0.55 * (offsets[-1] - offsets[0] + width) / (aspect * np.tan(np.deg2rad(0.5)))
    fig.camera = [float(c) for c in [*(center + distance * view), *center, 0, 0, 1]]
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
    """The rendered scene, with a colorbar on the left if color_range is given (Figure 1a)."""
    with plt.rc_context({"font.family": "serif", "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True}):
        if color_range is None:
            fig = plt.figure(figsize=(width, width * img.height / img.width))
            ax_img = fig.add_axes([0, 0, 1, 1])
        else:
            fig = plt.figure(figsize=(width, 0.9 * width * img.height / img.width))
            ax_img = fig.add_axes([0.13, 0.0, 0.87, 1.0])
            ax_cbar = fig.add_axes([0.02, 0.12, 0.015, 0.76])
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*color_range), cmap="jet"), cax=ax_cbar,
                         ticks=np.linspace(*color_range, 6))
        ax_img.imshow(img)
        ax_img.axis("off")
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
    runs = load_runs(exp_dir, task.figure_opts, args.seed)
    prefix = args.task if args.exp == "main" else f"{args.task}_{args.exp.replace('/', '_')}"
    out = lambda name: [os.path.join(PAPER_FIGURES_DIR, f"{prefix}_{name}.{ext}") for ext in ["png", "pdf"]]

    ## Training curves
    plot_over_time(runs, "loss", "Loss", out("loss"))
    plot_over_time(runs, "cd", "Chamfer distance", out("chamfer"))

    ## Best surfaces side by side: curvature heatmaps, or plain surfaces
    name = "heatmap" if hasattr(task, "curvature") else "surfaces"
    meshes = {opt: surface_mesh(task, os.path.join(exp_dir, f"{opt}_seed{args.seed}")) for opt in runs}
    for opt, mesh in meshes.items():
        with open(os.path.join(k3d_dir, f"{name}_{opt}.html"), "w") as f:
            f.write(k3d_scene(task, [mesh]).get_snapshot())
    html_path = os.path.join(k3d_dir, f"{name}.html")
    with open(html_path, "w") as f:
        f.write(k3d_scene(task, list(meshes.values())).get_snapshot())
    heatmap_figure(k3d_screenshot(html_path, 800 * len(meshes), 800), task.heatmap.get("color_range"), out(name))
    print(f"{name} order (left to right):", ", ".join(LABELS[opt] for opt in meshes))
