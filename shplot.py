"""
shplot -- shared plotting style and helpers for the notebook series.

Keeps the notebooks readable (one call per figure) and the visual language
consistent. Colour choices follow one rule each:

    signed data      (basis functions, coefficients, error with sign)
                     -> DIVERGING blue <-> red with a neutral grey midpoint
    magnitude data   (radiance, irradiance, |error|)
                     -> SEQUENTIAL single-hue blue ramp
    series identity  (convergence curves, per-band lines)
                     -> CATEGORICAL slots assigned in fixed order, never cycled

Never a rainbow colormap: `jet`/`hsv` invent contours that aren't in the data
and collapse under colour-vision deficiency.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "SERIES", "SURFACE", "INK", "MUTED", "GRID",
    "CM_DIVERGING", "CM_SEQUENTIAL", "CM_RADIANCE",
    "use_style", "signed_norm",
    "show_equirect", "show_images", "show_render",
    "coeff_bar", "coeff_matrix", "sphere_3d", "basis_grid",
]

# ---- validated categorical palette (adjacent-pair CVD safe in light mode) ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# ---- diverging: two hues, neutral grey midpoint (never a hue at the middle) --
CM_DIVERGING = LinearSegmentedColormap.from_list(
    "sh_diverging", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
                     "#f3b0af", "#e34948", "#8f2020"])

# ---- sequential: one hue, monotonic lightness -------------------------------
CM_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sh_sequential", ["#f6fafe", "#cde2fb", "#9ec5f4", "#5598e7",
                      "#2a78d6", "#184f95", "#0d366b"])

# Radiance reads better bright-is-more, so this is the same single-hue ramp
# oriented dark->light. Still monotonic in lightness, still one hue.
CM_RADIANCE = LinearSegmentedColormap.from_list(
    "sh_radiance", ["#06172e", "#0d366b", "#1c5cab", "#3987e5",
                    "#86b6ef", "#cde2fb", "#ffffff"])


def use_style():
    """Apply the notebook-wide matplotlib style. Call once per notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.family": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 10,
        "axes.labelcolor": SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "image.cmap": "sh_sequential",
        "text.color": INK,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })
    for name, cm in (("sh_diverging", CM_DIVERGING),
                     ("sh_sequential", CM_SEQUENTIAL),
                     ("sh_radiance", CM_RADIANCE)):
        try:
            mpl.colormaps.register(cm, name=name)
        except ValueError:
            pass   # already registered on notebook re-run


def signed_norm(data):
    """Symmetric limits about zero, so the diverging midpoint means zero."""
    a = float(np.nanmax(np.abs(np.asarray(data, dtype=float))))
    a = a if a > 0 else 1.0
    return dict(vmin=-a, vmax=a)


# =========================================================================
# Spherical function display
# =========================================================================

def show_equirect(img, title="", ax=None, signed=False, cmap=None, cbar=True,
                  vmin=None, vmax=None):
    """Show a (H,W) scalar or (H,W,3) colour equirectangular image.

    x axis is azimuth phi (0..2pi), y axis is polar angle theta (0 at +Z top).
    """
    img = np.asarray(img, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 2.6))

    kw = dict(extent=[0, 360, 180, 0], aspect="auto", interpolation="bilinear")
    if img.ndim == 3:
        im = ax.imshow(np.clip(img, 0, 1), **kw)
        cbar = False
    else:
        if cmap is None:
            cmap = CM_DIVERGING if signed else CM_RADIANCE
        if signed and vmin is None and vmax is None:
            kw.update(signed_norm(img))
        else:
            kw.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(img, cmap=cmap, **kw)

    ax.set_title(title)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180])
    ax.set_xlabel("azimuth $\\phi$ (deg)")
    ax.set_ylabel("polar $\\theta$")
    ax.grid(False)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    return ax


def show_images(images, titles=None, cols=None, signed=False, figsize=None,
                cmap=None, share_scale=True, suptitle=None):
    """Grid of equirect images with a shared colour scale for fair comparison."""
    n = len(images)
    cols = cols or min(n, 3)
    rows = int(np.ceil(n / cols))
    figsize = figsize or (5.0 * cols, 2.5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    vmin = vmax = None
    if share_scale and np.asarray(images[0]).ndim == 2:
        allv = np.concatenate([np.asarray(i).ravel() for i in images])
        if signed:
            a = np.abs(allv).max()
            vmin, vmax = -a, a
        else:
            vmin, vmax = float(allv.min()), float(allv.max())

    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        t = titles[k] if titles is not None else ""
        show_equirect(images[k], t, ax=ax, signed=signed, cmap=cmap,
                      vmin=vmin, vmax=vmax)
    if suptitle:
        fig.suptitle(suptitle, x=0.02, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    return fig, axes


def show_render(img, title="", ax=None, exposure=1.0, mode="aces"):
    """Tonemap + sRGB-encode a linear HDR render and display it."""
    import shscene as ss
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.imshow(ss.srgb_encode(ss.tonemap(img, exposure, mode)))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


# =========================================================================
# Coefficient display
# =========================================================================

def coeff_bar(coeffs, ax=None, title="SH coefficients", labels=True):
    """Bar chart of SH coefficients, coloured by sign, grouped by band."""
    import shlib as sh
    c = np.asarray(coeffs, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 2.6))
    n = len(c)
    colors = [SERIES[0] if v >= 0 else SERIES[7] for v in c]
    ax.bar(range(n), c, color=colors, width=0.72)
    ax.axhline(0, color=AXIS, lw=0.8)

    lmax = int(np.sqrt(n)) - 1
    for l in range(1, lmax + 1):
        ax.axvline(l * l - 0.5, color=GRID, lw=1.0, zorder=0)
    if labels:
        ticks, names = [], []
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                ticks.append(sh.sh_index(l, m))
                names.append(f"{l},{m}")
        ax.set_xticks(ticks)
        ax.set_xticklabels(names, fontsize=7.5, rotation=90)
    ax.set_xlabel("band $l$, order $m$")
    ax.set_title(title)
    return ax


def coeff_matrix(coeffs, ax=None, title="coefficients as a pyramid"):
    """The familiar triangle layout: row = band l, column = order m."""
    import shlib as sh
    c = np.asarray(coeffs, dtype=float)
    lmax = int(np.sqrt(len(c))) - 1
    grid = np.full((lmax + 1, 2 * lmax + 1), np.nan)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            grid[l, m + lmax] = c[sh.sh_index(l, m)]

    if ax is None:
        _, ax = plt.subplots(figsize=(1.0 + 0.5 * (2 * lmax + 1), 0.6 + 0.5 * (lmax + 1)))
    ax.imshow(grid, cmap=CM_DIVERGING, **signed_norm(c[~np.isnan(c)]))
    ax.set_xticks(range(2 * lmax + 1))
    ax.set_xticklabels(range(-lmax, lmax + 1), fontsize=8)
    ax.set_yticks(range(lmax + 1))
    ax.set_xlabel("order $m$")
    ax.set_ylabel("band $l$")
    ax.set_title(title)
    ax.grid(False)
    return ax


# =========================================================================
# 3D lobe display
# =========================================================================

def sphere_3d(fn, ax=None, title="", res=90, signed=True, radius_mode=True):
    """Plot a spherical function as a deformed sphere ("lobes").

    `radius_mode=True` uses |f| as the radius and colours by sign -- the classic
    SH picture. `False` keeps a unit sphere and only colours it.
    """
    import shlib as sh
    theta = np.linspace(0, np.pi, res)
    phi = np.linspace(0, 2 * np.pi, 2 * res)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    d = sh.dir_from_spherical(T, P)
    v = np.asarray(fn(d.reshape(-1, 3)), dtype=float).reshape(T.shape)

    r = np.abs(v) if radius_mode else np.ones_like(v)
    X, Y, Z = r * d[..., 0], r * d[..., 1], r * d[..., 2]

    if signed:
        a = np.abs(v).max() or 1.0
        norm = plt.Normalize(-a, a)
        colors = CM_DIVERGING(norm(v))
    else:
        norm = plt.Normalize(v.min(), v.max())
        colors = CM_RADIANCE(norm(v))

    if ax is None:
        fig = plt.figure(figsize=(3.0, 3.0))
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)

    m = float(np.max(np.abs([X, Y, Z]))) or 1.0
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, y=0.96)
    ax.view_init(elev=22, azim=35)
    return ax


def basis_grid(lmax=2, res=60, size=1.5):
    """The canonical pyramid of SH basis lobes, band per row."""
    import shlib as sh
    n_rows = lmax + 1
    n_cols = 2 * lmax + 1
    fig = plt.figure(figsize=(size * n_cols, size * n_rows))
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            col = m + lmax
            ax = fig.add_subplot(n_rows, n_cols, l * n_cols + col + 1,
                                 projection="3d")
            idx = sh.sh_index(l, m)
            sphere_3d(lambda d, i=idx: sh.sh_eval_dirs(lmax, d)[..., i],
                      ax=ax, title=f"$y_{{{l}}}^{{{m}}}$", res=res)
    fig.tight_layout()
    return fig
