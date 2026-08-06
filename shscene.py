"""
shscene -- procedural environment maps and a tiny numpy raytracer, so the
notebook series can light an actual scene rather than just plot basis functions.

Deliberately minimal: primary rays, analytic spheres, one ground plane, and a
choice of shading models so we can put "ground truth" and "SH light probe"
side by side and diff them.

Uses the same Z-up convention as shlib (see its module docstring).
"""

from __future__ import annotations

import numpy as np

import shlib as sh

__all__ = [
    "sky_gradient", "sky_sun", "sky_studio", "sky_split", "ENVIRONMENTS",
    "Sphere", "Scene", "default_scene",
    "camera_rays", "intersect_scene",
    "shade_ground_truth", "shade_sh_probe", "shade_probe_grid",
    "build_probe_grid", "render",
    "tonemap", "srgb_encode",
]


# =========================================================================
# 1. Environment maps  (direction -> RGB radiance)
# =========================================================================
# All of these take dirs of shape (..., 3) and return (..., 3) linear RGB.
# They are HDR: the sun is genuinely hundreds of times brighter than the sky,
# which is exactly what makes SH truncation interesting.

def sky_gradient(dirs):
    """Soft blue sky over warm ground bounce. Very low frequency -- SH nails it."""
    dirs = np.asarray(dirs, dtype=float)
    z = dirs[..., 2:3]
    up = np.clip(z, 0.0, 1.0)
    down = np.clip(-z, 0.0, 1.0)
    sky = np.array([0.35, 0.55, 0.95]) * (0.35 + 0.65 * up)
    ground = np.array([0.32, 0.24, 0.16]) * (0.30 + 0.70 * down)
    return sky * up + ground * down + np.array([0.18, 0.20, 0.24]) * (1.0 - up - down)


def sky_sun(dirs, sun_dir=(0.35, 0.25, 0.90), intensity=120.0, size=0.985):
    """Sky gradient plus a small, very bright sun disc.

    The hard case: a near-delta function on the sphere. This is where band
    truncation rings and where windowing earns its keep.
    """
    dirs = np.asarray(dirs, dtype=float)
    s = np.asarray(sun_dir, dtype=float)
    s = s / np.linalg.norm(s)
    cosang = dirs @ s
    disc = (cosang > size)[..., None].astype(float)
    sun = disc * np.array([1.0, 0.93, 0.80]) * intensity
    return sky_gradient(dirs) + sun


def sky_studio(dirs):
    """Three coloured area lights -- the classic key/fill/rim setup.

    Mid-frequency and strongly directional, so the band-1 terms carry real
    information and L0-only reconstruction looks obviously flat.
    """
    dirs = np.asarray(dirs, dtype=float)
    out = np.full(dirs.shape[:-1] + (3,), 0.04)
    lights = [
        ((0.6, 0.5, 0.62), (1.00, 0.85, 0.65), 9.0, 0.82),   # key, warm
        ((-0.75, 0.2, 0.35), (0.45, 0.62, 1.00), 4.0, 0.75),  # fill, cool
        ((-0.1, -0.9, 0.3), (0.95, 0.45, 0.55), 5.0, 0.80),   # rim, magenta
    ]
    for d, col, inten, cut in lights:
        a = np.asarray(d, dtype=float)
        a /= np.linalg.norm(a)
        c = dirs @ a
        fall = np.clip((c - cut) / (1.0 - cut), 0.0, 1.0)[..., None]
        out = out + np.asarray(col) * inten * (fall ** 2)
    return out


def sky_split(dirs):
    """Hard half-and-half: red hemisphere, teal hemisphere, sharp terminator.

    A step function on the sphere -- the spherical analogue of a square wave,
    and the cleanest way to see Gibbs ringing.
    """
    dirs = np.asarray(dirs, dtype=float)
    left = (dirs[..., 0:1] > 0).astype(float)
    return np.array([1.6, 0.25, 0.20]) * left + np.array([0.15, 0.75, 0.85]) * (1 - left)


ENVIRONMENTS = {
    "gradient": sky_gradient,
    "sun": sky_sun,
    "studio": sky_studio,
    "split": sky_split,
}


# =========================================================================
# 2. Scene description
# =========================================================================

class Sphere:
    """An analytic sphere with a Lambertian albedo."""

    def __init__(self, center, radius, albedo=(0.8, 0.8, 0.8)):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.albedo = np.asarray(albedo, dtype=float)

    def __repr__(self):
        return f"Sphere(c={self.center.tolist()}, r={self.radius})"


class Scene:
    """Spheres plus an optional infinite ground plane at z = `ground_z`."""

    def __init__(self, spheres, env=sky_studio, ground_z=0.0,
                 ground_albedo=(0.55, 0.55, 0.58), has_ground=True):
        self.spheres = list(spheres)
        self.env = env
        self.ground_z = float(ground_z)
        self.ground_albedo = np.asarray(ground_albedo, dtype=float)
        self.has_ground = bool(has_ground)


def default_scene(env=sky_studio):
    """Three spheres on a plane -- enough geometry to show probe interpolation."""
    return Scene(
        spheres=[
            Sphere((0.0, 0.0, 1.0), 1.0, (0.80, 0.80, 0.82)),
            Sphere((-2.4, 0.9, 0.65), 0.65, (0.85, 0.42, 0.35)),
            Sphere((2.1, -0.7, 0.80), 0.80, (0.38, 0.62, 0.85)),
        ],
        env=env,
    )


# =========================================================================
# 3. Ray generation and intersection
# =========================================================================

def camera_rays(width, height, eye=(0.0, -7.5, 3.2), target=(0.0, 0.0, 1.0),
                up=(0.0, 0.0, 1.0), fov_deg=45.0):
    """Pinhole camera. Returns (origins (H,W,3), dirs (H,W,3))."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)

    aspect = width / height
    half_h = np.tan(np.radians(fov_deg) * 0.5)
    half_w = half_h * aspect

    u = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    v = 1.0 - (np.arange(height) + 0.5) / height * 2.0
    U, V = np.meshgrid(u, v)

    d = (fwd[None, None, :]
         + (U * half_w)[..., None] * right[None, None, :]
         + (V * half_h)[..., None] * true_up[None, None, :])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(eye, d.shape).copy()
    return o, d


def intersect_scene(scene, origins, dirs, tmin=1e-4, tmax=1e9):
    """Closest-hit intersection.

    Returns dict with `hit` (bool), `t`, `point`, `normal`, `albedo`,
    all flattened to shape (N, ...) matching the input ray count.
    """
    o = np.asarray(origins, dtype=float).reshape(-1, 3)
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    n = o.shape[0]

    best_t = np.full(n, tmax)
    normal = np.zeros((n, 3))
    albedo = np.zeros((n, 3))
    hit = np.zeros(n, dtype=bool)

    for s in scene.spheres:
        oc = o - s.center
        b = np.einsum("ij,ij->i", oc, d)          # d is unit, so a == 1
        c = np.einsum("ij,ij->i", oc, oc) - s.radius ** 2
        disc = b * b - c
        valid = disc > 0.0
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = -b - sq
        t1 = -b + sq
        t = np.where(t0 > tmin, t0, t1)
        take = valid & (t > tmin) & (t < best_t)
        if take.any():
            best_t = np.where(take, t, best_t)
            p = o[take] + t[take, None] * d[take]
            normal[take] = (p - s.center) / s.radius
            albedo[take] = s.albedo
            hit |= take

    if scene.has_ground:
        denom = d[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (scene.ground_z - o[:, 2]) / denom
        take = np.isfinite(t) & (t > tmin) & (t < best_t)
        if take.any():
            best_t = np.where(take, t, best_t)
            normal[take] = np.array([0.0, 0.0, 1.0])
            albedo[take] = scene.ground_albedo
            hit |= take

    point = o + np.where(hit, best_t, 0.0)[:, None] * d
    return {"hit": hit, "t": best_t, "point": point,
            "normal": normal, "albedo": albedo}


def _occluded(scene, origins, dirs):
    """Any-hit shadow test. Returns bool array of shape (N,)."""
    return intersect_scene(scene, origins, dirs)["hit"]


# =========================================================================
# 4. Shading models
# =========================================================================
# All shaders take hit points + normals + albedo and return linear RGB.
# For a Lambertian surface,  L_out = (albedo / pi) * E(n).

def shade_ground_truth(scene, points, normals, albedo, spp=256, occlusion=False,
                       rng=None):
    """Reference: Monte Carlo irradiance, cosine-sampled.

    With cosine-weighted sampling the estimator collapses to
    `E = pi * mean(L)`, so outgoing radiance is just `albedo * mean(L)`.

    `occlusion=True` traces a shadow ray per sample -- the thing a light probe
    fundamentally cannot represent, and the honest baseline for notebook 08.
    """
    rng = np.random.default_rng(1234) if rng is None else rng
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    albedo = np.asarray(albedo, dtype=float).reshape(-1, 3)
    n = points.shape[0]
    acc = np.zeros((n, 3))

    # One shared set of local-frame samples, rotated per-pixel: much cheaper
    # than generating spp fresh directions for every one of N pixels.
    u1, u2 = rng.random(spp), rng.random(spp)
    r = np.sqrt(u1)
    ph = 2.0 * np.pi * u2
    local = np.stack([r * np.cos(ph), r * np.sin(ph),
                      np.sqrt(np.maximum(0.0, 1.0 - u1))], axis=-1)   # (spp,3)

    # Branchless orthonormal basis per normal (Duff et al. 2017).
    nz = normals
    sign = np.where(nz[:, 2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + nz[:, 2])
    b = nz[:, 0] * nz[:, 1] * a
    t = np.stack([1.0 + sign * nz[:, 0] ** 2 * a, sign * b, -sign * nz[:, 0]], -1)
    bt = np.stack([b, sign + nz[:, 1] ** 2 * a, -nz[:, 1]], -1)

    for k in range(spp):
        lx, ly, lz = local[k]
        d = lx * t + ly * bt + lz * nz                      # (N,3)
        radiance = np.asarray(scene.env(d), dtype=float)
        if occlusion:
            blocked = _occluded(scene, points + normals * 1e-4, d)
            radiance = np.where(blocked[:, None], 0.0, radiance)
        acc += radiance
    return albedo * (acc / spp)


def shade_sh_probe(coeffs, normals, albedo):
    """Light every point from one SH probe (radiance coefficients, (ncoef,3)).

    This is the shader-side operation: evaluate irradiance from the normal,
    multiply by albedo/pi. No sampling, no rays, constant cost.
    """
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    albedo = np.asarray(albedo, dtype=float).reshape(-1, 3)
    E = sh.irradiance_from_sh(np.asarray(coeffs, dtype=float), normals)
    return albedo * E / np.pi


def build_probe_grid(scene, bounds_min, bounds_max, counts, lmax=2,
                     n_samples=8192, occlusion=True):
    """Bake a regular 3D grid of light probes.

    Returns (positions (nx,ny,nz,3), coeffs (nx,ny,nz,ncoef,3)).

    With `occlusion=True` each probe captures the environment *as seen from its
    own position*, so probes tucked behind geometry correctly go dark -- this is
    what gives a probe grid any spatial variation at all.
    """
    bounds_min = np.asarray(bounds_min, dtype=float)
    bounds_max = np.asarray(bounds_max, dtype=float)
    nx, ny, nz = counts

    axes = [np.linspace(bounds_min[i], bounds_max[i], counts[i]) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    positions = np.stack([X, Y, Z], axis=-1)

    dirs = sh.fibonacci_sphere(n_samples)
    basis = sh.sh_eval_dirs(lmax, dirs)                     # (S, ncoef)
    weight = 4.0 * np.pi / n_samples

    flat_pos = positions.reshape(-1, 3)
    coeffs = np.zeros((flat_pos.shape[0], sh.sh_num_coeffs(lmax), 3))
    for i, p in enumerate(flat_pos):
        radiance = np.asarray(scene.env(dirs), dtype=float)
        if occlusion:
            o = np.broadcast_to(p, dirs.shape)
            blocked = _occluded(scene, o, dirs)
            radiance = np.where(blocked[:, None], 0.0, radiance)
        coeffs[i] = weight * (basis.T @ radiance)

    return positions, coeffs.reshape(nx, ny, nz, -1, 3)


def shade_probe_grid(positions, coeffs, points, normals, albedo):
    """Light points by trilinearly interpolating the nearest 8 probes.

    Interpolating *coefficients* (rather than final colours) is valid because
    SH projection is linear -- a blend of probes is the probe of the blend.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    albedo = np.asarray(albedo, dtype=float).reshape(-1, 3)

    ax = positions[:, 0, 0, 0]
    ay = positions[0, :, 0, 1]
    az = positions[0, 0, :, 2]
    nx, ny, nz = len(ax), len(ay), len(az)

    def axis_weights(vals, axis):
        n = len(axis)
        if n == 1:
            return np.zeros(len(vals), dtype=int), np.zeros(len(vals), dtype=int), \
                np.zeros(len(vals))
        step = axis[1] - axis[0]
        f = np.clip((vals - axis[0]) / step, 0.0, n - 1.0)
        i0 = np.clip(np.floor(f).astype(int), 0, n - 2)
        return i0, i0 + 1, f - i0

    ix0, ix1, fx = axis_weights(points[:, 0], ax)
    iy0, iy1, fy = axis_weights(points[:, 1], ay)
    iz0, iz1, fz = axis_weights(points[:, 2], az)

    blended = np.zeros((points.shape[0], coeffs.shape[3], 3))
    for cx, wx in ((ix0, 1 - fx), (ix1, fx)):
        for cy, wy in ((iy0, 1 - fy), (iy1, fy)):
            for cz, wz in ((iz0, 1 - fz), (iz1, fz)):
                w = (wx * wy * wz)[:, None, None]
                blended += w * coeffs[cx, cy, cz]

    ncoef = blended.shape[1]
    lmax = int(np.sqrt(ncoef)) - 1
    a = sh.cosine_lobe_zh(lmax)
    scale = np.array([a[sh.sh_band_of(i)] for i in range(ncoef)])
    irr_coeffs = blended * scale[None, :, None]

    Y = sh.sh_eval_dirs(lmax, normals)                      # (N, ncoef)
    E = np.einsum("ni,nic->nc", Y, irr_coeffs)
    return albedo * E / np.pi


def render(scene, width, height, shader, camera=None, background=None, **cam_kw):
    """Trace primary rays and shade the hits. Returns linear RGB (H, W, 3).

    `shader(points, normals, albedo) -> (N,3)` is called once for all hits.
    """
    if camera is None:
        o, d = camera_rays(width, height, **cam_kw)
    else:
        o, d = camera
    o_f = o.reshape(-1, 3)
    d_f = d.reshape(-1, 3)

    h = intersect_scene(scene, o_f, d_f)
    img = np.zeros((width * height, 3))

    miss = ~h["hit"]
    if miss.any():
        img[miss] = (np.asarray(scene.env(d_f[miss]), dtype=float)
                     if background is None else np.asarray(background))

    got = h["hit"]
    if got.any():
        img[got] = shader(h["point"][got], h["normal"][got], h["albedo"][got])
    return img.reshape(height, width, 3)


# =========================================================================
# 5. Display
# =========================================================================

def tonemap(img, exposure=1.0, mode="reinhard"):
    """HDR linear -> [0,1] linear."""
    x = np.asarray(img, dtype=float) * exposure
    x = np.maximum(x, 0.0)
    if mode == "reinhard":
        return x / (1.0 + x)
    if mode == "aces":                      # Narkowicz's cheap ACES fit
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    return np.clip(x, 0.0, 1.0)


def srgb_encode(linear):
    """Linear -> sRGB, for correct display in matplotlib."""
    x = np.clip(np.asarray(linear, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
