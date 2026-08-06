"""
shlib -- a small, dependency-light spherical harmonics library for the
"Spherical Harmonics for Games" notebook series.

Everything here is derived from scratch in the notebooks. This module is the
accumulated result: notebook N derives a concept by hand, checks its hand-rolled
version against the function here, and then imports from here onward so the
later notebooks stay readable.

--------------------------------------------------------------------------
COORDINATE CONVENTION  (read this once, save yourself an afternoon)
--------------------------------------------------------------------------
Internally this library uses the *math/physics* convention:

    +Z is up.
    theta ("polar" / "zenith") is measured DOWN FROM +Z,  theta in [0, pi]
    phi   ("azimuth")          is measured in the XY plane from +X toward +Y,
                               phi in [0, 2*pi)

    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)

Every textbook SH formula you will ever look up is written in this convention,
so the derivations stay recognizable. Most game engines are Y-up instead. Use
`to_gamedir` / `from_gamedir` at the boundary, and see notebook 07 for what
handedness does to the band-1 coefficients.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sh_index", "sh_num_coeffs", "sh_band_of",
    "dir_from_spherical", "spherical_from_dir",
    "to_gamedir", "from_gamedir",
    "assoc_legendre", "sh_normalization", "sh_eval", "sh_eval_dirs",
    "fibonacci_sphere", "uniform_sphere_dirs", "cosine_hemisphere_dirs",
    "project_env", "reconstruct",
    "cosine_lobe_zh", "convolve_cosine", "irradiance_from_sh",
    "irradiance_3x3", "irradiance_matrices",
    "hanning_window", "lanczos_window",
    "rotation_matrix_sh", "rotate_sh", "rotate_zh_to_axis",
    "equirect_grid", "env_to_equirect", "sh_to_equirect",
    "rms_error", "relative_error",
]


# =========================================================================
# 1. Indexing
# =========================================================================
# SH coefficients are stored in a flat array ordered by band, and within a
# band by m ascending from -l to +l:
#
#   index:  0    1     2    3     4     5     6    7    8
#   (l,m): (0,0)(1,-1)(1,0)(1,1)(2,-2)(2,-1)(2,0)(2,1)(2,2)
#
# This is the ordering used by essentially all graphics literature.

def sh_index(l: int, m: int) -> int:
    """Flat array index for band `l`, order `m`. Requires -l <= m <= l."""
    if not (-l <= m <= l):
        raise ValueError(f"m={m} out of range for l={l}")
    return l * l + l + m


def sh_num_coeffs(lmax: int) -> int:
    """Number of coefficients for bands 0..lmax inclusive. lmax=2 -> 9."""
    return (lmax + 1) ** 2


def sh_band_of(index: int) -> int:
    """Which band `l` a flat index belongs to."""
    return int(np.floor(np.sqrt(index)))


# =========================================================================
# 2. Directions
# =========================================================================

def dir_from_spherical(theta, phi):
    """(theta, phi) -> unit vector(s), shape (..., 3). Z-up convention."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


def spherical_from_dir(d):
    """Unit vector(s) -> (theta, phi). Inverse of `dir_from_spherical`."""
    d = np.asarray(d, dtype=float)
    theta = np.arccos(np.clip(d[..., 2], -1.0, 1.0))
    phi = np.arctan2(d[..., 1], d[..., 0]) % (2.0 * np.pi)
    return theta, phi


def to_gamedir(d):
    """Z-up math vector -> Y-up game vector. (x,y,z)_math -> (x,z,y)_game."""
    d = np.asarray(d, dtype=float)
    return np.stack([d[..., 0], d[..., 2], d[..., 1]], axis=-1)


def from_gamedir(d):
    """Y-up game vector -> Z-up math vector. Inverse of `to_gamedir`."""
    d = np.asarray(d, dtype=float)
    return np.stack([d[..., 0], d[..., 2], d[..., 1]], axis=-1)


# =========================================================================
# 3. Associated Legendre polynomials
# =========================================================================
# The three-term recurrence used throughout graphics (see Sloan, "Stupid
# Spherical Harmonics Tricks", 2008).
#
#   P(m,m)   = (2m-1)!! * (1-x^2)^(m/2)              with P(0,0) = 1
#   P(m+1,m) = x * (2m+1) * P(m,m)
#   P(l,m)   = ( x*(2l-1)*P(l-1,m) - (l+m-1)*P(l-2,m) ) / (l-m)
#
# CONVENTION: no Condon-Shortley phase. Physics texts (and scipy's `lpmv` and
# `sph_harm_y`) include a factor of (-1)^m; graphics drops it, which is what
# makes the familiar table come out with clean signs -- y(1,1) = +0.488603*x
# rather than -0.488603*x. Many published graphics snippets write the diagonal
# step as `(1-2m)`, which is -(2m-1) and therefore silently reintroduces the
# phase. If you cross-check against scipy, multiply its result by (-1)^m.

def assoc_legendre(l: int, m: int, x):
    """Associated Legendre polynomial P_l^m(x), no Condon-Shortley phase.

    `x` is cos(theta) and may be an array. Requires 0 <= m <= l.
    """
    if m < 0 or m > l:
        raise ValueError(f"assoc_legendre requires 0 <= m <= l, got l={l}, m={m}")
    x = np.asarray(x, dtype=float)

    # P(m,m) -- climb the diagonal.
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.maximum(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(m):
            pmm = pmm * fact * somx2      # fact runs 1, 3, 5, ... = (2m-1)!!
            fact += 2.0
    if l == m:
        return pmm

    # P(m+1,m) -- one step up in l.
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1

    # Walk up to P(l,m).
    pll = np.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


def sh_normalization(l: int, m: int) -> float:
    """The constant K_l^m that makes the real SH basis orthonormal.

        K_l^m = sqrt( (2l+1)/(4*pi) * (l-|m|)! / (l+|m|)! )

    The factorial ratio is computed as a product to stay stable at high l.
    """
    m = abs(m)
    ratio = 1.0
    for k in range(l - m + 1, l + m + 1):   # (l+m)! / (l-m)!
        ratio *= k
    return float(np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi) / ratio))


# =========================================================================
# 4. The real SH basis
# =========================================================================
# The real basis used in graphics, built from the complex Y_l^m by pairing
# +m with -m (notebook 04 does this derivation):
#
#   m > 0 :  sqrt(2) * K_l^m * cos( m*phi) * P_l^m(cos theta)
#   m < 0 :  sqrt(2) * K_l^m * sin(-m*phi) * P_l^-m(cos theta)
#   m = 0 :             K_l^0             * P_l^0(cos theta)

def sh_eval(lmax: int, theta, phi):
    """Evaluate all real SH basis functions up to band `lmax`.

    Returns array of shape (..., (lmax+1)^2) matching the input broadcast shape.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    theta, phi = np.broadcast_arrays(theta, phi)
    ct = np.cos(theta)

    out = np.empty(theta.shape + (sh_num_coeffs(lmax),), dtype=float)
    sqrt2 = np.sqrt(2.0)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            k = sh_normalization(l, m)
            if m == 0:
                v = k * assoc_legendre(l, 0, ct)
            elif m > 0:
                v = sqrt2 * k * np.cos(m * phi) * assoc_legendre(l, m, ct)
            else:
                v = sqrt2 * k * np.sin(-m * phi) * assoc_legendre(l, -m, ct)
            out[..., sh_index(l, m)] = v
    return out


def sh_eval_dirs(lmax: int, dirs):
    """Same as `sh_eval` but takes unit vectors of shape (..., 3)."""
    theta, phi = spherical_from_dir(dirs)
    return sh_eval(lmax, theta, phi)


# =========================================================================
# 5. Sampling the sphere
# =========================================================================

def fibonacci_sphere(n: int):
    """`n` near-uniformly spaced directions via the spherical Fibonacci spiral.

    Deterministic and very low discrepancy -- ideal for projection, where we
    want a stable answer rather than a noisy one. Returns shape (n, 3).
    """
    i = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * i / n                     # uniform in z  =>  uniform on sphere
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden = np.pi * (3.0 - np.sqrt(5.0))     # golden angle
    phi = golden * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def uniform_sphere_dirs(n: int, rng=None):
    """`n` directions drawn uniformly at random over the sphere. Shape (n, 3)."""
    rng = np.random.default_rng() if rng is None else rng
    u1, u2 = rng.random(n), rng.random(n)
    z = 1.0 - 2.0 * u1
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * np.pi * u2
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def cosine_hemisphere_dirs(n: int, normal=(0.0, 0.0, 1.0), rng=None):
    """`n` directions cosine-weighted about `normal`. Shape (n, 3).

    The natural sampling density for Lambertian irradiance: the estimator for
    E(n) becomes a plain mean of radiance times pi.
    """
    rng = np.random.default_rng() if rng is None else rng
    u1, u2 = rng.random(n), rng.random(n)
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    local = np.stack([r * np.cos(phi), r * np.sin(phi),
                      np.sqrt(np.maximum(0.0, 1.0 - u1))], axis=-1)

    # Build an orthonormal frame around `normal` (Duff et al. branchless ONB).
    nz = np.asarray(normal, dtype=float)
    nz = nz / np.linalg.norm(nz)
    sign = 1.0 if nz[2] >= 0.0 else -1.0
    a = -1.0 / (sign + nz[2])
    b = nz[0] * nz[1] * a
    t = np.array([1.0 + sign * nz[0] * nz[0] * a, sign * b, -sign * nz[0]])
    bt = np.array([b, sign + nz[1] * nz[1] * a, -nz[1]])
    return local[..., 0:1] * t + local[..., 1:2] * bt + local[..., 2:3] * nz


# =========================================================================
# 6. Projection and reconstruction
# =========================================================================

def project_env(env_fn, lmax: int, n_samples: int = 32768, dirs=None):
    """Project a spherical function onto the SH basis.

        c_i = integral over sphere of  f(d) * y_i(d) dω
            ~= (4*pi / N) * sum over samples of  f(d_k) * y_i(d_k)

    `env_fn` maps directions of shape (N,3) to values of shape (N,) or (N,C).
    Returns coefficients of shape ((lmax+1)^2,) or ((lmax+1)^2, C).
    """
    if dirs is None:
        dirs = fibonacci_sphere(n_samples)
    dirs = np.asarray(dirs, dtype=float)
    n = dirs.shape[0]

    values = np.asarray(env_fn(dirs), dtype=float)
    basis = sh_eval_dirs(lmax, dirs)                    # (N, ncoef)
    weight = 4.0 * np.pi / n

    if values.ndim == 1:
        return weight * (basis.T @ values)              # (ncoef,)
    return weight * (basis.T @ values)                  # (ncoef, C)


def reconstruct(coeffs, dirs):
    """Evaluate the SH series at `dirs`. Inverse of `project_env` (band-limited).

        f(d) ~= sum over i of  c_i * y_i(d)
    """
    coeffs = np.asarray(coeffs, dtype=float)
    lmax = int(np.sqrt(coeffs.shape[0])) - 1
    basis = sh_eval_dirs(lmax, dirs)                    # (..., ncoef)
    return basis @ coeffs                               # (...,) or (..., C)


# =========================================================================
# 7. Irradiance: convolution with the clamped cosine lobe
# =========================================================================
# The single most important result for light probes (Ramamoorthi & Hanrahan,
# "An Efficient Representation for Irradiance Environment Maps", 2001):
#
# Convolving radiance with the clamped cosine lobe max(0, dot(n,d)) is, in SH,
# just a PER-BAND SCALE. Irradiance is therefore a low-pass version of radiance,
# and 9 coefficients capture ~99% of it for any input. That is the whole reason
# light probes are 9 numbers per channel and not a cubemap.

def cosine_lobe_zh(lmax: int):
    """Per-band convolution factors  A_l  for the clamped cosine lobe.

    Returns array of length lmax+1 where  E_lm = A_l * L_lm.

        A_0 = pi,  A_1 = 2pi/3,  A_2 = pi/4,  A_3 = 0,  A_4 = -pi/24, ...

    Odd bands above 1 vanish identically -- the lobe is symmetric enough that
    they contribute nothing. That is why band 3 is never stored.
    """
    a = np.zeros(lmax + 1)
    if lmax >= 0:
        a[0] = np.pi
    if lmax >= 1:
        a[1] = 2.0 * np.pi / 3.0
    for l in range(2, lmax + 1):
        if l % 2 == 1:
            a[l] = 0.0
            continue
        half = l // 2
        # 2*pi * (-1)^(l/2 - 1) / ((l+2)(l-1)) * l! / (2^l * (l/2)!^2)
        num = 1.0
        for k in range(1, l + 1):
            num *= k
        den = (2.0 ** l)
        fh = 1.0
        for k in range(1, half + 1):
            fh *= k
        den *= fh * fh
        a[l] = (2.0 * np.pi * ((-1.0) ** (half - 1)) /
                ((l + 2.0) * (l - 1.0)) * num / den)
    return a


def convolve_cosine(coeffs):
    """Radiance SH -> irradiance SH. Scales each band by `cosine_lobe_zh`."""
    coeffs = np.asarray(coeffs, dtype=float)
    lmax = int(np.sqrt(coeffs.shape[0])) - 1
    a = cosine_lobe_zh(lmax)
    scale = np.array([a[sh_band_of(i)] for i in range(coeffs.shape[0])])
    if coeffs.ndim == 1:
        return coeffs * scale
    return coeffs * scale[:, None]


def irradiance_from_sh(coeffs, normals):
    """Irradiance E(n) at surface normals, from *radiance* SH coefficients."""
    return reconstruct(convolve_cosine(coeffs), normals)


# The 9-coefficient irradiance evaluation, folded into a 4x4 quadratic form.
# This is what ships in shaders: one matrix multiply per colour channel.
_C1, _C2, _C3, _C4, _C5 = 0.429043, 0.511664, 0.743125, 0.886227, 0.247708


def irradiance_matrices(coeffs):
    """Build the 4x4 matrices M such that  E = [n 1]^T M [n 1].

    `coeffs` is (9,) for one channel or (9, C) for C channels.
    Returns (4,4) or (C,4,4).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    single = coeffs.ndim == 1
    c = coeffs[:, None] if single else coeffs
    if c.shape[0] < 9:
        raise ValueError("irradiance_matrices needs at least 9 coefficients (L2)")

    L00, L1m1, L10, L11 = c[0], c[1], c[2], c[3]
    L2m2, L2m1, L20, L21, L22 = c[4], c[5], c[6], c[7], c[8]

    nchan = c.shape[1]
    m = np.zeros((nchan, 4, 4))
    m[:, 0, 0] = _C1 * L22
    m[:, 0, 1] = _C1 * L2m2
    m[:, 0, 2] = _C1 * L21
    m[:, 0, 3] = _C2 * L11
    m[:, 1, 0] = _C1 * L2m2
    m[:, 1, 1] = -_C1 * L22
    m[:, 1, 2] = _C1 * L2m1
    m[:, 1, 3] = _C2 * L1m1
    m[:, 2, 0] = _C1 * L21
    m[:, 2, 1] = _C1 * L2m1
    m[:, 2, 2] = _C3 * L20
    m[:, 2, 3] = _C2 * L10
    m[:, 3, 0] = _C2 * L11
    m[:, 3, 1] = _C2 * L1m1
    m[:, 3, 2] = _C2 * L10
    m[:, 3, 3] = _C4 * L00 - _C5 * L20
    return m[0] if single else m


def irradiance_3x3(coeffs, normals):
    """Evaluate irradiance via the quadratic form. Equivalent to
    `irradiance_from_sh` for lmax=2, but in the shape a shader would use."""
    normals = np.asarray(normals, dtype=float)
    m = irradiance_matrices(coeffs)
    n4 = np.concatenate([normals, np.ones(normals.shape[:-1] + (1,))], axis=-1)
    if m.ndim == 2:
        return np.einsum("...i,ij,...j->...", n4, m, n4)
    return np.einsum("...i,cij,...j->...c", n4, m, n4)


# =========================================================================
# 8. Windowing (ringing suppression)
# =========================================================================
# Truncating a series at band L is a brick-wall low-pass filter, and brick-wall
# filters ring (Gibbs phenomenon). A bright sun in an otherwise dark sky becomes
# a bright blob surrounded by a dark halo -- and, worse, negative radiance.
# Windowing tapers the high bands instead of cutting them off.

def hanning_window(coeffs, window_width: float):
    """Hanning (raised cosine) window. `window_width` ~ 4..8 for L2 probes.
    Smaller = more aggressive smoothing = less ringing, more blur."""
    coeffs = np.asarray(coeffs, dtype=float)
    n = coeffs.shape[0]
    w = np.array([
        (1.0 + np.cos(np.pi * sh_band_of(i) / window_width)) * 0.5
        if sh_band_of(i) < window_width else 0.0
        for i in range(n)
    ])
    return coeffs * w if coeffs.ndim == 1 else coeffs * w[:, None]


def lanczos_window(coeffs, window_width: float):
    """Lanczos (sinc) window. Gentler than Hanning at the same width."""
    coeffs = np.asarray(coeffs, dtype=float)
    n = coeffs.shape[0]
    w = np.empty(n)
    for i in range(n):
        l = sh_band_of(i)
        if l == 0:
            w[i] = 1.0
        else:
            t = np.pi * l / window_width
            w[i] = np.sin(t) / t
    return coeffs * w if coeffs.ndim == 1 else coeffs * w[:, None]


# =========================================================================
# 9. Rotation
# =========================================================================
# Rotation does not mix bands -- each band l rotates among its own 2l+1
# coefficients via a (2l+1)x(2l+1) matrix. Band 0 is invariant (a constant is a
# constant). Band 1 rotates exactly like a direction vector.
#
# The construction below is the "sample and solve" method: exact (because
# rotation is band-preserving), short, and easy to trust. Production code uses
# closed-form recurrences (Ivanic & Ruedenberg) for speed.

def rotation_matrix_sh(lmax: int, R):
    """Block-diagonal SH rotation matrix for the 3x3 rotation `R`.

    Returns M of shape (n, n) with n = (lmax+1)^2, such that if g(d) = f(R^T d)
    then  coeffs(g) = M @ coeffs(f).
    """
    R = np.asarray(R, dtype=float)
    n = sh_num_coeffs(lmax)
    M = np.zeros((n, n))

    for l in range(lmax + 1):
        cnt = 2 * l + 1
        lo = l * l
        # Pick well-separated sample directions, then add more for conditioning.
        d = fibonacci_sphere(max(cnt * 4, 16))
        A = sh_eval_dirs(l, d)[:, lo:lo + cnt]          # basis at d
        B = sh_eval_dirs(l, d @ R)[:, lo:lo + cnt]      # basis at R^T d
        # g(d_i) = f(R^T d_i) expands two ways:  A @ c' = B @ c,  so c' = A^-1 B.
        block, *_ = np.linalg.lstsq(A, B, rcond=None)
        M[lo:lo + cnt, lo:lo + cnt] = block
    return M


def rotate_sh(coeffs, R):
    """Rotate an SH-encoded function by the 3x3 rotation matrix `R`."""
    coeffs = np.asarray(coeffs, dtype=float)
    lmax = int(np.sqrt(coeffs.shape[0])) - 1
    return rotation_matrix_sh(lmax, R) @ coeffs


def rotate_zh_to_axis(zh, axis, lmax: int = None):
    """Point a zonal harmonic lobe (symmetric about +Z) along `axis`.

    This is the cheap trick: a ZH lobe needs only l+1 numbers and rotating it
    to an arbitrary direction costs one basis evaluation, no matrices.

        c_lm = sqrt(4*pi / (2l+1)) * zh_l * y_lm(axis)
    """
    zh = np.asarray(zh, dtype=float)
    lmax = (len(zh) - 1) if lmax is None else lmax
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    y = sh_eval_dirs(lmax, axis)
    out = np.zeros(sh_num_coeffs(lmax))
    for l in range(lmax + 1):
        s = np.sqrt(4.0 * np.pi / (2.0 * l + 1.0)) * zh[l]
        for m in range(-l, l + 1):
            out[sh_index(l, m)] = s * y[sh_index(l, m)]
    return out


# =========================================================================
# 10. Equirectangular helpers (for plotting)
# =========================================================================

def equirect_grid(width: int = 256, height: int = 128):
    """Directions on an equirectangular grid. Returns (H, W, 3)."""
    phi = (np.arange(width) + 0.5) / width * 2.0 * np.pi
    theta = (np.arange(height) + 0.5) / height * np.pi
    T, P = np.meshgrid(theta, phi, indexing="ij")
    return dir_from_spherical(T, P)


def env_to_equirect(env_fn, width: int = 256, height: int = 128):
    """Rasterize a direction-valued function to an equirect image."""
    d = equirect_grid(width, height)
    flat = np.asarray(env_fn(d.reshape(-1, 3)), dtype=float)
    return flat.reshape(height, width) if flat.ndim == 1 else \
        flat.reshape(height, width, -1)


def sh_to_equirect(coeffs, width: int = 256, height: int = 128):
    """Rasterize an SH-encoded function to an equirect image."""
    return env_to_equirect(lambda d: reconstruct(coeffs, d), width, height)


# =========================================================================
# 11. Error metrics
# =========================================================================

def rms_error(a, b, weights=None):
    """Root-mean-square difference, optionally solid-angle weighted."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d2 = (a - b) ** 2
    if weights is None:
        return float(np.sqrt(np.mean(d2)))
    w = np.asarray(weights, dtype=float)
    while w.ndim < d2.ndim:
        w = w[..., None]
    return float(np.sqrt(np.sum(d2 * w) / np.sum(w * np.ones_like(d2))))


def relative_error(approx, truth, weights=None):
    """RMS error as a fraction of the RMS magnitude of `truth`."""
    truth = np.asarray(truth, dtype=float)
    denom = rms_error(truth, np.zeros_like(truth), weights)
    if denom == 0.0:
        return 0.0
    return rms_error(approx, truth, weights) / denom
