"""Independent validation of shlib.py. Every claim the notebooks will make
should be checked here first against brute force or scipy."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shlib as sh

FAIL = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}   {detail}")
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- indexing
check("sh_index ordering",
      [sh.sh_index(l, m) for l in range(3) for m in range(-l, l + 1)] == list(range(9)))
check("sh_num_coeffs", sh.sh_num_coeffs(2) == 9 and sh.sh_num_coeffs(4) == 25)
check("sh_band_of", [sh.sh_band_of(i) for i in range(9)] == [0, 1, 1, 1, 2, 2, 2, 2, 2])

# ------------------------------------------------------ direction round trip
rng = np.random.default_rng(0)
d = sh.uniform_sphere_dirs(500, rng)
t, p = sh.spherical_from_dir(d)
check("dir <-> spherical round trip", np.allclose(sh.dir_from_spherical(t, p), d, atol=1e-12))
check("gamedir round trip", np.allclose(sh.from_gamedir(sh.to_gamedir(d)), d))

# ------------------------------------------------------------ orthonormality
LMAX = 5
N = 400_000
dirs = sh.fibonacci_sphere(N)
Y = sh.sh_eval_dirs(LMAX, dirs)
G = (4.0 * np.pi / N) * (Y.T @ Y)
I = np.eye(sh.sh_num_coeffs(LMAX))
err = np.abs(G - I).max()
check("real SH orthonormal to L5", err < 2e-3, f"max|G-I| = {err:.2e}")

# ---------------------------------------------- cross-check against scipy
# scipy's sph_harm_y is the complex physics basis WITH Condon-Shortley phase.
# Real basis relation (our convention, CS phase removed):
#   m>0:  sqrt(2) * (-1)^m * Re[Y_l^m]
#   m<0:  sqrt(2) * (-1)^m * Im[Y_l^|m|]
#   m=0:  Re[Y_l^0]
from scipy.special import sph_harm_y

tt, pp = sh.spherical_from_dir(sh.fibonacci_sphere(2000))
mine = sh.sh_eval(4, tt, pp)
ok = True
worst = 0.0
for l in range(5):
    for m in range(-l, l + 1):
        Ylm = sph_harm_y(l, abs(m), tt, pp)
        if m == 0:
            ref = Ylm.real
        elif m > 0:
            ref = np.sqrt(2) * ((-1) ** m) * Ylm.real
        else:
            ref = np.sqrt(2) * ((-1) ** abs(m)) * Ylm.imag
        e = np.abs(mine[:, sh.sh_index(l, m)] - ref).max()
        worst = max(worst, e)
        if e > 1e-10:
            ok = False
            print(f"       mismatch l={l} m={m}: {e:.2e}")
check("matches scipy sph_harm_y (real form)", ok, f"worst = {worst:.2e}")

# ------------------------------------ canonical graphics table (the real test)
# The polynomial forms every engine hard-codes. If these don't match, shader
# code copied from any paper or engine will be wrong.
dv = sh.fibonacci_sphere(5000)
x, y_, z = dv[:, 0], dv[:, 1], dv[:, 2]
table = {
    (0, 0):  0.282095 * np.ones_like(x),
    (1, -1): 0.488603 * y_,
    (1, 0):  0.488603 * z,
    (1, 1):  0.488603 * x,
    (2, -2): 1.092548 * x * y_,
    (2, -1): 1.092548 * y_ * z,
    (2, 0):  0.315392 * (3.0 * z * z - 1.0),
    (2, 1):  1.092548 * x * z,
    (2, 2):  0.546274 * (x * x - y_ * y_),
}
B = sh.sh_eval_dirs(2, dv)
ok = True
for (l, m), ref in table.items():
    e = np.abs(B[:, sh.sh_index(l, m)] - ref).max()
    if e > 1e-5:
        ok = False
        print(f"       table mismatch y({l},{m}): max err {e:.2e}")
check("matches canonical graphics polynomial table", ok)

# ------------------------------------------------- projection/reconstruction
def env(dv):
    """Smooth band-limited test signal: exactly representable at L2."""
    c = np.array([0.7, 0.2, -0.4, 0.1, 0.05, -0.3, 0.25, 0.15, -0.1])
    return sh.reconstruct(c, dv)

c_true = np.array([0.7, 0.2, -0.4, 0.1, 0.05, -0.3, 0.25, 0.15, -0.1])
c_fit = sh.project_env(env, 2, n_samples=100_000)
check("projection recovers band-limited signal",
      np.allclose(c_fit, c_true, atol=1e-4), f"max err {np.abs(c_fit-c_true).max():.2e}")

# multi-channel
def env_rgb(dv):
    return np.stack([env(dv), 2 * env(dv), -env(dv)], axis=-1)

c_rgb = sh.project_env(env_rgb, 2, n_samples=100_000)
check("multi-channel projection", c_rgb.shape == (9, 3) and
      np.allclose(c_rgb[:, 1], 2 * c_true, atol=1e-4))

# -------------------------------------------- THE key claim: cosine convolution
# Brute-force irradiance:  E(n) = integral L(d) * max(0, n.d) dω
def brute_irradiance(env_fn, normal, n=800_000):
    dv = sh.fibonacci_sphere(n)
    ndl = np.maximum(0.0, dv @ normal)
    return (4.0 * np.pi / n) * np.sum(np.asarray(env_fn(dv)) * ndl)

def sky(dv):
    """Non-band-limited: a sun disc plus a gradient. Realistic stress test."""
    sun = np.where(dv @ np.array([0.3, 0.4, 0.86]) > 0.97, 30.0, 0.0)
    grad = 0.6 + 0.4 * dv[..., 2]
    return sun + grad

c_sky = sh.project_env(sky, 8, n_samples=400_000)
test_n = sh.fibonacci_sphere(24)
e_sh = sh.irradiance_from_sh(c_sky, test_n)
e_bf = np.array([brute_irradiance(sky, n) for n in test_n])
rel = np.abs(e_sh - e_bf).max() / np.abs(e_bf).max()
check("cosine convolution == brute-force irradiance", rel < 0.02,
      f"max rel err {rel*100:.2f}%")

# the famous 9-coefficient accuracy claim
c2 = sh.project_env(sky, 2, n_samples=400_000)
e_l2 = sh.irradiance_from_sh(c2, test_n)
acc = 1.0 - np.abs(e_l2 - e_bf).max() / np.abs(e_bf).max()
check("L2 (9 coeffs) irradiance accuracy > 95%", acc > 0.95, f"accuracy {acc*100:.2f}%")

# -------------------------------------------------- 3x3 quadratic form form
e_mat = sh.irradiance_3x3(c2, test_n)
check("irradiance_3x3 == irradiance_from_sh",
      np.allclose(e_mat, e_l2, atol=1e-6), f"max diff {np.abs(e_mat-e_l2).max():.2e}")

c2_rgb = sh.project_env(lambda dv: np.stack([sky(dv), 0.5*sky(dv), 0.2*sky(dv)], -1),
                        2, n_samples=200_000)
e_rgb = sh.irradiance_3x3(c2_rgb, test_n)
check("irradiance_3x3 multi-channel", e_rgb.shape == (24, 3) and
      np.allclose(e_rgb[:, 1], 0.5 * e_rgb[:, 0], rtol=1e-6))

# ---------------------------------------------------------- cosine lobe ZH
a = sh.cosine_lobe_zh(6)
expected = [np.pi, 2*np.pi/3, np.pi/4, 0.0, -np.pi/24, 0.0, np.pi/64]
check("cosine_lobe_zh values", np.allclose(a, expected),
      f"{np.round(a,6).tolist()}")

# ------------------------------------------------------------------ rotation
def rand_rot(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

R = rand_rot(rng)
M = sh.rotation_matrix_sh(4, R)
check("SH rotation matrix is orthogonal",
      np.allclose(M @ M.T, np.eye(25), atol=1e-8), f"max {np.abs(M@M.T-np.eye(25)).max():.2e}")
check("SH rotation is block diagonal (band 0 fixed)", abs(M[0, 0] - 1.0) < 1e-10)

# rotating coefficients == projecting the rotated function
c = rng.normal(size=25)
c_rot = sh.rotate_sh(c, R)
probe = sh.fibonacci_sphere(3000)
lhs = sh.reconstruct(c_rot, probe)
rhs = sh.reconstruct(c, probe @ R)      # f(R^T d)
check("rotate_sh matches function rotation",
      np.allclose(lhs, rhs, atol=1e-8), f"max {np.abs(lhs-rhs).max():.2e}")

# irradiance must be rotation-equivariant
check("rotation commutes with cosine convolution",
      np.allclose(sh.convolve_cosine(sh.rotate_sh(c, R)),
                  sh.rotate_sh(sh.convolve_cosine(c), R), atol=1e-8))

# ----------------------------------------------------------------- ZH rotate
axis = np.array([0.5, -0.3, 0.81]); axis /= np.linalg.norm(axis)
zh = sh.cosine_lobe_zh(4) / np.array([np.sqrt(4*np.pi/(2*l+1)) for l in range(5)])
c_zh = sh.rotate_zh_to_axis(zh, axis)
# compare against directly projecting the clamped cosine lobe about `axis`
c_direct = sh.project_env(lambda dv: np.maximum(0.0, dv @ axis), 4, n_samples=400_000)
check("rotate_zh_to_axis matches direct projection",
      np.allclose(c_zh, c_direct, atol=2e-3), f"max {np.abs(c_zh-c_direct).max():.2e}")

# ------------------------------------------------------------------ windows
cw = sh.hanning_window(np.ones(25), 4.0)
check("hanning window: band0 untouched, tapers up", cw[0] == 1.0 and cw[24] < cw[4] < 1.0)
lw = sh.lanczos_window(np.ones(25), 6.0)
check("lanczos window: band0 untouched, tapers up", lw[0] == 1.0 and lw[24] < lw[4] < 1.0)

# ---------------------------------------------------------------- equirect
img = sh.sh_to_equirect(c_true, 64, 32)
check("sh_to_equirect shape", img.shape == (32, 64))
img_rgb = sh.sh_to_equirect(c_rgb, 64, 32)
check("sh_to_equirect rgb shape", img_rgb.shape == (32, 64, 3))

# ------------------------------------------------------------------ metrics
check("rms_error zero on identical", sh.rms_error(np.ones(10), np.ones(10)) == 0.0)
check("relative_error sane", abs(sh.relative_error(np.ones(10)*1.1, np.ones(10)) - 0.1) < 1e-9)

# ----------------------------------------------------------- cosine sampling
n_hat = np.array([0.2, -0.5, 0.84]); n_hat /= np.linalg.norm(n_hat)
cs = sh.cosine_hemisphere_dirs(200_000, n_hat, rng)
check("cosine_hemisphere_dirs unit length", np.allclose(np.linalg.norm(cs, axis=-1), 1.0))
check("cosine_hemisphere_dirs in hemisphere", (cs @ n_hat >= -1e-9).all())
# mean cos should be 2/3 for cosine-weighted sampling
mc = (cs @ n_hat).mean()
check("cosine_hemisphere_dirs density", abs(mc - 2/3) < 0.01, f"mean cos = {mc:.4f}")
# irradiance via cosine sampling == pi * mean(L)
e_cos = np.pi * np.mean(sky(cs))
e_ref = brute_irradiance(sky, n_hat)
check("cosine-sampled irradiance estimator", abs(e_cos - e_ref) / e_ref < 0.02,
      f"{e_cos:.4f} vs {e_ref:.4f}")

print()
print("=" * 60)
print(f"{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else " -- all good"))
sys.exit(1 if FAIL else 0)
