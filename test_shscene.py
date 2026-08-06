import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shlib as sh
import shscene as ss

FAIL = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")
    if not cond: FAIL.append(name)

d = sh.fibonacci_sphere(1000)
for name, fn in ss.ENVIRONMENTS.items():
    v = np.asarray(fn(d))
    check(f"env '{name}' shape+finite+nonneg",
          v.shape == (1000, 3) and np.isfinite(v).all() and (v >= 0).all(),
          f"range [{v.min():.3f}, {v.max():.1f}]")

# camera
o, dd = ss.camera_rays(64, 48)
check("camera_rays shapes", o.shape == (48, 64, 3) and dd.shape == (48, 64, 3))
check("camera dirs unit", np.allclose(np.linalg.norm(dd, axis=-1), 1.0))

# intersection sanity: ray straight down hits ground at z=0
scene = ss.default_scene()
h = ss.intersect_scene(scene, np.array([[0.0, 5.0, 4.0]]), np.array([[0.0, 0.0, -1.0]]))
check("ground hit", h["hit"][0] and abs(h["point"][0, 2]) < 1e-6
      and np.allclose(h["normal"][0], [0, 0, 1]), f"t={h['t'][0]:.3f}")

# ray at the centre sphere from the camera
h2 = ss.intersect_scene(scene, np.array([[0.0, -7.5, 1.0]]), np.array([[0.0, 1.0, 0.0]]))
expected_t = 7.5 - 1.0  # sphere at (0,0,1) r=1
check("sphere hit", h2["hit"][0] and abs(h2["t"][0] - expected_t) < 1e-6,
      f"t={h2['t'][0]:.4f} expected {expected_t}")
check("sphere normal points back at camera", np.allclose(h2["normal"][0], [0, -1, 0]))

# ---- the central claim of notebook 08: SH probe ~= ground truth (unoccluded)
scene_g = ss.default_scene(env=ss.sky_studio)
pts = np.zeros((64, 3))
nrm = sh.fibonacci_sphere(64)
alb = np.ones((64, 3)) * 0.8

gt = ss.shade_ground_truth(scene_g, pts, nrm, alb, spp=4096, occlusion=False)
c = sh.project_env(scene_g.env, 2, n_samples=200_000)
approx = ss.shade_sh_probe(c, nrm, alb)
rel = np.abs(approx - gt).max() / gt.max()
rms = sh.relative_error(approx, gt)
check("SH probe roughly tracks MC ground truth", rel < 0.10, f"max rel {rel*100:.2f}%")

# The literature's "9 coefficients capture ~99% of irradiance" is an RMS claim
# about typical *natural* illumination. Measure it per environment so the
# notebooks quote real numbers instead of repeating the slogan.
print("\n  L2 irradiance accuracy by environment (RMS rel / max rel):")
budget = {"gradient": (0.01, 0.02), "sun": (0.20, 0.40),
          "studio": (0.08, 0.10), "split": (0.10, 0.25)}
for name, fn in ss.ENVIRONMENTS.items():
    s = ss.default_scene(env=fn)
    g = ss.shade_ground_truth(s, pts, nrm, alb, spp=8192, occlusion=False)
    cc = sh.project_env(fn, 2, n_samples=200_000)
    ap = ss.shade_sh_probe(cc, nrm, alb)
    r_rms = sh.relative_error(ap, g)
    r_max = np.abs(ap - g).max() / g.max()
    lim_rms, lim_max = budget[name]
    print(f"    {name:9s}  RMS {r_rms*100:6.2f}%   max {r_max*100:6.2f}%")
    check(f"  '{name}' within expected budget", r_rms < lim_rms and r_max < lim_max)

# higher band should be closer
c4 = sh.project_env(scene_g.env, 4, n_samples=200_000)
approx4 = ss.shade_sh_probe(c4, nrm, alb)
rel4 = np.abs(approx4 - gt).max() / gt.max()
check("L4 no worse than L2", rel4 <= rel + 1e-3, f"L2 {rel*100:.2f}% -> L4 {rel4*100:.2f}%")

# ---- render smoke test
t0 = time.time()
img = ss.render(scene_g, 96, 64, lambda p, n, a: ss.shade_sh_probe(c, n, a))
check("render output", img.shape == (64, 96, 3) and np.isfinite(img).all()
      and img.max() > 0, f"{time.time()-t0:.2f}s, max {img.max():.3f}")

t0 = time.time()
img_gt = ss.render(scene_g, 96, 64,
                   lambda p, n, a: ss.shade_ground_truth(scene_g, p, n, a, spp=64,
                                                         occlusion=True))
check("render ground truth w/ occlusion", img_gt.shape == (64, 96, 3)
      and np.isfinite(img_gt).all(), f"{time.time()-t0:.2f}s")

# ---- probe grid
t0 = time.time()
pos, coef = ss.build_probe_grid(scene_g, (-4, -3, 0.2), (4, 3, 3.0), (3, 3, 2),
                                lmax=2, n_samples=2048, occlusion=True)
check("probe grid shapes", pos.shape == (3, 3, 2, 3) and coef.shape == (3, 3, 2, 9, 3),
      f"{time.time()-t0:.2f}s")
check("probe grid varies in space",
      np.abs(coef[0, 0, 0] - coef[2, 2, 1]).max() > 1e-3,
      f"delta {np.abs(coef[0,0,0]-coef[2,2,1]).max():.4f}")

shaded = ss.shade_probe_grid(pos, coef, np.zeros((10, 3)),
                             sh.fibonacci_sphere(10), np.ones((10, 3)) * 0.7)
check("shade_probe_grid output", shaded.shape == (10, 3) and np.isfinite(shaded).all()
      and (shaded >= 0).all())

# grid of identical probes == single probe
pos2, coef2 = ss.build_probe_grid(scene_g, (-4, -3, 0.2), (4, 3, 3.0), (2, 2, 2),
                                  lmax=2, n_samples=4096, occlusion=False)
uni = ss.shade_probe_grid(pos2, coef2, np.zeros((32, 3)), sh.fibonacci_sphere(32),
                          np.ones((32, 3)) * 0.8)
single = ss.shade_sh_probe(coef2[0, 0, 0], sh.fibonacci_sphere(32), np.ones((32, 3)) * 0.8)
check("unoccluded grid == single probe", np.allclose(uni, single, atol=1e-9),
      f"max {np.abs(uni-single).max():.2e}")

# ---- display
check("tonemap in range", (0 <= ss.tonemap(img, 1.0)).all() and (ss.tonemap(img, 1.0) <= 1).all())
check("srgb_encode in range", (0 <= ss.srgb_encode(ss.tonemap(img))).all()
      and (ss.srgb_encode(ss.tonemap(img)) <= 1).all())
check("srgb midpoint", abs(ss.srgb_encode(np.array([0.5]))[0] - 0.7354) < 1e-3,
      f"{ss.srgb_encode(np.array([0.5]))[0]:.4f}")

print("\n" + "=" * 60)
print(f"{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else " -- all good"))
sys.exit(1 if FAIL else 0)
