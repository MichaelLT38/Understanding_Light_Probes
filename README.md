# Spherical Harmonics for Games

A hands-on guide to spherical harmonics and light probes, built as a series of
Jupyter notebooks. Starts from complex exponentials and ends with a small scene lit
entirely by a baked probe grid, with error measured against a ray-traced reference
at every step.

The through-line is one question: **why is a light probe 27 floats?**

---

## Setup

A virtual environment with everything installed is already in `.venv/`, and a
Jupyter kernel named **Python 3 (SH4Games)** is registered.

```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab
```

Or just open any `.ipynb` in VS Code and pick the **Python 3 (SH4Games)** kernel.

Every notebook ships with its outputs already executed, so you can read the whole
series without running anything.

To rebuild the environment from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install jupyter jupyterlab numpy matplotlib scipy imageio ipywidgets
.\.venv\Scripts\python.exe -m ipykernel install --user --name sh4games --display-name "Python 3 (SH4Games)"
```

---

## Reading order

Work through them in order — each builds on the last.

| # | Notebook | What it answers |
|---|----------|-----------------|
| 00 | [Why light probes exist](00_why_light_probes.ipynb) | Why compress lighting at all? Shows the payoff up front. |
| 01 | [Functions on a sphere](01_functions_on_a_sphere.ipynb) | Solid angle, spherical integration, Monte Carlo, and the ground truth everything else is measured against. |
| 02 | [From Fourier series to SH](02_fourier_to_sh.ipynb) | Complex exponentials, orthogonality, projection, truncation and ringing — all in 1D where it's easy to see. |
| 03 | [Legendre & complex SH](03_legendre_and_complex_sh.ipynb) | Where $Y_\ell^m$ comes from: separation of variables, associated Legendre polynomials, the normalisation constant. |
| 04 | [The real SH basis](04_real_sh_basis.ipynb) | The nine polynomials every engine hard-codes, and why graphics uses a different basis than physics. |
| 05 | [Projection & ringing](05_projection_and_ringing.ipynb) | What truncation costs, negative radiance, and windowing. |
| 06 | **[Irradiance convolution](06_irradiance_convolution.ipynb)** | **The payoff.** Why nine coefficients is enough — derived, verified, and with honest error bars. |
| 07 | [Rotation, storage, shaders](07_rotation_storage_shaders.ipynb) | Band-preserving rotation, packing layouts, HLSL/GLSL, and the conventions that break lighting. |
| 08 | [Lighting a scene](08_lighting_a_scene.ipynb) | A baked probe grid vs a ray-traced reference — and what probes fundamentally cannot do. |

**Short on time?** Read 00 for the motivation, 04 for the basis, and 06 for the
result. Those three carry the argument.

---

## The argument, compressed

A light probe stores a function on a sphere. Spherical harmonics are the frequency
basis for such functions: orthonormal, so projection is one independent integral per
coefficient; rotation-closed, so bands survive transformation intact.

Diffuse lighting convolves radiance with a clamped cosine lobe. That kernel is
rotationally symmetric, so in SH it reduces to a **per-band scale** $\hat{A}_\ell$
— which is exactly zero for band 3 and every odd band above it, and under 4% from
band 4 on.

So irradiance is inherently low-frequency *no matter what the lighting is*. Nine
coefficients per channel capture it to about a percent for natural illumination, and
evaluating it costs one quadratic form. That is why light probes are 27 floats.

---

## Code

| File | Contents |
|---|---|
| [shlib.py](shlib.py) | The SH library: Legendre recurrences, the real basis, projection, the cosine convolution, windowing, rotation. Derived piece by piece across the notebooks. |
| [shscene.py](shscene.py) | Procedural HDR environments and a small numpy raytracer, so the final notebook lights a real scene. |
| [shplot.py](shplot.py) | Shared matplotlib style and figure helpers. |
| [test_shlib.py](test_shlib.py) | 30 checks on the math — orthonormality, agreement with SciPy, the canonical polynomial table, convolution vs brute force, rotation. |
| [test_shscene.py](test_shscene.py) | Scene, camera, probe-grid and tonemapping checks, plus the per-environment accuracy table. |

```powershell
.\.venv\Scripts\python.exe test_shlib.py
.\.venv\Scripts\python.exe test_shscene.py
```

Both should report `0 failure(s)`. Run them if you change anything in `shlib.py` —
several of the checks catch sign and convention errors that are otherwise very hard
to see.

---

## Conventions

This series uses the **math convention**: **+Z is up**, $\theta$ measured down from
$+Z$, $\phi$ in the XY plane from $+X$. Every formula you look up in a paper is
written this way. Most engines are Y-up — `shlib.to_gamedir` / `from_gamedir`
convert at the boundary, and notebook 07 covers handedness in detail.

The associated Legendre polynomials **omit the Condon–Shortley phase**, following
graphics convention, so the standard table comes out with clean signs
($y_1^1 = +0.488603\,x$). SciPy includes the phase; multiply by $(-1)^m$ to compare.

---

## Further reading

- Ramamoorthi & Hanrahan, *An Efficient Representation for Irradiance Environment
  Maps* (SIGGRAPH 2001) — the result notebook 06 is built around.
- Green, *Spherical Harmonic Lighting: The Gritty Details* (GDC 2003) — the standard
  practical introduction.
- Sloan, *Stupid Spherical Harmonics Tricks* (GDC 2008) — zonal harmonics, rotation,
  windowing.
- Sloan, Kautz & Snyder, *Precomputed Radiance Transfer* (SIGGRAPH 2002) — where SH
  lighting goes beyond probes.
