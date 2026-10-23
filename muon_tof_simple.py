"""
Simple time-of-flight Monte Carlo for the two-paddle apparatus (apparatus_sketch.png).

Only two ingredients:
  1. muon speed   - momentum sampled from the measured sea-level mu+ and mu- spectra
                    (CAPRICE94, Kremer et al., PRL 83, 4241 (1999), Table I), beta = p / sqrt(p^2 + m^2)
  2. geometry     - cos^2(theta) zenith distribution (Rossi 1948), uniform hit point on paddle A,
                    keep the muon if the straight track also crosses paddle B; path L = h / cos(theta)

No energy loss, no decay, no detector smearing:  TOF = L / (beta c).

Because speed and angle are independent, <TOF> = <L> <1/beta> / c, so the straight-line fit of
<TOF> vs <L> returns v = c / <1/beta> (the harmonic mean speed of the spectrum).

Run: python3 muon_tof_simple.py   ->  figures/fig5_tof_simple.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
os.makedirs("figures", exist_ok=True)

app_heights = np.array([50, 100, 150, 200, 250, 300, 350])   # paddle B positions, cm (as in the notebook)
W, Lp = 20.0, 50.0           # paddle footprint, cm (from the notebook)
c = 29.9792458               # cm/ns
M_MU = 0.1056583755          # muon mass, GeV/c^2 (PDG 2024)

# CAPRICE94 (Kremer et al. 1999, Table I): mean momentum (GeV/c), mu+ flux, mu- flux [(GeV/c m^2 sr s)^-1]
CAPRICE94 = np.array([
    [0.25, 14.0, 11.0], [0.35, 16.8, 13.6], [0.47, 17.2, 14.4], [0.62, 16.6, 13.5],
    [0.78, 15.6, 13.3], [0.92, 14.8, 12.1], [1.1, 13.0, 11.0], [1.3, 12.0, 10.1],
    [1.5, 10.2, 8.7], [1.84, 9.1, 7.3], [2.49, 6.6, 5.20], [3.49, 4.12, 3.38],
    [4.78, 2.53, 1.98], [6.21, 1.61, 1.25], [8.37, 0.90, 0.69], [12.42, 0.389, 0.309],
    [18.85, 0.138, 0.108], [26.68, 0.063, 0.046], [36.69, 0.028, 0.019], [51.47, 9.9e-3, 7.1e-3],
    [72.08, 3.6e-3, 3.0e-3], [100.96, 1.4e-3, 1.2e-3],
])

# ----------------------------------------------------------------------------------------------
# 1. Speed: sample momentum from the mu+ + mu- spectrum (log-log interpolated, 0.2-120 GeV/c)
# ----------------------------------------------------------------------------------------------
p_grid = np.geomspace(0.2, 120.0, 4000)
flux = np.exp(np.interp(np.log(p_grid), np.log(CAPRICE94[:, 0]),
                        np.log(CAPRICE94[:, 1] + CAPRICE94[:, 2])))
cdf = np.concatenate([[0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(p_grid))])

N = 2_000_000
p = np.interp(rng.random(N) * cdf[-1], cdf, p_grid)
beta = p / np.sqrt(p**2 + M_MU**2)

# ----------------------------------------------------------------------------------------------
# 2. Geometry: flux through a horizontal paddle ~ cos^2(theta) cos(theta) dOmega -> cos(theta) = U^(1/4)
# ----------------------------------------------------------------------------------------------
cos_t = rng.random(N) ** 0.25
tan_t = np.sqrt(1 - cos_t**2) / cos_t
phi = rng.uniform(0, 2 * np.pi, N)
xA = rng.uniform(-W / 2, W / 2, N)
yA = rng.uniform(-Lp / 2, Lp / 2, N)

rows, tofs = [], {}
for h in app_heights:
    xB = xA + h * tan_t * np.cos(phi)
    yB = yA + h * tan_t * np.sin(phi)
    hit = (np.abs(xB) < W / 2) & (np.abs(yB) < Lp / 2)
    L = h / cos_t[hit]
    tof = L / (beta[hit] * c)
    tofs[h] = tof
    rows.append((h, hit.sum(), L.mean(), tof.mean(), tof.std(), tof.std() / np.sqrt(hit.sum())))

h_, n_, L_, t_, sd_, err_ = map(np.array, zip(*rows))
(slope, t0), cov = np.polyfit(L_, t_, 1, w=1 / err_, cov="unscaled")
v_fit, v_err = 1 / slope, np.sqrt(cov[0, 0]) / slope**2
v_harm = c / np.mean(1 / beta)                   # expected answer: harmonic mean speed of the spectrum

print(f"{'h (cm)':>7} {'accepted':>9} {'<L> (cm)':>9} {'<TOF> (ns)':>11} {'sd (ns)':>8}")
for r in rows:
    print(f"{r[0]:7.0f} {r[1]:9d} {r[2]:9.1f} {r[3]:11.3f} {r[4]:8.3f}")
print(f"fit: v = {v_fit:.3f} +- {v_err:.3f} cm/ns = {v_fit/c:.4f} c,  intercept {t0*1e3:.1f} ps")
print(f"expected c/<1/beta> = {v_harm:.3f} cm/ns = {v_harm/c:.4f} c;  mean beta = {beta.mean():.4f}")

# ----------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------
INK, INK2, GRID, C1 = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "mathtext.fontset": "dejavuserif",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200,
})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(app_heights)))
bins = np.arange(0, 14, 0.05)
for h, col in zip(app_heights, blues):
    a1.hist(tofs[h], bins, density=True, histtype="step", color=col, lw=1.4, label=f"{h} cm")
a1.set(yscale="log", ylim=(1e-3, 3e3), xlabel=r"TOF $= L/(\beta c)$ (ns)", ylabel="probability density",
       title="TOF spectra: spectrum speed + geometry only")
a1.legend(title="h", fontsize=9, title_fontsize=9, ncol=4, loc="upper center")

a2.errorbar(L_, t_, err_, fmt="o", ms=6, color=C1, label="MC (mean of each spectrum)")
Lf = np.linspace(0, 380, 50)
a2.plot(Lf, t0 + slope * Lf, color=INK2, lw=1.2, ls="--",
        label=f"fit: v = {v_fit:.3f} ± {v_err:.3f} cm/ns = {v_fit/c:.4f} c")
a2.set(xlabel=r"mean slant path $\langle L\rangle$ (cm)", ylabel=r"$\langle$TOF$\rangle$ (ns)",
       title="Speed of cosmic-ray muons")
a2.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig("figures/fig5_tof_simple.png")
