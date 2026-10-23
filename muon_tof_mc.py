"""
Monte Carlo of the two-paddle cosmic-ray muon time-of-flight apparatus (apparatus_sketch.png).

Paddle A is fixed; paddle B sits at h = 50 ... 350 cm below it. For every muon we:
  1. sample charge and momentum from the measured sea-level spectra (CAPRICE94, Kremer et al. 1999),
  2. sample a zenith angle from I(theta) ~ cos^2(theta) (Rossi 1948) and a hit point on paddle A,
  3. keep it only if the straight track also crosses paddle B,
  4. step it through plastic and air with the Bethe stopping power (PDG), tracking the lab time
     (-> TOF) and the proper time (-> decay probability, tau_0 = 2.197 us),
  5. smear the start/stop times with light transit in the paddle and PMT/CFD jitter.

Outputs (figures/):
  fig1_tof.png                 TOF spectra at each height and <TOF> vs path length -> muon speed
  fig2_counts_vs_distance.png  coincidence counts vs path length for a realistic run (geometry dominated)
  fig3_survival_vs_distance.png  the exponential decay-in-flight curve for mu+ and mu-, and the lifetime fit
  fig4_stopped_and_convolution.png  stopped-muon decay curve (sum of mu+ and mu- exponentials) and the
                                    convolution of two exponentials

Run: python3 muon_tof_mc.py      (about a minute; needs numpy, scipy, matplotlib)
Sources for every number are in MC_NOTES.md.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

rng = np.random.default_rng(20261023)
os.makedirs("figures", exist_ok=True)

# ----------------------------------------------------------------------------------------------
# Constants  (units: cm, ns, MeV)
# ----------------------------------------------------------------------------------------------
app_heights = np.array([50, 100, 150, 200, 250, 300, 350])   # paddle B positions, cm (same as notebook)
W, Lp = 20.0, 50.0           # paddle footprint, cm (from the notebook, confirm against the real paddles)
T_PAD = 1.0                  # paddle thickness, cm (ASSUMED)
I_V = 0.83e-2                # vertical muon intensity, cm^-2 s^-1 sr^-1 (Rossi 1948)
c = 29.9792458               # cm/ns

M_MU = 105.6583755           # muon mass, MeV (PDG 2024)
M_E = 0.51099895             # electron mass, MeV (PDG 2024)
TAU0 = 2196.9811             # free muon lifetime, ns (PDG 2024); same for mu+ and mu- (CPT)
TAU_MINUS_C = 2026.3         # mu- lifetime bound in carbon, ns (Suzuki, Measday & Roalsvig 1987)
K_BETHE = 0.307075           # 4 pi N_A r_e^2 m_e c^2, MeV cm^2 / mol (PDG)

# Materials: Z/A, mean excitation energy I (MeV), density (g/cm^3)  -- PDG Atomic & Nuclear Properties
AIR = dict(ZA=0.49919, I=85.7e-6, rho=1.205e-3)        # dry air, 20 C, 1 atm
PVT = dict(ZA=0.54141, I=64.7e-6, rho=1.023)           # polyvinyltoluene (EJ-200); rho from Eljen datasheet

N_SCINT = 1.58               # EJ-200 refractive index (Eljen datasheet)
V_LIGHT = c / N_SCINT        # light speed in the paddle, cm/ns (upper bound: ignores reflections)
SIGMA_JIT = 0.25             # PMT + CFD timing jitter per channel, ns (ASSUMED; Hamamatsu handbook §4.3.1(2), Table 4-3)
T_RUN = 30 * 60              # live time per paddle position, s (MIT guide suggests 10-45 min)

# Plot styling (reference palette from the dataviz skill)
C_PLUS, C_MINUS, C_SUM = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "mathtext.fontset": "dejavuserif",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.dpi": 200,
})

# ----------------------------------------------------------------------------------------------
# Sea-level muon momentum spectrum, CAPRICE94 (Kremer et al., PRL 83, 4241 (1999), Table I)
# columns: mean momentum (GeV/c), mu+ flux, mu- flux   [(GeV/c m^2 sr s)^-1]
# ----------------------------------------------------------------------------------------------
CAPRICE94 = np.array([
    [0.25, 14.0, 11.0], [0.35, 16.8, 13.6], [0.47, 17.2, 14.4], [0.62, 16.6, 13.5],
    [0.78, 15.6, 13.3], [0.92, 14.8, 12.1], [1.1, 13.0, 11.0], [1.3, 12.0, 10.1],
    [1.5, 10.2, 8.7], [1.84, 9.1, 7.3], [2.49, 6.6, 5.20], [3.49, 4.12, 3.38],
    [4.78, 2.53, 1.98], [6.21, 1.61, 1.25], [8.37, 0.90, 0.69], [12.42, 0.389, 0.309],
    [18.85, 0.138, 0.108], [26.68, 0.063, 0.046], [36.69, 0.028, 0.019], [51.47, 9.9e-3, 7.1e-3],
    [72.08, 3.6e-3, 3.0e-3], [100.96, 1.4e-3, 1.2e-3],
])
P_MIN, P_MAX = 0.2, 120.0    # GeV/c, range covered by the table (nothing is extrapolated)

p_grid = np.geomspace(P_MIN, P_MAX, 4000)
def spectrum(col):
    """log-log interpolated differential flux on p_grid and its normalised CDF."""
    f = np.exp(np.interp(np.log(p_grid), np.log(CAPRICE94[:, 0]), np.log(CAPRICE94[:, col])))
    cdf = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(p_grid))])
    return f, cdf

f_plus, cdf_plus = spectrum(1)
f_minus, cdf_minus = spectrum(2)
CHARGE_RATIO = cdf_plus[-1] / cdf_minus[-1]           # integrated mu+/mu- over 0.2-120 GeV/c
FRAC_PLUS = CHARGE_RATIO / (1 + CHARGE_RATIO)

def sample_momentum(n, cdf):
    """Inverse-CDF sampling of momentum (MeV/c)."""
    return 1e3 * np.interp(rng.random(n) * cdf[-1], cdf, p_grid)

# ----------------------------------------------------------------------------------------------
# Kinematics and stopping power
# ----------------------------------------------------------------------------------------------
def bethe(T, mat):
    """Mean energy loss -dE/dx (MeV/cm) of a muon with kinetic energy T (MeV), PDG eq. 34.5, no density effect."""
    gamma = 1 + T / M_MU
    bg2 = gamma**2 - 1
    b2 = bg2 / gamma**2
    Tmax = 2 * M_E * bg2 / (1 + 2 * gamma * M_E / M_MU + (M_E / M_MU)**2)
    return K_BETHE * mat["ZA"] * mat["rho"] / b2 * (0.5 * np.log(2 * M_E * bg2 * Tmax / mat["I"]**2) - b2)

def propagate(T, length, mat, nsteps=10):
    """Step muons (kinetic energy T, MeV) through `length` cm of `mat` (arrays, one entry per muon).
    Returns new T, lab time (ns), proper time (ns), and a mask of muons that ranged out."""
    T = T.copy()
    t_lab, t_prop = np.zeros_like(T), np.zeros_like(T)
    stopped = np.zeros(T.shape, bool)
    ds = length / nsteps
    for _ in range(nsteps):
        Tm = np.maximum(T - 0.5 * bethe(np.maximum(T, 1e-3), mat) * ds, 1e-3)   # midpoint energy
        gamma = 1 + Tm / M_MU
        bg = np.sqrt(gamma**2 - 1)
        t_lab += ds * gamma / (bg * c)          # ds / (beta c)
        t_prop += ds / (bg * c)                 # ds / (beta gamma c)
        T = T - bethe(Tm, mat) * ds
        stopped |= T <= 0
        T = np.maximum(T, 1e-3)
    return T, t_lab, t_prop, stopped

# ----------------------------------------------------------------------------------------------
# 1. Generate muons crossing paddle A (mid-plane), then follow them to paddle B at each height
# ----------------------------------------------------------------------------------------------
N_GEN = 3_000_000
is_plus = rng.random(N_GEN) < FRAC_PLUS
p = np.where(is_plus, sample_momentum(N_GEN, cdf_plus), sample_momentum(N_GEN, cdf_minus))
T0 = np.sqrt(p**2 + M_MU**2) - M_MU
# flux through a horizontal plane: dN ~ cos^2(theta) * cos(theta) dOmega  ->  cos(theta) = U^(1/4)
cos_t = rng.random(N_GEN) ** 0.25
sin_t = np.sqrt(1 - cos_t**2)
phi = rng.uniform(0, 2 * np.pi, N_GEN)
xA = rng.uniform(-W / 2, W / 2, N_GEN)
yA = rng.uniform(-Lp / 2, Lp / 2, N_GEN)
R_SINGLE = np.pi / 2 * I_V * W * Lp            # muons/s crossing paddle A (as in the notebook)

# first half of paddle A happens before the start signal
T_startA, _, _, stopA = propagate(T0, (T_PAD / 2) / cos_t, PVT, nsteps=2)

rows, tof_samples = [], {}
for h in app_heights:
    xB = xA + h * sin_t / cos_t * np.cos(phi)
    yB = yA + h * sin_t / cos_t * np.sin(phi)
    hit = (np.abs(xB) < W / 2) & (np.abs(yB) < Lp / 2) & ~stopA
    ct, Ti = cos_t[hit], T_startA[hit]
    # start (mid-plane A) -> plastic t/2 -> air (h - t) -> plastic t/2 -> stop (mid-plane B)
    T1, tl1, tp1, s1 = propagate(Ti, (T_PAD / 2) / ct, PVT, 2)
    T2, tl2, tp2, s2 = propagate(T1, (h - T_PAD) / ct, AIR, 10)
    T3, tl3, tp3, s3 = propagate(T2, (T_PAD / 2) / ct, PVT, 2)
    reached = ~(s1 | s2 | s3)
    t_lab, t_prop = tl1 + tl2 + tl3, tp1 + tp2 + tp3
    survive_p = np.exp(-t_prop / TAU0) * reached        # per-muon survival probability (no MC noise)
    L = h / ct                                          # slant path between mid-planes
    gb = np.sqrt((1 + Ti / M_MU)**2 - 1)                # beta*gamma at paddle A
    # measured TOF: true TOF + light transit to the PMT at the +y end of each paddle + jitter
    t_meas = (t_lab + ((Lp / 2 - yB[hit]) - (Lp / 2 - yA[hit])) / V_LIGHT
              + rng.normal(0, SIGMA_JIT * np.sqrt(2), hit.sum()))
    tof_samples[h] = t_meas[rng.random(hit.sum()) < survive_p]
    acc = hit.mean()
    for q, m in (("+", is_plus[hit]), ("-", ~is_plus[hit])):
        rows.append(dict(h=h, q=q, L=L[m].mean(), S=survive_p[m].mean(),
                         inv_bg=(1 / gb[m]).mean(), n_acc=m.sum()))
    rows.append(dict(h=h, q="all", L=L.mean(), S=survive_p.mean(), inv_bg=(1 / gb).mean(),
                     acc=acc, R_geom=R_SINGLE * acc, R=R_SINGLE * acc * survive_p.mean(),
                     tof=tof_samples[h].mean(), tof_err=tof_samples[h].std() / np.sqrt(len(tof_samples[h])),
                     tof_sd=tof_samples[h].std(), dE=(Ti - T3)[reached].mean()))

def pick(q, key):
    return np.array([r[key] for r in rows if r["q"] == q])

L_all, S_all = pick("all", "L"), pick("all", "S")
R_geom, R_true = pick("all", "R_geom"), pick("all", "R")

print(f"mu+/mu- charge ratio (CAPRICE94, 0.2-120 GeV/c): {CHARGE_RATIO:.3f}")
print(f"single-paddle rate {R_SINGLE:.2f} /s; mean muon momentum {p.mean()/1e3:.2f} GeV/c")
print(f"{'h':>5} {'<L>':>7} {'R(/min)':>8} {'N/30min':>8} {'<TOF>':>7} {'sd':>5} {'dE':>5} {'1-S (ppm)':>10}")
for r in (r for r in rows if r["q"] == "all"):
    print(f"{r['h']:5.0f} {r['L']:7.1f} {r['R']*60:8.2f} {r['R']*T_RUN:8.0f} {r['tof']:7.3f} "
          f"{r['tof_sd']:5.2f} {r['dE']:5.2f} {(1-r['S'])*1e6:10.1f}")

# ----------------------------------------------------------------------------------------------
# Fig 1: TOF spectra and muon speed
# ----------------------------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(app_heights)))
bins = np.arange(-4, 18, 0.2)
for h, col in zip(app_heights, blues):
    a1.hist(tof_samples[h], bins, density=True, histtype="step", color=col, lw=1.6, label=f"{h} cm")
a1.set(xlabel="measured TOF, B minus A (ns)", ylabel="probability density",
       title="Simulated TOF spectra (light transit + 0.25 ns jitter)")
a1.set_ylim(0, 0.62)
a1.legend(title="h", fontsize=9, title_fontsize=9, ncol=4, loc="upper center")

tof, tof_err = pick("all", "tof"), pick("all", "tof_err")
(slope, t0), cov = np.polyfit(L_all, tof, 1, w=1 / tof_err, cov="unscaled")
v_fit = 1 / slope
a2.errorbar(L_all, tof, tof_err, fmt="o", ms=6, color=C_PLUS, label="MC (mean of each spectrum)")
Lf = np.linspace(0, 380, 50)
a2.plot(Lf, t0 + slope * Lf, color=INK2, lw=1.2, ls="--",
        label=f"fit: v = {v_fit:.2f} cm/ns = {v_fit/c:.3f} c")
a2.set(xlabel=r"mean slant path $\langle L\rangle$ (cm)", ylabel=r"$\langle$TOF$\rangle$ (ns)",
       title="Speed of cosmic-ray muons")
a2.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig("figures/fig1_tof.png")
print(f"TOF fit: v = {v_fit:.3f} cm/ns = {v_fit/c:.4f} c")

# ----------------------------------------------------------------------------------------------
# Fig 2: counts vs distance for a realistic run -- the falloff is geometry, not decay
# ----------------------------------------------------------------------------------------------
N_exp_geom, N_exp = R_geom * T_RUN, R_true * T_RUN
N_obs = rng.poisson(N_exp)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
a1.errorbar(L_all, N_obs, np.sqrt(N_obs), fmt="o", ms=6, color=C_PLUS, label="simulated run (Poisson)")
a1.plot(L_all, N_exp_geom, color=INK2, ls="--", lw=1.2, label="geometric acceptance only")
a1.plot(L_all, N_exp, color=C_SUM, lw=1.2, label="acceptance × decay survival (on top of dashed)")
a1.set(yscale="log", ylabel=f"coincidences per {T_RUN//60} min",
       title="Counts vs distance: the drop is solid angle, not decay")
a1.legend(fontsize=9)
a2.errorbar(L_all, N_obs / N_exp_geom, np.sqrt(N_obs) / N_exp_geom, fmt="o", ms=6, color=C_PLUS)
a2.plot(L_all, S_all, color=C_SUM, lw=1.2)
a2.axhline(1, color=INK2, lw=0.8)
a2.set(xlabel=r"mean slant path $\langle L\rangle$ (cm)", ylabel="counts / acceptance")
a2.text(0.98, 0.08, f"decay effect at 350 cm: {(1-S_all[-1])*1e2:.3f}% (green line)",
        transform=a2.transAxes, ha="right", fontsize=9, color=INK2)
fig.tight_layout()
fig.savefig("figures/fig2_counts_vs_distance.png")

# ----------------------------------------------------------------------------------------------
# Fig 3: the exponential -- survival fraction vs distance for mu+ and mu-
#   S(L) = < exp(-L / (beta gamma c tau)) >  ->  -ln S ~ L <1/(beta gamma)> / (c tau)
# ----------------------------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
fits = {}
for q, col, lab in (("+", C_PLUS, r"$\mu^+$"), ("-", C_MINUS, r"$\mu^-$")):
    Lq, Sq, ibg = pick(q, "L"), pick(q, "S"), pick(q, "inv_bg")
    k = np.sum(Lq * -np.log(Sq)) / np.sum(Lq**2)          # fit -ln S = k L through the origin
    tau_fit = ibg.mean() / (c * k)
    fits[q] = tau_fit
    a1.plot(Lq, (1 - Sq) * 1e4, "o", ms=7, color=col, label=f"{lab} MC")
    a1.plot(Lf, (1 - np.exp(-k * Lf)) * 1e4, color=col, lw=1.2,
            label=rf"{lab} fit: $\tau$ = {tau_fit/1e3:.3f} $\mu$s, $\Gamma$ = {1e3/tau_fit:.3f} $\mu$s$^{{-1}}$")
a1.set(xlabel=r"mean slant path $\langle L\rangle$ (cm)", ylabel=r"fraction decayed in flight ($\times 10^{-4}$)",
       title="In the apparatus (0–3.5 m)")
a1.legend(fontsize=8.5, loc="upper left")

# extended range, vertical muons: what the same exponential looks like over kilometres of air
N_LONG = 200_000
L_long = np.linspace(0, 8e5, 161)                     # cm (0 - 8 km)
for q, cdf, col, lab in (("+", cdf_plus, C_PLUS, r"$\mu^+$"), ("-", cdf_minus, C_MINUS, r"$\mu^-$")):
    T = np.sqrt(sample_momentum(N_LONG, cdf)**2 + M_MU**2) - M_MU
    bg0 = np.sqrt((1 + T / M_MU)**2 - 1)
    alive = np.ones(N_LONG)
    tp = np.zeros(N_LONG)
    S_loss, S_noloss = [1.0], [1.0]
    for L0, L1 in zip(L_long[:-1], L_long[1:]):
        T, _, dtp, st = propagate(T, np.full(N_LONG, L1 - L0), AIR, nsteps=4)
        tp += dtp
        alive[st] = 0
        S_loss.append(np.mean(alive * np.exp(-tp / TAU0)))
        S_noloss.append(np.mean(np.exp(-L1 / (bg0 * c * TAU0))))
    a2.plot(L_long / 1e5, S_noloss, color=col, lw=1.2, ls="--", label=f"{lab} decay only")
    a2.plot(L_long / 1e5, S_loss, color=col, lw=2, label=f"{lab} decay + air stopping power")
a2.axvspan(0, 3.5e-3, color=C_SUM, alpha=0.4)
a2.annotate("apparatus range\n(0–3.5 m)", xy=(0.01, 0.97), xytext=(1.2, 0.5), fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
a2.set(yscale="log", ylim=(0.2, 1.05), xlabel="distance travelled in air (km)",
       ylabel=r"surviving fraction $N/N_0$", title="Same muons, extended to kilometres (vertical)")
a2.legend(fontsize=8.5, loc="lower left")
fig.tight_layout()
fig.savefig("figures/fig3_survival_vs_distance.png")
print(f"lifetime from decay-in-flight fit: mu+ {fits['+']:.0f} ns, mu- {fits['-']:.0f} ns (input {TAU0} ns)")

# statistics needed to see the decay between the top and bottom positions
d = S_all[0] - S_all[-1]
n_needed = 9 / d**2 * (1 + R_true[-1] / R_true[0])    # counts at 350 cm for a 3-sigma difference
print(f"decay difference 50 vs 350 cm: {d:.2e};  3-sigma needs {n_needed:.2e} counts at 350 cm "
      f"= {n_needed / R_true[-1] / 3.156e7:.0f} years of running")

# ----------------------------------------------------------------------------------------------
# Fig 4: stopped muons (sum of two exponentials) and the convolution of two exponentials
# ----------------------------------------------------------------------------------------------
# Stopped muons are the low-momentum ones; use the charge ratio of the lowest CAPRICE94 bins.
R_LOW = CAPRICE94[:3, 1].sum() / CAPRICE94[:3, 2].sum()
N_STOP = 200_000
n_plus = rng.binomial(N_STOP, R_LOW / (1 + R_LOW))
n_minus = N_STOP - n_plus
# a captured mu- gives no decay electron: only a fraction tau_C/tau_0 of stopped mu- produce a STOP pulse
n_minus_e = rng.binomial(n_minus, TAU_MINUS_C / TAU0)
t_plus = rng.exponential(TAU0, n_plus) / 1e3                     # us
t_minus = rng.exponential(TAU_MINUS_C, n_minus_e) / 1e3
bg_per_bin = 30                                                   # flat accidental background, counts/bin
edges = np.linspace(0, 20, 101)
ctr, bw = 0.5 * (edges[1:] + edges[:-1]), edges[1] - edges[0]
h_plus, _ = np.histogram(t_plus, edges)
h_minus, _ = np.histogram(t_minus, edges)
h_tot = h_plus + h_minus + rng.poisson(bg_per_bin, len(ctr))

def one_exp(t, A, tau, B):
    return A * np.exp(-t / tau) + B

def two_exp(t, A, B):      # lifetimes and charge ratio fixed; dN-/dt = (n-/tau0) e^{-t/tau-}, so amplitudes go as n+ : n-
    return A * (R_LOW * np.exp(-t / (TAU0 / 1e3)) + np.exp(-t / (TAU_MINUS_C / 1e3))) + B

popt1, pcov1 = curve_fit(one_exp, ctr, h_tot, p0=(h_tot[0], 2.2, bg_per_bin), sigma=np.sqrt(h_tot + 1))
popt2, _ = curve_fit(two_exp, ctr, h_tot, p0=(h_tot[0] / 2, bg_per_bin), sigma=np.sqrt(h_tot + 1))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
a1.errorbar(ctr, h_tot, np.sqrt(h_tot), fmt="o", ms=3, color=INK2, elinewidth=0.6, label="simulated data (mixed)")
tt = np.linspace(0, 20, 400)
a1.plot(tt, n_plus * bw / (TAU0 / 1e3) * np.exp(-tt / (TAU0 / 1e3)), color=C_PLUS, lw=1.4,
        label=rf"$\mu^+$: $\tau_+$ = {TAU0/1e3:.4f} $\mu$s (free decay)")
a1.plot(tt, n_minus_e * bw / (TAU_MINUS_C / 1e3) * np.exp(-tt / (TAU_MINUS_C / 1e3)), color=C_MINUS, lw=1.4,
        label=rf"$\mu^-$: $\tau_-$ = {TAU_MINUS_C/1e3:.4f} $\mu$s (capture in C)")
a1.plot(tt, two_exp(tt, *popt2), color=C_SUM, lw=1.4, label="sum of both + background")
a1.set(yscale="log", ylim=(10, None), xlabel=r"decay time $t$ ($\mu$s)", ylabel=f"counts / {bw*1e3:.0f} ns",
       title="Stopped muons: the two exponentials add")
a1.text(0.97, 0.62, rf"single-exp fit: $\tau_{{eff}}$ = {popt1[1]:.3f} ± {np.sqrt(pcov1[1,1]):.3f} $\mu$s",
        transform=a1.transAxes, ha="right", fontsize=9, color=INK2)
a1.legend(fontsize=8.5, loc="upper right")

# Convolution: t = t1 + t2 with t1 ~ Exp(l1), t2 ~ Exp(l2)  ->  pdf = l1 l2/(l2-l1) (e^{-l1 t} - e^{-l2 t})
l1, l2 = 1e3 / TAU0, 1e3 / TAU_MINUS_C                           # 1/us
t_conv = rng.exponential(1 / l1, N_STOP) + rng.exponential(1 / l2, N_STOP)
a2.hist(t_conv, edges, density=True, histtype="stepfilled", color=C_SUM, alpha=0.35, label="MC: $t_1 + t_2$")
a2.plot(tt, l1 * l2 / (l2 - l1) * (np.exp(-l1 * tt) - np.exp(-l2 * tt)), color=C_SUM, lw=2,
        label=r"convolution $\frac{\Gamma_+\Gamma_-}{\Gamma_- - \Gamma_+}(e^{-\Gamma_+ t} - e^{-\Gamma_- t})$")
mix = R_LOW / (1 + R_LOW)
a2.plot(tt, mix * l1 * np.exp(-l1 * tt) + (1 - mix) * l2 * np.exp(-l2 * tt), color=INK2, lw=1.2, ls="--",
        label=r"sum (mixture) $f_+\Gamma_+e^{-\Gamma_+ t} + f_-\Gamma_-e^{-\Gamma_- t}$")
a2.set(xlabel=r"$t$ ($\mu$s)", ylabel="probability density",
       title="Convolution vs sum of two exponentials")
a2.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig("figures/fig4_stopped_and_convolution.png")
print(f"stopped muons: charge ratio {R_LOW:.3f}; single-exp tau_eff = {popt1[1]:.3f} +- {np.sqrt(pcov1[1,1]):.3f} us")
print(f"decay rates: Gamma+ = {1e3/TAU0:.4f} /us, Gamma-(C) = {1e3/TAU_MINUS_C:.4f} /us, "
      f"capture rate = {1e3/TAU_MINUS_C - 1e3/TAU0:.4f} /us")
