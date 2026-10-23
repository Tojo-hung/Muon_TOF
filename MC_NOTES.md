# Muon TOF Monte Carlo: method, preliminary results, sources

Script: `muon_tof_mc.py` (`python3 muon_tof_mc.py`, ~30 s, needs numpy/scipy/matplotlib).
Figures go to `figures/`, and the printed numbers are saved in `figures/mc_output.txt`.

## 1. What the simulation does

For each muon:

1. **Charge and momentum.** These are sampled from the measured sea-level μ⁺ and μ⁻ spectra (CAPRICE94, 0.2–120 GeV/c).
   The integrated charge ratio comes out as μ⁺/μ⁻ = 1.24.
2. **Direction and hit point.** The zenith angle follows I(θ) ∝ cos²θ. The flux through a horizontal paddle carries an extra cos θ, so
   cos θ = U^{1/4}. The azimuth is uniform, and the hit point is uniform on paddle A (20 × 50 cm).
3. **Acceptance.** The straight track is projected to paddle B at h = 50 … 350 cm and kept if it lands on B.
4. **Transport.** The muon is stepped through ½ paddle → (h − t) of air → ½ paddle using the Bethe formula for −dE/dx.
   At each step it accumulates
   * lab time  dt = ds / (βc), which gives the TOF, and
   * proper time  dτ = ds / (βγc), which gives the survival probability exp(−τ/τ₀).
5. **Timing.** The start and stop times are smeared by the light transit to the PMT at the end of each paddle (v = c/1.58)
   and by Gaussian PMT/CFD jitter.

## 2. Preliminary results

| h (cm) | ⟨L⟩ (cm) | rate (/min) | counts / 30 min | ⟨TOF⟩ (ns) | ΔE in air+plastic (MeV) | decayed in flight (ppm) |
|---|---|---|---|---|---|---|
| 50  | 52.9  | 132.9 | 3985 | 1.78  | 2.8 | 70  |
| 100 | 102.1 | 43.7  | 1312 | 3.43  | 2.8 | 135 |
| 150 | 151.5 | 20.8  | 623  | 5.10  | 2.9 | 201 |
| 200 | 201.2 | 12.0  | 360  | 6.76  | 3.1 | 268 |
| 250 | 250.9 | 7.8   | 234  | 8.43  | 3.2 | 332 |
| 300 | 300.8 | 5.5   | 164  | 10.11 | 3.3 | 399 |
| 350 | 350.7 | 4.1   | 122  | 11.78 | 3.4 | 463 |

* **Speed (Fig. 1).** A fit of ⟨TOF⟩ against ⟨L⟩ gives v = 0.993 c. This is the measurement the apparatus is built for.
* **Counts vs distance (Fig. 2).** The count rate falls by about 30× from 50 cm to 350 cm. All of that drop is solid angle
  (geometric acceptance, matching the G factor in `Sketch&Tables.ipynb`). Decay removes only 4.6 × 10⁻⁴ of the muons at 350 cm.
* **The exponential (Fig. 3).** The surviving fraction is
  `N(L)/N₀ = ⟨exp(−L / (βγ c τ))⟩`. For L ≪ βγcτ this is a straight line with slope ⟨1/βγ⟩/(cτ).
  Fitting the MC gives back τ = 2.19 µs for μ⁺ and 2.21 µs for μ⁻ (input 2.197 µs).
  The two charges overlap because muons decaying **in flight** have the same lifetime (CPT). Their curves differ only
  through their slightly different momentum spectra. The right panel extends the same muons to kilometres of air.
  The dashed curves are decay only. The solid curves add the air stopping power, which ranges out the
  low-momentum muons (about 2 MeV per g/cm², i.e. about 240 MeV per km of air).
* **Feasibility.** Telling S(50 cm) and S(350 cm) apart at 3σ would need about 6 × 10⁷ counts at 350 cm, which is
  **about 28 years** of running. On top of that, the acceptance would have to be known to 10⁻⁴, and TOF measures only β
  (≈ 1), not γ. The TOF setup therefore measures v, not τ. The lifetimes, and the μ⁺/μ⁻ difference, have to come from
  **stopped** muons (Fig. 4), using the same CFD → TAC → MCA chain (MIT guide, part 2).

## 3. μ⁺ and μ⁻ decay rates (stopped muons, Fig. 4 left)

| | lifetime | decay rate Γ = 1/τ |
|---|---|---|
| μ⁺ (free decay) | τ₊ = 2.196 981 µs | 0.4552 µs⁻¹ |
| μ⁻ bound in carbon (decay + nuclear capture μ⁻p → nν) | τ₋ = 2.0263 µs | 0.4935 µs⁻¹ |
| capture rate Λ_c = Γ₋ − Γ₊ | | 0.0383 µs⁻¹ |

A stopped μ⁻ produces a decay electron only with probability Γ₀/Γ₋ = τ₋/τ₀ = 0.922. With n₊ and n₋ stopped muons, the
observed decay-time spectrum is therefore a **sum** of two exponentials:

```
dN/dt = (n₊/τ₀) e^{−t/τ₊} + (n₋/τ₀) e^{−t/τ₋} + B          (B = flat accidental background)
```

A single-exponential fit to the simulated mixture (n₊/n₋ = 1.23) gives τ_eff = 2.130 ± 0.006 µs.
Hu et al. (2016) measured 2.098 µs in plastic, which is the same effect. To separate Γ₊ and Γ₋, fit the two-exponential
form with the charge ratio fixed (as the script does), or fit τ₋ with τ₊ fixed to the PDG value.

## 4. Convolution of two exponentials (Fig. 4 right)

**When it applies.** The convolution describes a measured time that is the **sum** of two independent exponential
waiting times, t = t₁ + t₂:

```
f(t) = Γ₁ e^{−Γ₁ t},   g(t) = Γ₂ e^{−Γ₂ t}

(f * g)(t) = ∫₀ᵗ f(t′) g(t − t′) dt′
           = Γ₁Γ₂ e^{−Γ₂ t} ∫₀ᵗ e^{−(Γ₁−Γ₂) t′} dt′
           = Γ₁Γ₂/(Γ₂ − Γ₁) · (e^{−Γ₁ t} − e^{−Γ₂ t})          (Γ₁ ≠ Γ₂)
           → Γ² t e^{−Γ t}                                     (Γ₁ = Γ₂)
```

The Laplace-transform check gives the same result: ℒ{f * g} = Γ₁/(s+Γ₁) · Γ₂/(s+Γ₂). Partial fractions bring back the
line above.

The result starts at zero, peaks at t* = ln(Γ₂/Γ₁)/(Γ₂ − Γ₁), and its mean is τ₁ + τ₂. Fig. 4 (right) checks the formula
against a direct MC of t₁ + t₂ with Γ₁ = Γ₊ and Γ₂ = Γ₋.

**Which one your data needs.**
* **Mixed μ⁺/μ⁻ sample.** The spectrum is a **sum (mixture)** of the two exponentials, as in §3, not a convolution.
  Each muon has only one lifetime.
* **Detector timing resolution.** A convolution does apply here. The measured spectrum is exponential ⊗ Gaussian
  (an exponentially modified Gaussian):
  `(Γ/2) exp(Γ(Γσ²/2 − t)) · erfc((Γσ² − t)/(√2 σ))`.
  At σ ≈ 1 ns versus τ ≈ 2 µs this matters only in the first few bins.
* **Sequential processes.** A convolution also applies to chains such as π → μ → e (τ_π = 26 ns, then τ_μ). The Bateman
  form above is the distribution of the total time.

## 5. Assumptions to confirm

| Quantity | Value used | Status |
|---|---|---|
| paddle footprint | 20 × 50 cm | from notebook; confirm |
| paddle thickness | 1.0 cm | **assumed** |
| PMT/CFD jitter per channel | 0.25 ns (σ) | **assumed** |
| light speed in paddle | c/1.58 | upper bound (ignores reflections) |
| run time per position | 30 min | MIT guide suggests 10–45 min |
| momentum range | 0.2–120 GeV/c | CAPRICE94 table only, no extrapolation below 0.2 GeV/c (slightly underestimates decays) |
| angular distribution | cos²θ, independent of momentum | standard approximation |
| Bethe formula | no density-effect correction | effect is negligible for < 4 MeV total loss |
| flat accidental background (Fig. 4) | 30 counts/bin | illustrative |

## 6. Sources

* **Muon mass, free lifetime τ₀ = 2.1969811 µs.** Particle Data Group, R. L. Workman et al., *Review of Particle Physics*,
  PTEP 2022, 083C01, with the 2024 update (S. Navas et al., Phys. Rev. D 110, 030001). Also quoted in the Hu 2016 paper in this repo.
* **Stopping power (Bethe formula, K = 0.307075 MeV cm²/mol, Tmax).** PDG *Passage of particles through matter*, eq. 34.5.
* **Air and polyvinyltoluene parameters** (Z/A, I = 85.7 eV and 64.7 eV, air ρ = 1.205 mg/cm³).
  PDG *Atomic and Nuclear Properties of Materials*, https://pdg.lbl.gov/AtomicNuclearProperties/
* **EJ-200 density 1.023 g/cm³, refractive index 1.58, rise/decay 0.9/2.1 ns.** `Eljen_EJ-200_Plastic_Scintillator_Datasheet.pdf` (this repo).
* **μ⁺ and μ⁻ momentum spectra and charge ratio.** J. Kremer et al., PRL 83, 4241 (1999), Table I, CAPRICE94 columns
  (`Energy_Spectrum_Cosmic_Muons.pdf`, this repo).
* **Vertical intensity I_V = 0.83 × 10⁻² cm⁻² s⁻¹ sr⁻¹ and cos²θ.** B. Rossi, Rev. Mod. Phys. 20, 537 (1948), as used in `Sketch&Tables.ipynb`.
* **μ⁻ lifetime in carbon, 2026.3 ± 1.5 ns.** T. Suzuki, D. F. Measday, J. P. Roalsvig, *Total nuclear capture rates for
  negative muons*, Phys. Rev. C 35, 2212 (1987).
* **Stopped-muon method, charge-ratio effect on τ, background subtraction.** `MIT_3rd_Muon.pdf` and
  `Hu2016_Simple_Setup_Muon_Lifetime_and_Michel_Spectrum_MC.pdf` (this repo).
* **PMT timing jitter.** `Hamamatsu_PMT_Handbook_v4.pdf` (time-response chapter; transit-time spread is a few hundred ps).
