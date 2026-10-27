import math
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import matrix_ml_engine

# ------------------------------------------------------------------------------
# MULTI-MATERIAL LITERATURE PRIORS
# ------------------------------------------------------------------------------
MATERIAL_PRIORS = {
    "Generic": [
        [100.0, 5.0, 7.0, 200.0, 5.0, 30.0, 0.0, 45.0],
        [120.0, 4.0, 5.0, 200.0, 5.0, 30.0, 0.5, 78.0],
        [80.0,  8.0, 7.0, 300.0, 1.0, 25.0, 0.0, 22.0],
        [150.0, 3.0, 3.0, 200.0, 10.0, 30.0, 1.0, 142.0],
        [100.0, 5.0, 5.0, 100.0, 5.0, 30.0, 0.0, 18.0],
        [130.0, 4.0, 4.0, 300.0, 10.0, 35.0, 0.5, 95.0],
        [150.0, 3.0, 3.0, 250.0, 10.0, 40.0, 1.0, 168.0],
    ],
    "WO3": [
        [100.0, 5.0, 7.0, 200.0, 5.0, 30.0, 0.0, 45.0],
        [130.0, 4.0, 4.0, 300.0, 10.0, 35.0, 0.5, 95.0],
        [150.0, 3.0, 3.0, 250.0, 10.0, 40.0, 1.0, 168.0],
    ],
    "TiO2": [
        [150.0, 3.0, 5.0, 150.0, 5.0, 40.0, 0.5, 60.0],
        [200.0, 2.0, 4.0, 150.0, 10.0, 50.0, 1.0, 110.0],
    ],
    "ZnO": [
        [80.0, 6.0, 6.0, 300.0, 5.0, 20.0, 0.0, 30.0],
        [100.0, 4.0, 5.0, 250.0, 5.0, 25.0, 1.0, 85.0],
    ],
    # NEW -- Pd as its OWN standalone material track (not the WO3_Pd catalyst
    # layer -- this is for logging/optimizing a pure Pd film on its own, e.g.
    # control samples or a separate study). Pd is a metal, so this is
    # DC-sputtered in practice, same literature source already used for the
    # WO3_Pd catalyst step's own defaults (Slavcheva et al. 2014: 100W DC,
    # ~7.8cm target-substrate distance, deposition rate 3.2-4.5 nm/min).
    # CAVEAT: this system's UI still labels the first parameter "RF_Power"
    # everywhere, even though real Pd depositions use DC -- that label is
    # generic across every material track, not accurate for this one
    # specifically. Treat the number as "primary sputtering power (W)"
    # regardless of what the label says, until that's worth a UI-wide fix.
    "Pd": [
        [100.0, 5.0, 7.0, 20.0, 5.0, 30.0, 0.0, 40.0],
        [100.0, 3.75, 7.8, 10.0, 5.0, 30.0, 0.5, 90.0],
        [80.0, 5.0, 5.0, 15.0, 10.0, 25.0, 1.0, 130.0],
    ],
    # NEW -- ITO (Indium Tin Oxide), a transparent conducting oxide, added as
    # its own separate sensing-layer track (not related to WO3 or the Pd
    # catalyst). Literature-seeded from RF-sputtered, ROOM-TEMPERATURE ITO
    # studies specifically (no post-anneal), since that matches this system's
    # no-heater constraint: ~150-200W RF, 2-3.5 mTorr, thickness 200-350nm
    # gave good crystallinity/transmittance without any heat treatment.
    # PLACEHOLDER seed values -- these papers report resistivity/transmittance,
    # not this project's intensity-response metric, so calibrate against your
    # own first few ITO runs before trusting these as targets, same caveat as
    # every other literature-seeded row in this dict.
    "ITO": [
        [150.0, 3.5, 7.0, 200.0, 5.0, 30.0, 0.5, 50.0],
        [200.0, 2.0, 5.0, 350.0, 5.0, 25.0, 1.0, 95.0],
        [100.0, 3.5, 7.0, 200.0, 5.0, 30.0, 0.0, 20.0],
    ],
    # NEW — WO3 + Pd catalyst track. Rows keep the original 8-value layout
    # ([rf, pressure, distance, thickness, rotation, ar, xrd, wave]) and
    # APPEND pd_thickness (nm) as a 9th value, so any existing code reading
    # p[6]/p[7] (xrd/wave) on other material keys is completely unaffected.
    # PLACEHOLDER seed values — the literature we reviewed reports optical
    # transmittance % / resistance-ratio, not pm wavelength shift, so these
    # need calibrating against your own first few WO3_Pd runs before they're
    # trustworthy priors. The structure is correct; the numbers aren't yet.
    "WO3_Pd": [
        [100.0, 5.0, 7.0, 200.0, 5.0, 30.0, 0.0, 90.0, 8.0],
        [130.0, 4.0, 4.0, 300.0, 10.0, 35.0, 0.5, 160.0, 6.0],
        [150.0, 3.0, 3.0, 250.0, 10.0, 40.0, 1.0, 250.0, 10.0],
    ],
}

XRD_MAP = {"Monoclinic": 1.0, "Partial": 0.75, "Amorphous": 0.0}
PARAM_NAMES = ["RF Power", "Pressure", "Target Distance", "Film Thickness", "Rotation Speed", "Ar Flow", "Pd Thickness"]

# Rotation fixture range is 5-50 RPM. Suggestions are drawn from steps of 5.
ROTATION_CHOICES = [float(r) for r in range(5, 55, 5)]
PHYSICS_FEATURE_NAMES = ["Energy Density", "Dep Rate Est", "Film Thickness", "Plasma Density", "Rotation Factor", "Ar Normalized", "Catalyst Loading", "Core Diameter Effect"]

# ------------------------------------------------------------------------------
# UPGRADE 1: PHYSICS-INFORMED FEATURE TRANSFORMATION
# ------------------------------------------------------------------------------
def transform_to_physics_features(
    rf_power: float,
    pressure: float,
    distance: float,
    thickness: float = 200.0,
    rotation: float = 5.0,
    ar_flow: float = 30.0,
    pd_thickness: float = 0.0,
    sputter_time_s: float = None,
    substrate_type: str = "Si Wafer",
    core_diameter_um: float = 9.0
) -> list:
    """Transforms raw machine parameters into kinetic and thermodynamic derived features.
    pd_thickness (nm) defaults to 0.0, so every existing caller that doesn't pass it
    keeps behaving exactly as before — this is purely additive.
    core_diameter_um defaults to 9.0 (the middle of the 8-10um etch target), which
    makes diameter_effect exactly neutral (1.0) for every existing caller that
    doesn't pass a real measured diameter -- also purely additive."""
    rf_power = max(10.0, float(rf_power))
    distance = max(1.0, float(distance))
    pressure = max(0.5, float(pressure))
    rotation = max(0.0, float(rotation))
    ar_flow = max(1.0, float(ar_flow))
    pd_thickness = max(0.0, float(pd_thickness))

    # 1. Energy Density (W/cm^2 scaling)
    energy_density = rf_power / (distance ** 2)

    # 2. Estimated Deposition Rate (nm/s scaling for WO3)
    dep_rate_estimate = (rf_power * 0.85) / (distance ** 2)

    # 3. Film thickness (normalized to 200 nm). This slot used to hold
    # "energy per nm" = energy_density / dep_rate_estimate, which is the
    # constant 1/0.85 for every recipe, so the model never saw thickness at
    # all -- even though the etched-core sensor response depends strongly on it.
    thickness_norm = max(1.0, float(thickness)) / 200.0

    # 4. Plasma Density Index (Ion bombardment intensity)
    plasma_density = rf_power / (pressure * distance)

    # 5. Rotational Uniformity (Critical for 360-deg fiber coverage, neutral for wafer)
    if str(substrate_type).strip() == "Optical Fiber":
        # Time constant 15 RPM (was 3 when the range was 1-10 RPM). With the range
        # now 5-50 RPM, a constant of 3 would saturate at ~1.0 by 10 RPM and the
        # model could not tell 10 RPM from 50 RPM. PLACEHOLDER shape -- the GP
        # learns the real rotation effect from your fiber runs.
        rotation_factor = 1.0 - math.exp(-rotation / 15.0)
    else:
        rotation_factor = 0.5

    # 6. Normalized Argon Flow (Gas density / mean free path baseline)
    ar_normalized = ar_flow / 30.0

    # 7. Catalyst Loading (normalized Pd thickness, 0 when no catalyst present)
    pd_normalized = pd_thickness / 30.0

    # 8. Core Diameter Effect (NEW) -- a thinner fiber core means a larger
    # fraction of the guided light's evanescent field extends into the
    # coating, i.e. stronger optical/gasochromic interaction. Normalized
    # against 9.0um (the middle of the 8-10um etch target) so this feature
    # is exactly neutral (1.0) for Si Wafer runs and any caller that doesn't
    # pass a real measured diameter. PLACEHOLDER functional form (simple
    # inverse ratio) -- the GP learns the real relationship from your actual
    # fiber data over time; this just gives it a physically-motivated
    # starting direction instead of no signal at all.
    core_diameter_um = max(1.0, float(core_diameter_um))
    diameter_effect = 9.0 / core_diameter_um

    return [
        float(energy_density),
        float(dep_rate_estimate),
        float(thickness_norm),
        float(plasma_density),
        float(rotation_factor),
        float(ar_normalized),
        float(pd_normalized),
        float(diameter_effect)
    ]

# ------------------------------------------------------------------------------
# UPGRADE 2: PHYSICS-INFORMED PRIOR MEAN FUNCTION
# ------------------------------------------------------------------------------
def physics_prior_mean(X_physics: np.ndarray) -> np.ndarray:
    """
    Sigmoid prior: Crystallinity increases with energy density up to saturation.
    Ranges from 0.25 (amorphous baseline) to 0.75 (likely monoclinic).
    """
    if X_physics.ndim == 1:
        X_physics = X_physics.reshape(1, -1)
    
    energy_density = X_physics[:, 0]
    threshold = 5.0
    steepness = 0.8

    xrd_prior = 0.25 + 0.5 * (1.0 / (1.0 + np.exp(-steepness * (energy_density - threshold))))
    return xrd_prior

# ------------------------------------------------------------------------------
# NEW: PD CATALYST PRIOR (peaked, not monotonic)
# ------------------------------------------------------------------------------
def pd_catalyst_prior(X_physics: np.ndarray) -> np.ndarray:
    """
    Literature-informed prior for the Pd-catalyzed sensor-response objective.
    Unlike the XRD prior (monotonic sigmoid), catalytic enhancement from a Pd
    layer PEAKS at a moderate thickness: too thin gives insufficient catalytic
    coverage for H2 dissociation, too thick blocks the optical path into the
    WO3 film underneath. Returns an enhancement factor (roughly 0-1).

    PLACEHOLDER optimum (~8nm) and width (~6nm) — calibrate against your own
    early WO3_Pd runs, or against pm-scale literature data, once available.
    Only used when target_material == "WO3_Pd"; has zero effect otherwise.
    """
    if X_physics.ndim == 1:
        X_physics = X_physics.reshape(1, -1)
    pd_norm = X_physics[:, 6]            # normalized pd_thickness (feature 7, /30nm)
    pd_nm = pd_norm * 30.0
    optimum_nm = 8.0
    # Zero-preserving bump: exactly 0 at pd_nm=0 (every existing zero-Pd
    # experiment is completely unaffected), peaks at optimum_nm, decays beyond it.
    enhancement = (pd_nm / optimum_nm) * np.exp(1.0 - (pd_nm / optimum_nm))
    return np.clip(enhancement, 0.0, None)

def build_physics_kernel() -> ConstantKernel:
    """Constructs an RBF Kernel bounded directly to physical length-scales."""
    kernel = ConstantKernel(1.0, (0.01, 100.0)) * RBF(
        length_scale=[
            2.0,   # Energy Density
            2.0,   # Deposition Rate
            1.0,   # Film Thickness (normalized)
            1.5,   # Plasma Density
            0.5,   # Rotation Factor
            2.0,   # Ar Normalized
            1.0,   # Catalyst Loading
            1.0    # Core Diameter Effect (NEW)
        ],
        length_scale_bounds=(0.1, 100.0)
    ) + WhiteKernel(
        noise_level=0.05,
        noise_level_bounds=(1e-5, 0.5)
    )
    return kernel

def calculate_quality_score(xrd_phase, wavelength_shift_pm, h2_response_s, grain_size_nm, all_experiments):
    xrd_score = XRD_MAP[normalize_xrd_phase(xrd_phase)]

    def normalize(val, key, default=0.5, invert=False):
        if val is None:
            return default
        vals = [float(e[key]) for e in all_experiments if e.get(key) is not None]
        if not vals or max(vals) == min(vals):
            return 0.5
        norm = (float(val) - min(vals)) / (max(vals) - min(vals))
        return (1.0 - norm) if invert else norm

    wave_norm = normalize(wavelength_shift_pm, "wavelength_shift_pm", default=0.5)
    h2_norm = normalize(h2_response_s, "h2_response_time_s", default=0.5, invert=True)
    grain_norm = normalize(grain_size_nm, "grain_size_nm", default=0.5)

    quality = (xrd_score * 40.0) + (wave_norm * 30.0) + (h2_norm * 20.0) + (grain_norm * 10.0)
    return round(float(np.clip(quality, 0.0, 100.0)), 1)

def format_candidate(candidate_array, mean_xrd, mean_wave, min_w, denom, core_diameter_um=9.0):
    # candidate_array is now 7-wide (added Pd Thickness); unpack defensively so
    # this still works if ever called with an old 6-wide array.
    if len(candidate_array) >= 7:
        s_rf, s_press, s_dist, s_thick, s_rot, s_ar, s_pd = candidate_array[:7]
    else:
        s_rf, s_press, s_dist, s_thick, s_rot, s_ar = candidate_array[:6]
        s_pd = 0.0
    pred_xrd = float(mean_xrd)
    pred_wave_pm = round(float(min_w + float(mean_wave) * denom), 1)

    if pred_xrd >= 0.8:
        phase = "Monoclinic"
    elif pred_xrd >= 0.4:
        phase = "Partial"
    else:
        phase = "Amorphous"
    
    return {
        "rf_power": round(float(s_rf), 1),
        "working_pressure": round(float(s_press), 1),
        "target_distance": round(float(s_dist), 1),
        "film_thickness": round(float(s_thick), 1),
        "rotation_speed": float(s_rot),
        "ar_flow": round(float(s_ar), 1),
        "pd_thickness": round(float(s_pd), 1),
        "core_diameter_um": round(float(core_diameter_um), 2),
        "expected_phase": phase,
        "expected_shift": pred_wave_pm
    }

# ------------------------------------------------------------------------------
# SHARED TRAINING-SET PIPELINE
# ------------------------------------------------------------------------------
# Every model in this file (suggestion engine, sandbox, phase map, validation)
# used to rebuild its own training set with slightly different rules. They now
# all go through build_training_set(), so the sandbox and the suggestion card
# can never disagree about what the data says.

# Literature rows are placeholders in different units (pm shift, not uW), so
# they get extra per-point noise that grows as real runs accumulate: they
# steer the first few suggestions and then fade out instead of competing with
# your measurements forever.
PRIOR_BASE_NOISE = 0.02
PRIOR_NOISE_PER_RUN = 0.03
PRIOR_MAX_NOISE = 1.0
REAL_RUN_NOISE = 1e-6          # the fitted WhiteKernel carries real measurement noise

# Once a track has this many real intensity measurements, the intensity model
# uses only real runs (the literature rows are in other units, see above).
MIN_REAL_INTENSITY_RUNS = 3


def normalize_xrd_phase(phase) -> str:
    """'MONOCLINIC', ' monoclinic ' and 'Mono' all map to 'Monoclinic'.
    Free-text phases used to fall through XRD_MAP and silently score as amorphous."""
    p = str(phase or "").strip().lower()
    if p.startswith("mono"):
        return "Monoclinic"
    if p.startswith("part") or p.startswith("mixed"):
        return "Partial"
    return "Amorphous"


def _measured_intensity(exp):
    for key in ("light_intensity_uw", "wavelength_shift_pm"):
        v = exp.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _exp_raw_params(exp) -> dict:
    return {
        "rf_power": float(exp.get("rf_power") or exp.get("rf_power_w") or 120.0),
        "pressure": float(exp.get("working_pressure") or exp.get("working_pressure_mtorr") or 8.0),
        "distance": float(exp.get("target_distance") or exp.get("target_substrate_distance_cm") or 7.0),
        "thickness": float(exp.get("film_thickness") or exp.get("film_thickness_nm") or 200.0),
        "rotation": float(exp.get("rotation_speed") or exp.get("rotation_speed_rpm") or 5.0),
        "ar_flow": float(exp.get("ar_flow") or exp.get("ar_flow_sccm") or 30.0),
        "pd_thickness": float(exp.get("pd_thickness") or 0.0),
        "substrate_type": str(exp.get("substrate_type") or "Si Wafer").strip(),
        "core_diameter_um": float(exp.get("core_diameter_um") or 9.0),
    }


def runs_for_material(user_experiments: list, target_material: str) -> list:
    """Real runs that belong to this track. 'Generic' pools everything."""
    if target_material == "Generic":
        return list(user_experiments)
    return [e for e in user_experiments if (e.get("target_material") or "Generic") == target_material]


def build_training_set(user_experiments: list, target_material: str) -> dict:
    real_runs = runs_for_material(user_experiments, target_material)
    n_real = len(real_runs)
    prior_noise = min(PRIOR_MAX_NOISE, PRIOR_BASE_NOISE + PRIOR_NOISE_PER_RUN * n_real)

    X_xrd, y_xrd, a_xrd = [], [], []
    X_wave_real, y_wave_real = [], []
    X_wave_prior, y_wave_prior = [], []

    for p in MATERIAL_PRIORS.get(target_material, MATERIAL_PRIORS["Generic"]):
        # Optional trailing values: pd_thickness (8), substrate_type (9), core diameter (10).
        feat = transform_to_physics_features(
            rf_power=p[0], pressure=p[1], distance=p[2], thickness=p[3], rotation=p[4], ar_flow=p[5],
            pd_thickness=float(p[8]) if len(p) > 8 else 0.0,
            substrate_type=str(p[9]).strip() if len(p) > 9 else "Si Wafer",
            core_diameter_um=float(p[10]) if len(p) > 10 else 9.0,
        )
        X_xrd.append(feat); y_xrd.append(p[6]); a_xrd.append(prior_noise)
        X_wave_prior.append(feat); y_wave_prior.append(p[7])

    for exp in real_runs:
        r = _exp_raw_params(exp)
        feat = transform_to_physics_features(
            rf_power=r["rf_power"], pressure=r["pressure"], distance=r["distance"], thickness=r["thickness"],
            rotation=r["rotation"], ar_flow=r["ar_flow"], pd_thickness=r["pd_thickness"],
            substrate_type=r["substrate_type"], core_diameter_um=r["core_diameter_um"],
        )
        X_xrd.append(feat)
        y_xrd.append(XRD_MAP[normalize_xrd_phase(exp.get("xrd_phase"))])
        a_xrd.append(REAL_RUN_NOISE)
        intensity = _measured_intensity(exp)
        # Runs without a measured intensity are left out of the intensity
        # model entirely (they used to be imputed as the midpoint, i.e. the
        # model was told "this recipe gave an average result" with no evidence).
        if intensity is not None:
            X_wave_real.append(feat); y_wave_real.append(intensity)

    uses_prior_wave = len(y_wave_real) < MIN_REAL_INTENSITY_RUNS
    if uses_prior_wave:
        X_wave = X_wave_prior + X_wave_real
        y_wave = y_wave_prior + y_wave_real
        a_wave = [prior_noise] * len(y_wave_prior) + [REAL_RUN_NOISE] * len(y_wave_real)
    else:
        X_wave, y_wave, a_wave = X_wave_real, y_wave_real, [REAL_RUN_NOISE] * len(y_wave_real)

    min_w, max_w = min(y_wave), max(y_wave)
    denom = (max_w - min_w) if max_w > min_w else 1.0

    return {
        "real_runs": real_runs,
        "n_real": n_real,
        "n_real_intensity": len(y_wave_real),
        "uses_prior_wave": uses_prior_wave,
        "prior_noise": prior_noise,
        "X_xrd": np.array(X_xrd), "y_xrd": np.array(y_xrd), "a_xrd": np.array(a_xrd),
        "X_wave": np.array(X_wave), "y_wave_norm": (np.array(y_wave) - min_w) / denom, "a_wave": np.array(a_wave),
        "min_w": min_w, "denom": denom,
    }


def fit_models(ts: dict, n_restarts: int = 5):
    """Fits the crystallinity and intensity GPs on residuals from their physics priors."""
    gp_xrd = GaussianProcessRegressor(
        kernel=build_physics_kernel(), alpha=ts["a_xrd"],
        n_restarts_optimizer=n_restarts, normalize_y=False, random_state=42,
    )
    gp_xrd.fit(ts["X_xrd"], ts["y_xrd"] - physics_prior_mean(ts["X_xrd"]))

    gp_wave = GaussianProcessRegressor(
        kernel=build_physics_kernel(), alpha=ts["a_wave"],
        n_restarts_optimizer=n_restarts, normalize_y=True, random_state=42,
    )
    gp_wave.fit(ts["X_wave"], ts["y_wave_norm"] - 0.3 * pd_catalyst_prior(ts["X_wave"]))
    return gp_xrd, gp_wave


def predict_models(gp_xrd, gp_wave, X):
    """Returns (xrd mean 0-1, xrd std, intensity mean normalized, intensity std normalized)."""
    res_x, std_x = gp_xrd.predict(X, return_std=True)
    mean_x = np.clip(res_x + physics_prior_mean(X), 0.0, 1.0)
    res_w, std_w = gp_wave.predict(X, return_std=True)
    mean_w = res_w + 0.3 * pd_catalyst_prior(X)
    return mean_x, std_x, mean_w, std_w


def phase_from_score(score: float) -> str:
    if score >= 0.8:
        return "Monoclinic"
    if score >= 0.4:
        return "Partial"
    return "Amorphous"


def _features_for(params, substrate_type, core_diameter_um):
    return transform_to_physics_features(
        rf_power=params[0], pressure=params[1], distance=params[2], thickness=params[3],
        rotation=params[4], ar_flow=params[5], pd_thickness=params[6] if len(params) > 6 else 0.0,
        substrate_type=substrate_type, core_diameter_um=core_diameter_um,
    )


def _feature_relevance(gp) -> dict:
    """Which physics features the fitted kernel thinks matter: a short RBF
    length-scale means the outcome changes quickly along that feature."""
    try:
        ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale).astype(float)
        inv = 1.0 / ls
        rel = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)
    except Exception:
        rel = np.full(len(PHYSICS_FEATURE_NAMES), 0.5)
    return {PHYSICS_FEATURE_NAMES[i]: round(float(np.clip(rel[i], 0.05, 1.0)), 2) for i in range(len(PHYSICS_FEATURE_NAMES))}


# ------------------------------------------------------------------------------
# MAIN OPTIMIZATION LOGIC (DUAL WAFER & FIBER PIGP ROUTING)
# ------------------------------------------------------------------------------
def generate_bayesian_suggestion(
    user_experiments: list,
    recent_suggestions: list = None,
    target_material: str = "Generic",
    model_type: str = "standard",
    acquisition_strategy: str = "ucb",
    measured_core_diameter_um: float = None
) -> dict:
    ts = build_training_set(user_experiments, target_material)
    real_runs = ts["real_runs"]
    real_count = ts["n_real"]
    kappa = 1.5 if real_count <= 15 else 0.5

    sentences = [f"Physics-informed GP trained on {real_count} of your {target_material} runs "
                 f"({ts['n_real_intensity']} with measured intensity) plus literature priors "
                 f"weighted at noise {ts['prior_noise']:.2f}."]
    if ts["uses_prior_wave"]:
        sentences.append(f"Intensity predictions still lean on placeholder literature rows "
                         f"(fewer than {MIN_REAL_INTENSITY_RUNS} measured runs) -- treat the µW estimate as a rough guide.")
    if model_type == "dkl":
        sentences.append("Deep-kernel model is not enabled yet (it needs far more runs than a GP to train reliably); "
                         "the standard GP was used.")

    wafer_exps = [e for e in real_runs if str(e.get("substrate_type", "Si Wafer")).strip() in ("Si Wafer", "Glass")]
    fiber_exps = [e for e in real_runs if str(e.get("substrate_type", "")).strip() == "Optical Fiber"]
    target_substrate = "Optical Fiber" if (len(fiber_exps) >= 3 or target_material == "WO3_Pd") else "Si Wafer"
    if target_substrate == "Optical Fiber":
        sentences.append(f"Substrate focus: optical fiber ({len(fiber_exps)} runs), rotational coverage factor active.")
    else:
        sentences.append(f"Substrate focus: flat substrate, Si wafer / glass ({len(wafer_exps)} runs), optimizing for monoclinic growth.")

    gp_xrd, gp_wave = fit_models(ts, n_restarts=8)

    # Anomaly check on the most recent measured run: does the model, trained
    # with that run included, still struggle to explain it?
    anomaly_detected = False
    last_measured = next((e for e in real_runs if _measured_intensity(e) is not None), None)
    if last_measured is not None and not ts["uses_prior_wave"]:
        r = _exp_raw_params(last_measured)
        f = np.array([_features_for([r["rf_power"], r["pressure"], r["distance"], r["thickness"], r["rotation"], r["ar_flow"], r["pd_thickness"]],
                                    r["substrate_type"], r["core_diameter_um"])])
        _, _, m_w, _ = predict_models(gp_xrd, gp_wave, f)
        actual_norm = (_measured_intensity(last_measured) - ts["min_w"]) / ts["denom"]
        anomaly_detected = abs(float(m_w[0]) - actual_norm) > 0.45

    # --------------------------------------------------------------------------
    # BOUNDS & CANDIDATE POOL GENERATION
    # --------------------------------------------------------------------------
    if real_runs:
        best = _exp_raw_params(max(real_runs, key=lambda e: float(e.get("quality_score") or 0.0)))
        b_rf, b_dist, b_thick, b_ar, b_pd = best["rf_power"], best["distance"], best["thickness"], best["ar_flow"], best["pd_thickness"]
    else:
        b_rf, b_dist, b_thick, b_ar, b_pd = 120.0, 5.0, 200.0, 30.0, 0.0

    # Pressure is fixed, not searched: 5 mTorr was confirmed not to work well
    # on the actual CST8 system, so 8 mTorr is used every time instead.
    press_lo, press_hi = 8.0, 8.0

    pd_bounds = (3.0, 15.0) if target_material == "WO3_Pd" else (0.0, 0.0)

    # The etched-core evanescent sensor only uses the first ~100-200 nm of
    # film, and thick high-index WO3 pulls light out of the bare core, so the
    # fiber / WO3_Pd track searches a thinner range than flat-wafer work.
    if target_material == "WO3_Pd" or target_substrate == "Optical Fiber":
        thick_lo, thick_hi = 30.0, 200.0
    else:
        thick_lo, thick_hi = 100.0, 500.0
    ar_lo, ar_hi = 3.0, 40.0

    b_thick = min(max(b_thick, thick_lo), thick_hi)
    b_ar = min(max(b_ar, ar_lo), ar_hi)
    if pd_bounds[1] > 0.0:
        b_pd = min(max(b_pd, pd_bounds[0]), pd_bounds[1])

    # Physical machine constraints: CST8 RF magnetron sputtering. The search
    # window narrows around the best run as evidence accumulates.
    if real_count <= 5:
        bounds = [(80.0, 150.0), (press_lo, press_hi), (3.0, 7.0), (thick_lo, thick_hi), ROTATION_CHOICES, (ar_lo, ar_hi), pd_bounds]
    else:
        w, wpd = (0.25, (0.6, 1.4, 2.0)) if real_count <= 12 else (0.12, (0.8, 1.2, 1.0))
        bounds = [
            (max(80.0, b_rf * (1 - w)), min(150.0, b_rf * (1 + w))),
            (press_lo, press_hi),
            (max(3.0, b_dist * (1 - w)), min(7.0, b_dist * (1 + w))),
            (max(thick_lo, b_thick * (1 - w)), min(thick_hi, b_thick * (1 + w))),
            ROTATION_CHOICES,
            (max(ar_lo, b_ar * (1 - w)), min(ar_hi, b_ar * (1 + w))),
            pd_bounds if pd_bounds[1] == 0.0 else (max(pd_bounds[0], b_pd * wpd[0]), min(pd_bounds[1], max(b_pd * wpd[1], pd_bounds[0] + wpd[2]))),
        ]

    rng = np.random.default_rng(42 + real_count)
    num_candidates = 3000
    raw_candidates = np.column_stack([
        rng.uniform(bounds[0][0], bounds[0][1], num_candidates),
        rng.uniform(bounds[1][0], bounds[1][1], num_candidates),
        rng.uniform(bounds[2][0], bounds[2][1], num_candidates),
        rng.uniform(bounds[3][0], bounds[3][1], num_candidates),
        rng.choice(bounds[4], num_candidates),
        rng.uniform(bounds[5][0], bounds[5][1], num_candidates),
        rng.uniform(bounds[6][0], bounds[6][1], num_candidates) if bounds[6][1] > bounds[6][0] else np.zeros(num_candidates),
    ])

    validated_cands = matrix_ml_engine.apply_physical_constraints(raw_candidates)
    if not validated_cands:
        validated_cands = [{"params": c, "penalty": 1.0} for c in raw_candidates[:500]]
    candidates = np.array([c["params"] for c in validated_cands])
    penalties = np.array([c["penalty"] for c in validated_cands])

    # The diameter of the specific fiber you're about to coat today -- fixed
    # for this request, not searched. 9.0 um is neutral.
    active_diameter_um = float(measured_core_diameter_um) if measured_core_diameter_um else 9.0

    candidates_physics = np.array([_features_for(c, target_substrate, active_diameter_um) for c in candidates])
    mean_xrd, std_xrd, mean_wave, std_wave = predict_models(gp_xrd, gp_wave, candidates_physics)

    if target_substrate == "Optical Fiber":
        w_xrd, w_wave = 0.35, 0.65
    else:
        w_xrd, w_wave = 0.75, 0.25
    mean_combined = w_xrd * mean_xrd + w_wave * mean_wave
    std_combined = np.sqrt((w_xrd * std_xrd) ** 2 + (w_wave * std_wave) ** 2)

    active_strat = acquisition_strategy if acquisition_strategy in ("ucb", "ts", "hybrid") else "ucb"
    if active_strat == "hybrid":
        active_strat = "ts" if real_count % 2 == 0 else "ucb"

    if active_strat == "ts":
        # Thompson sampling: draw plausible response surfaces from both GPs and
        # take each candidate's best draw. (The old path handed the sklearn
        # model to a GPyTorch-only sampler and crashed.)
        n_draws = 64
        xrd_draws = gp_xrd.sample_y(candidates_physics, n_draws, random_state=7) + physics_prior_mean(candidates_physics)[:, None]
        wave_draws = gp_wave.sample_y(candidates_physics, n_draws, random_state=11) + 0.3 * pd_catalyst_prior(candidates_physics)[:, None]
        combined_draws = w_xrd * np.clip(xrd_draws, 0.0, 1.0) + w_wave * wave_draws
        acquisition = combined_draws[:, int(rng.integers(n_draws))] * penalties
        sentences.append("Exploration strategy: Thompson sampling (one posterior draw of the combined objective).")
    else:
        acquisition = (mean_combined + kappa * std_combined) * penalties
        sentences.append(f"Exploration strategy: upper confidence bound, kappa={kappa}.")

    # Diversity between the three options is measured in normalized parameter
    # space; raw units let film thickness (hundreds of nm) swamp distance (cm).
    span = np.array([max(1e-9, (max(b) - min(b)) if isinstance(b, list) else (b[1] - b[0])) for b in bounds])
    scaled = candidates / span
    min_sep = 0.15 * math.sqrt(int(np.sum(span > 1e-6)) or 1)

    idx_1 = int(np.argmax(acquisition))
    opt1 = candidates[idx_1]
    dist_to_opt1 = np.linalg.norm(scaled - scaled[idx_1], axis=1)

    cost_penalty = 0.5 * (candidates[:, 3] / bounds[3][1]) + 0.5 * (candidates[:, 5] / bounds[5][1])
    if bounds[6][1] > 0:
        # Pd is a precious-metal catalyst layer, so thicker Pd costs more.
        cost_penalty = cost_penalty + 0.3 * (candidates[:, 6] / bounds[6][1])
    acq_eff = mean_combined - 0.3 * cost_penalty
    acq_eff[dist_to_opt1 < min_sep] = -np.inf
    idx_2 = int(np.argmax(acq_eff)) if np.isfinite(acq_eff).any() else idx_1
    opt2 = candidates[idx_2]
    dist_to_opt2 = np.linalg.norm(scaled - scaled[idx_2], axis=1)

    acq_exp = np.copy(std_combined)
    acq_exp[(dist_to_opt1 < min_sep) | (dist_to_opt2 < min_sep)] = -np.inf
    idx_3 = int(np.argmax(acq_exp)) if np.isfinite(acq_exp).any() else int(np.argmax(std_combined))
    opt3 = candidates[idx_3]

    def option(idx, c):
        d = format_candidate(c, mean_xrd[idx], mean_wave[idx], ts["min_w"], ts["denom"], active_diameter_um)
        d["intensity_uncertainty"] = round(float(std_wave[idx] * ts["denom"]), 1)
        d["phase_probability"] = round(float(mean_xrd[idx]), 2)
        return d

    batch_options = [
        {"type": "Max Quality", "data": option(idx_1, opt1)},
        {"type": "High Efficiency", "data": option(idx_2, opt2)},
        {"type": "Pure Exploration", "data": option(idx_3, opt3)},
    ]

    best_candidate = opt1
    s_rf, s_press, s_dist, s_thick, s_rot, s_ar, s_pd = best_candidate
    pred_xrd_val = float(mean_xrd[idx_1])
    pred_wave = round(float(ts["min_w"] + float(mean_wave[idx_1]) * ts["denom"]), 1)

    # Sandbox curves: sweep one parameter around the chosen recipe.
    def generate_sandbox_curve(param_index, bounds_tuple):
        test_vals = np.linspace(min(bounds_tuple), max(bounds_tuple), 20)
        pts = []
        for val in test_vals:
            pt = list(best_candidate)
            pt[param_index] = val
            pts.append(_features_for(pt, target_substrate, active_diameter_um))
        _, _, m_w, s_w = predict_models(gp_xrd, gp_wave, np.array(pts))
        return {
            "x": test_vals.tolist(),
            "y": (m_w * ts["denom"] + ts["min_w"]).tolist(),
            "std": (s_w * ts["denom"]).tolist(),
        }

    sandbox_data = {
        "rf_curve": generate_sandbox_curve(0, bounds[0]),
        # Pressure is pinned at 8 mTorr, so its sweep was a single point;
        # thickness is the more useful second axis.
        "thickness_curve": generate_sandbox_curve(3, bounds[3]),
    }
    if bounds[6][1] > bounds[6][0]:
        sandbox_data["pd_curve"] = generate_sandbox_curve(6, bounds[6])

    converged = False
    if recent_suggestions and len(recent_suggestions) >= 2:
        try:
            last3 = [
                [float(r["suggested_rf_power"]), float(r["suggested_pressure"]), float(r["suggested_distance"]),
                 float(r["suggested_thickness"]), float(r["suggested_rotation"]), float(r["suggested_ar_flow"]),
                 float(r.get("suggested_pd_thickness") or 0.0)]
                for r in recent_suggestions
            ]
            last3.append(best_candidate.tolist())
            variances = np.var(last3, axis=0)
            converged = bool(variances[0] < 20.0 and variances[1] < 0.20 and variances[2] < 0.08 and variances[3] < 200.0)
        except (KeyError, TypeError, ValueError):
            converged = False

    # Confidence now comes from the model's own predictive spread at the
    # chosen recipe, not from a fixed 40/92 by run count. std_combined is on a
    # 0-1 objective scale; 0.05 -> ~90%, 0.25 -> ~50%.
    chosen_std = float(std_combined[idx_1])
    conf_score = int(np.clip(round(100 * math.exp(-2.8 * chosen_std)), 5, 99))
    conf_label = "HIGH" if conf_score >= 75 else ("MEDIUM" if conf_score >= 45 else "LOW")

    sentences.append(f"Derived energy density target: {round(s_rf / (s_dist ** 2), 2)} W/cm².")
    if s_rf > b_rf + 5.0 and real_count <= 12:
        sentences.append(f"Stepping RF power up to {round(s_rf, 1)} W to raise adatom surface mobility.")
    elif s_rf < b_rf - 5.0 and real_count <= 12:
        sentences.append(f"Lowering RF power to {round(s_rf, 1)} W to reduce re-sputtering defects.")
    if anomaly_detected:
        sentences.append("WARNING: your most recent measured run sits far from what the model expects. Inspect target erosion or gas calibration.")

    return {
        "app_name": "MatrixAI",
        "run_number": real_count + 1,
        "kappa_used": round(float(kappa), 2),
        "target_material": target_material,
        "target_substrate": target_substrate,
        "acquisition_used": active_strat,
        "model_used": "standard",
        "hardware_anomaly": anomaly_detected,
        "batch_options": batch_options,
        "suggested": {
            "rf_power": round(float(s_rf), 1),
            "working_pressure": round(float(s_press), 1),
            "target_distance": round(float(s_dist), 1),
            "film_thickness": round(float(s_thick), 1),
            "rotation_speed": float(s_rot),
            "ar_flow": round(float(s_ar), 1),
            "pd_thickness": round(float(s_pd), 1),
            "core_diameter_um": round(active_diameter_um, 2),
        },
        "expected": {
            "xrd_phase": phase_from_score(pred_xrd_val),
            "xrd_score_predicted": round(pred_xrd_val, 2),
            "light_intensity_uw_estimate": pred_wave,
            "light_intensity_uw_uncertainty": round(float(std_wave[idx_1] * ts["denom"]), 1),
        },
        "confidence": {
            "score": conf_score,
            "label": conf_label,
            "basis": "model predictive spread at the suggested recipe",
        },
        "convergence": {
            "converged": converged,
            "convergence_score": min(100, int((real_count / 15.0) * 100)),
            "runs_to_convergence_estimate": 0 if converged else max(1, 15 - real_count),
        },
        "training_data": {
            "real_runs": real_count,
            "real_intensity_runs": ts["n_real_intensity"],
            "literature_priors_for_intensity": ts["uses_prior_wave"],
        },
        "feature_relevance": _feature_relevance(gp_xrd),
        "explanation": " ".join(sentences),
        "digital_twin": sandbox_data,
    }

# ------------------------------------------------------------------------------
# DIGITAL TWIN SANDBOX SIMULATOR (WITH PHYSICS TRANSFORMATION)
# ------------------------------------------------------------------------------
def simulate_sandbox_point(user_experiments: list, target_material: str, slider_params: list) -> dict:
    ts = build_training_set(user_experiments, target_material)
    gp_xrd, gp_wave = fit_models(ts, n_restarts=0)

    target_substrate = "Optical Fiber" if target_material == "WO3_Pd" else "Si Wafer"
    params = [
        slider_params[0], slider_params[1], slider_params[2],
        slider_params[3] if len(slider_params) > 3 else 200.0,
        slider_params[4] if len(slider_params) > 4 else 5.0,
        slider_params[5] if len(slider_params) > 5 else 30.0,
        slider_params[6] if len(slider_params) > 6 and target_material == "WO3_Pd" else 0.0,
    ]
    diameter = slider_params[7] if len(slider_params) > 7 else 9.0
    X = np.array([_features_for(params, target_substrate, diameter)])
    mean_x, _, mean_w, std_w = predict_models(gp_xrd, gp_wave, X)

    xrd_score = float(mean_x[0])
    return {
        "predicted_light_intensity_uw": round(float(ts["min_w"] + float(mean_w[0]) * ts["denom"]), 1),
        "uncertainty": round(float(std_w[0] * ts["denom"]), 1),
        "expected_phase": phase_from_score(xrd_score),
        "xrd_score": round(xrd_score, 2),
        "real_runs_used": ts["n_real"],
    }

# ------------------------------------------------------------------------------
# ACTIVE PHASE MAP GENERATOR
# ------------------------------------------------------------------------------
# Axis ranges per parameter (the old code used 3-10 for everything that
# wasn't RF power, e.g. a 3-10 nm "thickness" axis).
PHASE_MAP_RANGES = {
    "rf_power": (0, (80.0, 150.0)),
    "working_pressure": (1, (3.0, 10.0)),
    "target_distance": (2, (3.0, 7.0)),
    "film_thickness": (3, (30.0, 500.0)),
    "rotation_speed": (4, (5.0, 50.0)),
    "ar_flow": (5, (3.0, 40.0)),
}

def generate_phase_map(user_experiments: list, target_material: str, param_x: str = "rf_power", param_y: str = "working_pressure", resolution: int = 15) -> dict:
    ts = build_training_set(user_experiments, target_material)
    gp_xrd, gp_wave = fit_models(ts, n_restarts=0)
    target_substrate = "Optical Fiber" if target_material == "WO3_Pd" else "Si Wafer"

    idx_x, (x_min, x_max) = PHASE_MAP_RANGES.get(param_x, PHASE_MAP_RANGES["rf_power"])
    idx_y, (y_min, y_max) = PHASE_MAP_RANGES.get(param_y, PHASE_MAP_RANGES["working_pressure"])

    defaults = [120.0, 8.0, 5.0, 200.0, 5.0, 30.0, 0.0]
    diameter = 9.0
    if ts["real_runs"]:
        best = _exp_raw_params(max(ts["real_runs"], key=lambda e: float(e.get("quality_score") or 0.0)))
        defaults = [best["rf_power"], best["pressure"], best["distance"], best["thickness"], best["rotation"], best["ar_flow"], best["pd_thickness"]]
        diameter = best["core_diameter_um"]

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    grid = []
    for y_v in y_vals:
        for x_v in x_vals:
            pt = list(defaults)
            pt[idx_x] = x_v
            pt[idx_y] = y_v
            grid.append(_features_for(pt, target_substrate, diameter))
    mean_x, std_x, _, _ = predict_models(gp_xrd, gp_wave, np.array(grid))
    z = mean_x.reshape(resolution, resolution).round(2).tolist()
    z_std = std_x.reshape(resolution, resolution).round(2).tolist()

    # Real runs on the map, so you can see where the model has evidence.
    points = []
    for e in ts["real_runs"]:
        r = _exp_raw_params(e)
        raw = [r["rf_power"], r["pressure"], r["distance"], r["thickness"], r["rotation"], r["ar_flow"]]
        points.append({"x": raw[idx_x], "y": raw[idx_y], "phase": normalize_xrd_phase(e.get("xrd_phase"))})

    return {
        "x": x_vals.tolist(),
        "y": y_vals.tolist(),
        "z": z,
        "z_std": z_std,
        "points": points,
        "param_x": param_x,
        "param_y": param_y,
    }

# ------------------------------------------------------------------------------
# NOISE CALIBRATION & MODEL VALIDATION
# ------------------------------------------------------------------------------
def calibrate_noise_variance(user_experiments: list, target_material: str = "Generic") -> dict:
    """Reports the measurement noise the intensity GP actually fitted (its
    WhiteKernel), rather than a variance/mean ratio of raw intensities, which
    measured how different your recipes were, not how noisy the sensor is."""
    ts = build_training_set(user_experiments, target_material)
    if ts["n_real_intensity"] < MIN_REAL_INTENSITY_RUNS:
        return {"calibrated_noise": None, "sample_variance": None,
                "message": f"Need at least {MIN_REAL_INTENSITY_RUNS} runs with measured intensity to estimate noise."}
    _, gp_wave = fit_models(ts, n_restarts=3)
    noise_norm = float(gp_wave.kernel_.k2.noise_level)
    # normalize_y=True scales targets by their std, so convert back to uW.
    y_std = float(np.std(ts["y_wave_norm"]) * ts["denom"]) or 1.0
    noise_uw = math.sqrt(noise_norm) * y_std
    return {
        "calibrated_noise": round(noise_norm, 4),
        "sample_variance": round(noise_uw ** 2, 2),
        "noise_uw": round(noise_uw, 2),
        "message": f"Fitted run-to-run noise is about ±{noise_uw:.1f} µW (1σ). Differences smaller than "
                   f"~{2 * noise_uw:.1f} µW between two runs are probably not real.",
    }


def leave_one_out_validation(user_experiments: list, target_material: str = "Generic") -> dict:
    """Honest model check: hide each measured run, predict it from the rest,
    and see how often the truth lands inside the model's ±2σ band."""
    measured = [e for e in runs_for_material(user_experiments, target_material) if _measured_intensity(e) is not None]
    if len(measured) < 5:
        return {"status": "INSUFFICIENT", "n": len(measured),
                "message": f"{len(measured)} measured runs on this track. Validation starts at 5."}

    errors, covered, phase_hits, rows = [], 0, 0, []
    for i, held in enumerate(measured):
        rest = measured[:i] + measured[i + 1:]
        ts = build_training_set(rest, target_material)
        gp_xrd, gp_wave = fit_models(ts, n_restarts=0)
        r = _exp_raw_params(held)
        stype = r["substrate_type"]
        X = np.array([_features_for([r["rf_power"], r["pressure"], r["distance"], r["thickness"], r["rotation"], r["ar_flow"], r["pd_thickness"]],
                                    stype, r["core_diameter_um"])])
        mean_x, _, mean_w, std_w = predict_models(gp_xrd, gp_wave, X)
        pred = float(ts["min_w"] + mean_w[0] * ts["denom"])
        sd = float(std_w[0] * ts["denom"])
        actual = _measured_intensity(held)
        err = abs(pred - actual)
        errors.append(err)
        covered += int(err <= 2 * sd + 1e-9)
        phase_hits += int(phase_from_score(float(mean_x[0])) == normalize_xrd_phase(held.get("xrd_phase")))
        rows.append({"id": held.get("id"), "actual": round(actual, 2), "predicted": round(pred, 2), "sigma": round(sd, 2)})

    n = len(measured)
    actual_vals = np.array([_measured_intensity(e) for e in measured])
    baseline = float(np.mean(np.abs(actual_vals - actual_vals.mean())))
    mae = float(np.mean(errors))
    coverage = covered / n
    if coverage < 0.75:
        status, msg = "OVERCONFIDENT", "The model's error bars are too narrow; trust its µW estimates less."
    elif mae >= baseline:
        status, msg = "NO_SKILL", "Predictions are no better than guessing your average result yet."
    else:
        status, msg = "CALIBRATED", "Error bars contain the truth about as often as they should."
    return {
        "status": status,
        "n": n,
        "mae_uw": round(mae, 2),
        "baseline_mae_uw": round(baseline, 2),
        "coverage_2sigma": round(coverage, 2),
        "phase_accuracy": round(phase_hits / n, 2),
        "message": msg,
        "rows": rows,
    }

# ------------------------------------------------------------------------------
# AUTOMATED SYNTHESIS RECIPE GENERATOR
# ------------------------------------------------------------------------------
def generate_synthesis_recipe(params: dict, target_material: str) -> str:
    rf = params.get("rf_power", 120.0)
    press = params.get("working_pressure", 8.0)
    ar = params.get("ar_flow", 30.0)
    o2 = params.get("o2_flow", 5.0)
    dist = params.get("target_distance", 7.0)
    thick = params.get("film_thickness", 100.0)
    rot = params.get("rotation_speed", 5.0)
    pd_thick = params.get("pd_thickness")

    recipe = f"""# MatrixAI PVD Deposition Recipe ({target_material})
## Target Thickness: {thick} nm | Substrate Rotation: {rot} RPM

### Step 1: Chamber Evacuation & Base Pressure
* **Target Base Pressure:** ≤ $5.0 \\times 10^{-5}$ Torr
* **Substrate Pre-heating:** Room temperature baseline (No in-situ heater). Post-anneal max: 230 °C.

### Step 2: Gas Stabilization
* **Argon (Ar) Flow:** {ar} SCCM
* **Oxygen ($O_2$) Flow:** {o2} SCCM
* **Working Pressure Setpoint:** {press} mTorr (Throttle valve automated stabilization: 120s delay).

### Step 3: Target Pre-Sputtering
* **Shutter Status:** CLOSED
* **RF Power Ramp:** Ramp to {rf} W at 20 W/min to prevent ceramic target thermal shock.
* **Duration:** 5 minutes (Target cleaning phase).

### Step 4: Thin-Film Deposition (WO3, RF magnetron)
* **Shutter Status:** OPEN
* **Target-Substrate Distance:** {dist} cm
* **Estimated Deposition Time:** {int(thick * 0.35)} minutes (Based on kinetic energy density rate model).

### Step 5: Post-Deposition Cool Down
* **RF Power:** Ramp down to 0 W.
* **Gas Flow:** Maintain Ar flow for 10 minutes during chamber vent cool down.
"""

    if target_material == "WO3_Pd" and pd_thick:
        recipe += f"""
### Step 6: Pd Catalyst Layer -- SEPARATE sputtering run, Pd metal target, DC magnetron
* **Vent chamber, swap to Pd target, re-pump to base pressure** (this is a distinct
  run from Steps 1-5, not a continuation -- CST8 cannot hold both targets at once).
* **Target Pd Thickness:** {pd_thick} nm
* **Suggested starting point:** see CALC_PD_RECIPE on the Process Engine page for
  a DC power / time estimate based on your calibrated (or literature) deposition rate.
* **Shutter Status:** OPEN only for the calculated deposition time, then CLOSED.

### Step 7: Final Cool Down & Unload
* **DC Power:** Ramp down to 0 W.
* **Gas Flow:** Maintain Ar flow for 10 minutes during chamber vent cool down.
* Sample is now the complete WO3 + Pd bilayer sensor, ready for characterization.
"""
    return recipe


def suggest_pd_deposition_step(
    target_pd_thickness_nm: float,
    calibrated_rate_nm_per_min: float = None,
    distance_cm: float = 5.0,
    ar_flow_sccm: float = 30.0,
) -> dict:
    """
    Pd is deposited in a SEPARATE sputtering run from WO3 -- different target,
    DC magnetron sputtering (pure metal, not RF -- RF is for insulating/oxide
    targets like WO3; Pd is a metal so DC is the standard, faster choice).
    There is no experimental Pd-deposition data yet for this system, so this
    is NOT a Bayesian-optimized suggestion the way the WO3 parameters are.
    It's a literature-informed starting recipe, meant to be calibrated
    against your own measured deposition rate after a first test run.

    Literature reference point used for the defaults below (DC magnetron Pd
    sputtering onto Si/glass, lab-scale system comparable to CST8):
      - Slavcheva et al. 2014 (Sci. World J.): 100 W DC, 78mm target-substrate
        distance, deposition rate 3.2-4.5 nm/min (varies with Ar pressure
        5e-3 to 1e-1 mbar -- lower pressure gives the faster rate). Pure Ar,
        no O2 (metallic Pd, not an oxide).

    These are starting points, not predictions -- your actual rate depends on
    target purity, chamber geometry, and base pressure, which is why the
    calibrated_rate_nm_per_min argument exists: once you've measured your own
    rate from a single test deposition, pass it in for an exact time instead
    of the literature-based range.
    """
    LIT_DC_POWER_W = 100.0
    LIT_PRESSURE_MTORR = 8.0  # matches the fixed working pressure now used system-wide
    LIT_RATE_LOW_NM_MIN = 3.0
    LIT_RATE_HIGH_NM_MIN = 5.0

    target_pd_thickness_nm = max(0.1, float(target_pd_thickness_nm))

    result = {
        "step": "Pd catalyst layer -- separate DC sputtering run, Pd metal target",
        "suggested_dc_power_w": LIT_DC_POWER_W,
        "suggested_pressure_mtorr": LIT_PRESSURE_MTORR,
        "suggested_ar_flow_sccm": float(ar_flow_sccm),
        "suggested_o2_flow_sccm": 0.0,
        "suggested_distance_cm": float(distance_cm),
        "target_pd_thickness_nm": round(target_pd_thickness_nm, 1),
        "is_calibrated": calibrated_rate_nm_per_min is not None,
    }

    if calibrated_rate_nm_per_min and calibrated_rate_nm_per_min > 0:
        rate = float(calibrated_rate_nm_per_min)
        time_min = target_pd_thickness_nm / rate
        result["calibrated_rate_nm_per_min"] = rate
        result["suggested_time_min"] = round(time_min, 2)
        result["suggested_time_s"] = round(time_min * 60, 0)
        result["note"] = (
            f"Calculated directly from your calibrated deposition rate of "
            f"{rate} nm/min. This is a real number for your system, not an estimate."
        )
    else:
        t_low = target_pd_thickness_nm / LIT_RATE_HIGH_NM_MIN
        t_high = target_pd_thickness_nm / LIT_RATE_LOW_NM_MIN
        result["estimated_time_min_range"] = [round(t_low, 2), round(t_high, 2)]
        result["note"] = (
            "ESTIMATE ONLY, based on a published DC Pd sputtering rate "
            "(3-5 nm/min) at comparable power on a comparable lab-scale "
            "system -- not calibrated to your CST8 system yet. Run one short "
            "test deposition, measure the actual thickness (SEM or "
            "profilometer), divide thickness by time to get your real rate, "
            "then re-run this with that number for an exact suggested time."
        )

    return result


# ------------------------------------------------------------------------------
# SUGGESTION TRACK RECORD
# ------------------------------------------------------------------------------
def suggestion_track_record(runs: list) -> dict:
    """Predicted-vs-actual for runs that came from a suggestion. The
    prediction is the one stored on the run when it was logged, so this
    measures what the model believed at the time, not a refit in hindsight."""
    linked = [r for r in runs if r.get("suggestion_id") is not None]
    points, errors, covered, phase_hits, phase_n = [], [], 0, 0, 0
    for r in linked:
        actual = _measured_intensity(r)
        pred, sigma = r.get("predicted_intensity_uw"), r.get("predicted_intensity_sigma")
        if r.get("predicted_phase"):
            phase_n += 1
            phase_hits += int(normalize_xrd_phase(r.get("predicted_phase")) == normalize_xrd_phase(r.get("xrd_phase")))
        if actual is None or pred is None:
            continue
        err = abs(float(pred) - actual)
        inside = sigma is not None and err <= 2 * float(sigma)
        errors.append(err)
        covered += int(inside)
        points.append({
            "id": r.get("id"),
            "option": r.get("suggestion_option"),
            "predicted": round(float(pred), 2),
            "sigma": round(float(sigma), 2) if sigma is not None else None,
            "actual": round(actual, 2),
            "within_2sigma": inside,
            "followed": r.get("followed_suggestion"),
            "created_at": r.get("created_at"),
        })
    points.sort(key=lambda p: str(p.get("created_at") or ""))

    def mean_quality(rs):
        q = [float(r["quality_score"]) for r in rs if r.get("quality_score") is not None]
        return round(float(np.mean(q)), 1) if q else None

    manual = [r for r in runs if r.get("suggestion_id") is None]
    n = len(points)
    return {
        "linked_runs": len(linked),
        "measured_linked_runs": n,
        "manual_runs": len(manual),
        "mae_uw": round(float(np.mean(errors)), 2) if errors else None,
        "coverage_2sigma": round(covered / n, 2) if n else None,
        "phase_accuracy": round(phase_hits / phase_n, 2) if phase_n else None,
        "mean_quality_suggested": mean_quality(linked),
        "mean_quality_manual": mean_quality(manual),
        "points": points,
    }
