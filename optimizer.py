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
    ]
}

XRD_MAP = {"Monoclinic": 1.0, "Partial": 0.75, "Amorphous": 0.0}
PARAM_NAMES = ["RF Power", "Pressure", "Target Distance", "Film Thickness", "Rotation Speed", "Ar Flow"]
PHYSICS_FEATURE_NAMES = ["Energy Density", "Dep Rate Est", "Energy Per nm", "Plasma Density", "Rotation Factor", "Ar Normalized"]

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
    sputter_time_s: float = None,
    substrate_type: str = "Si Wafer"
) -> list:
    """Transforms raw machine parameters into kinetic and thermodynamic derived features."""
    rf_power = max(10.0, float(rf_power))
    distance = max(1.0, float(distance))
    pressure = max(0.5, float(pressure))
    rotation = max(0.0, float(rotation))
    ar_flow = max(1.0, float(ar_flow))

    # 1. Energy Density (W/cm^2 scaling)
    energy_density = rf_power / (distance ** 2)

    # 2. Estimated Deposition Rate (nm/s scaling for WO3)
    dep_rate_estimate = (rf_power * 0.85) / (distance ** 2)

    # 3. Normalized Film Growth Energy (Energy input per nm)
    if dep_rate_estimate > 0:
        energy_per_nm = energy_density / dep_rate_estimate
    else:
        energy_per_nm = 0.0

    # 4. Plasma Density Index (Ion bombardment intensity)
    plasma_density = rf_power / (pressure * distance)

    # 5. Rotational Uniformity (Critical for 360-deg fiber coverage, neutral for wafer)
    if str(substrate_type).strip() == "Optical Fiber":
        rotation_factor = 1.0 - math.exp(-rotation / 3.0)
    else:
        rotation_factor = 0.5

    # 6. Normalized Argon Flow (Gas density / mean free path baseline)
    ar_normalized = ar_flow / 30.0

    return [
        float(energy_density),
        float(dep_rate_estimate),
        float(energy_per_nm),
        float(plasma_density),
        float(rotation_factor),
        float(ar_normalized)
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

def build_physics_kernel() -> ConstantKernel:
    """Constructs an RBF Kernel bounded directly to physical length-scales."""
    kernel = ConstantKernel(1.0, (0.01, 100.0)) * RBF(
        length_scale=[
            2.0,   # Energy Density
            2.0,   # Deposition Rate
            3.0,   # Energy Per nm
            1.5,   # Plasma Density
            0.5,   # Rotation Factor
            2.0    # Ar Normalized
        ],
        length_scale_bounds=(0.1, 100.0)
    ) + WhiteKernel(
        noise_level=0.05,
        noise_level_bounds=(1e-5, 0.5)
    )
    return kernel

def calculate_quality_score(xrd_phase, wavelength_shift_pm, h2_response_s, grain_size_nm, all_experiments):
    xrd_score = XRD_MAP.get(str(xrd_phase).strip(), 0.0)

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

def format_candidate(candidate_array, mean_xrd, mean_wave, min_w, denom):
    s_rf, s_press, s_dist, s_thick, s_rot, s_ar = candidate_array
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
        "expected_phase": phase,
        "expected_shift": pred_wave_pm
    }

# ------------------------------------------------------------------------------
# MAIN OPTIMIZATION LOGIC (UPGRADE 3: DUAL WAFER & FIBER PIGP ROUTING)
# ------------------------------------------------------------------------------
def generate_bayesian_suggestion(
    user_experiments: list, 
    recent_suggestions: list = None, 
    target_material: str = "Generic", 
    model_type: str = "standard", 
    acquisition_strategy: str = "ucb"
) -> dict:
    real_count = len(user_experiments)
    kappa = 1.5 if real_count <= 15 else 0.5  

    sentences = []
    sentences.append("Physics-Informed Gaussian Process Active: Transformed parameter space into kinetic energy density & plasma dynamics.")

    # Partition experiments into Si Wafer vs Optical Fiber
    wafer_exps = [e for e in user_experiments if str(e.get("substrate_type", "Si Wafer")).strip() == "Si Wafer"]
    fiber_exps = [e for e in user_experiments if str(e.get("substrate_type", "")).strip() == "Optical Fiber"]

    target_substrate = "Optical Fiber" if len(fiber_exps) >= 3 else "Si Wafer"
    
    if target_substrate == "Optical Fiber":
        sentences.append(f"Substrate Focus: Optical Fiber ({len(fiber_exps)} runs). Activating rotational core coverage factor.")
    else:
        sentences.append(f"Substrate Focus: Si Wafer Calibration ({len(wafer_exps)} runs). Optimizing for Monoclinic Phase growth.")

    # --------------------------------------------------------------------------
    # PREPARE PHYSICS FEATURE MATRICES
    # --------------------------------------------------------------------------
    X_physics_list = []
    y_xrd_list = []
    y_wave_list = []

    # Inject Literature Priors transformed through physics pipeline
    priors = MATERIAL_PRIORS.get(target_material, MATERIAL_PRIORS["Generic"])
    for p in priors:
        p_feat = transform_to_physics_features(
            rf_power=p[0], pressure=p[1], distance=p[2], thickness=p[3], rotation=p[4], ar_flow=p[5], substrate_type="Si Wafer"
        )
        X_physics_list.append(p_feat)
        y_xrd_list.append(p[6])
        y_wave_list.append(p[7])

    # Inject Real User Runs
    for exp in user_experiments:
        if exp.get("target_material", target_material) != target_material and target_material != "Generic":
            continue
            
        rf = float(exp.get("rf_power") or exp.get("rf_power_w") or 120.0)
        press = float(exp.get("working_pressure") or exp.get("working_pressure_mtorr") or 5.0)
        dist = float(exp.get("target_distance") or exp.get("target_substrate_distance_cm") or 7.0)
        thick = float(exp.get("film_thickness") or exp.get("film_thickness_nm") or 200.0)
        rot = float(exp.get("rotation_speed") or exp.get("rotation_speed_rpm") or 5.0)
        ar = float(exp.get("ar_flow") or exp.get("ar_flow_sccm") or 30.0)
        stype = str(exp.get("substrate_type", "Si Wafer")).strip()

        p_feat = transform_to_physics_features(
            rf_power=rf, pressure=press, distance=dist, thickness=thick, rotation=rot, ar_flow=ar, substrate_type=stype
        )

        phase = str(exp.get("xrd_phase") or "Amorphous").strip()
        y_xrd_list.append(XRD_MAP.get(phase, 0.0))
        
        wave_key = "wavelength_shift" if "wavelength_shift" in exp else "wavelength_shift_pm"
        y_wave_list.append(float(exp[wave_key]) if exp.get(wave_key) is not None else None)
        X_physics_list.append(p_feat)

    X_physics = np.array(X_physics_list)
    y_xrd = np.array(y_xrd_list)

    valid_waves = [v for v in y_wave_list if v is not None]
    min_w, max_w = (min(valid_waves), max(valid_waves)) if valid_waves else (0.0, 200.0)
    denom = (max_w - min_w) if max_w > min_w else 1.0

    y_wave_norm = np.array([(v - min_w) / denom if v is not None else 0.5 for v in y_wave_list])

    # --------------------------------------------------------------------------
    # FIT PHYSICS-INFORMED GPS (WITH MEAN PRIOR SUBTRACTION)
    # --------------------------------------------------------------------------
    xrd_prior_train = physics_prior_mean(X_physics)
    y_xrd_adjusted = y_xrd - xrd_prior_train

    gp_xrd = GaussianProcessRegressor(
        kernel=build_physics_kernel(), 
        n_restarts_optimizer=15, 
        normalize_y=False, 
        random_state=42
    )
    gp_xrd.fit(X_physics, y_xrd_adjusted)

    gp_wave = GaussianProcessRegressor(
        kernel=build_physics_kernel(), 
        n_restarts_optimizer=15, 
        normalize_y=True, 
        random_state=42
    )
    gp_wave.fit(X_physics, y_wave_norm)
    using_dkl = False

    anomaly_detected = False
    if real_count > 0:
        pred_last, _ = gp_wave.predict([X_physics[-1]], return_std=True)
        if abs(pred_last[0] - y_wave_norm[-1]) > 0.45:
            anomaly_detected = True

    # --------------------------------------------------------------------------
    # BOUNDS & CANDIDATE POOL GENERATION
    # --------------------------------------------------------------------------
    if user_experiments:
        best_run = max(user_experiments, key=lambda e: float(e.get("quality_score") or 0.0))
        b_rf = float(best_run.get("rf_power") or best_run.get("rf_power_w") or 120.0)
        b_press = float(best_run.get("working_pressure") or best_run.get("working_pressure_mtorr") or 5.0)
        b_dist = float(best_run.get("target_distance") or best_run.get("target_substrate_distance_cm") or 5.0)
        b_thick = float(best_run.get("film_thickness") or best_run.get("film_thickness_nm") or 200.0)
        b_ar = float(best_run.get("ar_flow") or best_run.get("ar_flow_sccm") or 30.0)
    else:
        b_rf, b_press, b_dist, b_thick, b_ar = 120.0, 5.0, 5.0, 200.0, 30.0

    # Physical machine constraints: CST8 RF magnetron sputtering
    if real_count <= 5:
        bounds = [(80.0, 150.0), (3.0, 10.0), (3.0, 7.0), (100.0, 500.0), [1.0, 5.0, 10.0], (20.0, 40.0)]
    elif real_count <= 12:
        bounds = [
            (max(80.0, b_rf * 0.75), min(150.0, b_rf * 1.25)),
            (max(3.0, b_press * 0.75), min(10.0, b_press * 1.25)),
            (max(3.0, b_dist * 0.75), min(7.0, b_dist * 1.25)),
            (max(100.0, b_thick * 0.75), min(500.0, b_thick * 1.25)),
            [1.0, 5.0, 10.0],
            (max(20.0, b_ar * 0.75), min(40.0, b_ar * 1.25))
        ]
    else:
        bounds = [
            (max(80.0, b_rf * 0.88), min(150.0, b_rf * 1.12)),
            (max(3.0, b_press * 0.88), min(10.0, b_press * 1.12)),
            (max(3.0, b_dist * 0.88), min(7.0, b_dist * 1.12)),
            (max(100.0, b_thick * 0.88), min(500.0, b_thick * 1.12)),
            [1.0, 5.0, 10.0],
            (max(20.0, b_ar * 0.88), min(40.0, b_ar * 1.12))
        ]

    np.random.seed(42)
    num_candidates = 3000
    raw_candidates = np.column_stack([
        np.random.uniform(bounds[0][0], bounds[0][1], num_candidates),
        np.random.uniform(bounds[1][0], bounds[1][1], num_candidates),
        np.random.uniform(bounds[2][0], bounds[2][1], num_candidates),
        np.random.uniform(bounds[3][0], bounds[3][1], num_candidates),
        np.random.choice(bounds[4], num_candidates),
        np.random.uniform(bounds[5][0], bounds[5][1], num_candidates)
    ])

    validated_cands = matrix_ml_engine.apply_physical_constraints(raw_candidates)
    if not validated_cands:
        validated_cands = [{"params": c, "penalty": 1.0} for c in raw_candidates[:500]]
        
    candidates = np.array([c["params"] for c in validated_cands])
    penalties = np.array([c["penalty"] for c in validated_cands])

    # Transform candidate array into physics features
    candidates_physics = np.array([
        transform_to_physics_features(
            rf_power=c[0], pressure=c[1], distance=c[2], thickness=c[3], rotation=c[4], ar_flow=c[5], substrate_type=target_substrate
        )
        for c in candidates
    ])

    # Predict with Prior Re-addition
    cand_xrd_prior = physics_prior_mean(candidates_physics)
    pred_xrd_res, std_xrd = gp_xrd.predict(candidates_physics, return_std=True)
    mean_xrd = np.clip(pred_xrd_res + cand_xrd_prior, 0.0, 1.0)

    mean_wave, std_wave = gp_wave.predict(candidates_physics, return_std=True)

    # Substrate Specific Multi-Objective Balance
    if target_substrate == "Optical Fiber":
        mean_combined = 0.35 * mean_xrd + 0.65 * mean_wave
        std_combined = 0.35 * std_xrd + 0.65 * std_wave
    else:
        mean_combined = 0.75 * mean_xrd + 0.25 * mean_wave
        std_combined = 0.75 * std_xrd + 0.25 * std_wave
    
    # Acquisition Strategy Evaluation
    active_strat = acquisition_strategy
    if acquisition_strategy == "hybrid":
        active_strat = "ts" if real_count % 2 == 0 else "ucb"
        
    if active_strat == "ts":
        sentences.append("Exploration Strategy: Thompson Sampling on physics feature space.")
        ts_scores = matrix_ml_engine.thompson_sampling(gp_wave, None, candidates_physics, 100)
        acquisition = ts_scores * penalties
    else:
        sentences.append("Exploration Strategy: Upper Confidence Bound (UCB).")
        acquisition = (mean_combined + (kappa * std_combined)) * penalties

    # Selection of Candidates
    idx_1 = int(np.argmax(acquisition))
    opt1 = candidates[idx_1]
    
    cost_penalty = 0.5 * (candidates[:, 3] / bounds[3][1]) + 0.5 * (candidates[:, 5] / bounds[5][1])
    acq_eff = acquisition - (1.2 * cost_penalty) 
    
    dist_to_opt1 = np.linalg.norm(candidates - opt1, axis=1)
    acq_eff[dist_to_opt1 < 10.0] = -9999.0 
    idx_2 = int(np.argmax(acq_eff))
    opt2 = candidates[idx_2]
    
    acq_exp = np.copy(std_combined)
    dist_to_opt2 = np.linalg.norm(candidates - opt2, axis=1)
    acq_exp[(dist_to_opt1 < 10.0) | (dist_to_opt2 < 10.0)] = -9999.0
    idx_3 = int(np.argmax(acq_exp))
    opt3 = candidates[idx_3]

    batch_options = [
        {"type": "Max Quality", "data": format_candidate(opt1, mean_xrd[idx_1], mean_wave[idx_1], min_w, denom)},
        {"type": "High Efficiency", "data": format_candidate(opt2, mean_xrd[idx_2], mean_wave[idx_2], min_w, denom)},
        {"type": "Pure Exploration", "data": format_candidate(opt3, mean_xrd[idx_3], mean_wave[idx_3], min_w, denom)}
    ]

    best_candidate = opt1
    s_rf, s_press, s_dist, s_thick, s_rot, s_ar = best_candidate
    pred_xrd_val = float(mean_xrd[idx_1])
    pred_wave_pm = round(float(min_w + float(mean_wave[idx_1]) * denom), 1)

    if pred_xrd_val >= 0.8:
        expected_phase = "Monoclinic"
    elif pred_xrd_val >= 0.4:
        expected_phase = "Partial"
    else:
        expected_phase = "Amorphous"

    # Sandbox curves
    def generate_sandbox_curve(param_index, base_params, bounds_tuple):
        sandbox_raw = []
        test_vals = np.linspace(bounds_tuple[0], bounds_tuple[1], 20)
        for val in test_vals:
            pt = list(base_params)
            pt[param_index] = val
            sandbox_raw.append(pt)
        
        sandbox_physics = np.array([
            transform_to_physics_features(
                rf_power=pt[0], pressure=pt[1], distance=pt[2], thickness=pt[3], rotation=pt[4], ar_flow=pt[5], substrate_type=target_substrate
            )
            for pt in sandbox_raw
        ])
        
        mean_w, std_w = gp_wave.predict(sandbox_physics, return_std=True)
        return {
            "x": test_vals.tolist(),
            "y": (mean_w * denom + min_w).tolist(),
            "std": (std_w * denom).tolist()
        }

    sandbox_data = {
        "rf_curve": generate_sandbox_curve(0, best_candidate, bounds[0]),
        "pressure_curve": generate_sandbox_curve(1, best_candidate, bounds[1])
    }

    try:
        rbf_k = gp_xrd.kernel_.k1.k2
        l_scales = rbf_k.length_scale
        scaled_ls = (l_scales - np.min(l_scales)) / (np.max(l_scales) - np.min(l_scales) + 1e-6)
    except Exception:
        scaled_ls = np.ones(6) * 0.5
        
    uncertainties = {PARAM_NAMES[i]: round(float(np.clip(scaled_ls[i], 0.1, 1.0)), 2) for i in range(6)}

    converged = False
    if recent_suggestions and len(recent_suggestions) >= 2:
        last3 = [
            [float(r["suggested_rf_power"]), float(r["suggested_pressure"]), float(r["suggested_distance"]), 
             float(r["suggested_thickness"]), float(r["suggested_rotation"]), float(r["suggested_ar_flow"])] 
            for r in recent_suggestions
        ]
        last3.append(best_candidate.tolist())
        variances = np.var(last3, axis=0)
        if variances[0] < 20.0 and variances[1] < 0.20 and variances[2] < 0.08 and variances[3] < 200.0:
            converged = True

    opt_energy_dens = s_rf / (s_dist ** 2)
    sentences.append(f"Derived Energy Density target: {round(opt_energy_dens, 2)} W/cm².")

    if s_rf > b_rf + 5.0 and real_count <= 12:
        sentences.append(f"Stepping RF power up to {round(s_rf, 1)}W to elevate adatom surface mobility.")
    elif s_rf < b_rf - 5.0 and real_count <= 12:
        sentences.append(f"Decreasing RF power to {round(s_rf, 1)}W to eliminate re-sputtering resputter defects.")
    if anomaly_detected:
        sentences.append("WARNING: Residual variance detected in recent runs. Inspect target erosion or gas calibration.")

    return {
        "app_name": "MatrixAI",
        "run_number": real_count + 1,
        "kappa_used": round(float(kappa), 2),
        "target_material": target_material,
        "hardware_anomaly": anomaly_detected,
        "batch_options": batch_options,
        "suggested": {
            "rf_power": round(float(s_rf), 1),
            "working_pressure": round(float(s_press), 1),
            "target_distance": round(float(s_dist), 1),
            "film_thickness": round(float(s_thick), 1),
            "rotation_speed": float(s_rot),
            "ar_flow": round(float(s_ar), 1),
        },
        "expected": {
            "xrd_phase": expected_phase,
            "wavelength_shift_estimate": pred_wave_pm,
        },
        "confidence": {
            "score": int(np.clip((40 if real_count < 6 else 92) + (8 if converged else 0), 10, 99)),
            "label": "HIGH" if real_count >= 6 else "MEDIUM",
        },
        "convergence": {
            "converged": converged,
            "convergence_score": min(100, int((real_count / 15.0) * 100)),
            "runs_to_convergence_estimate": 0 if converged else max(1, 15 - real_count),
        },
        "parameter_uncertainties": uncertainties,
        "explanation": " ".join(sentences),
        "digital_twin": sandbox_data
    }

# ------------------------------------------------------------------------------
# DIGITAL TWIN SANDBOX SIMULATOR (WITH PHYSICS TRANSFORMATION)
# ------------------------------------------------------------------------------
def simulate_sandbox_point(user_experiments: list, target_material: str, slider_params: list) -> dict:
    X_physics_list = []
    y_xrd_list = []
    y_wave_list = []

    priors = MATERIAL_PRIORS.get(target_material, MATERIAL_PRIORS["Generic"])
    for p in priors:
        p_feat = transform_to_physics_features(
            rf_power=p[0], pressure=p[1], distance=p[2], thickness=p[3], rotation=p[4], ar_flow=p[5]
        )
        X_physics_list.append(p_feat)
        y_xrd_list.append(p[6])
        y_wave_list.append(p[7])

    for exp in user_experiments:
        if exp.get("target_material", target_material) != target_material and target_material != "Generic":
            continue
        rf = float(exp.get("rf_power") or exp.get("rf_power_w") or 120.0)
        press = float(exp.get("working_pressure") or exp.get("working_pressure_mtorr") or 5.0)
        dist = float(exp.get("target_distance") or exp.get("target_substrate_distance_cm") or 7.0)
        thick = float(exp.get("film_thickness") or exp.get("film_thickness_nm") or 200.0)
        rot = float(exp.get("rotation_speed") or exp.get("rotation_speed_rpm") or 5.0)
        ar = float(exp.get("ar_flow") or exp.get("ar_flow_sccm") or 30.0)
        stype = str(exp.get("substrate_type", "Si Wafer")).strip()

        p_feat = transform_to_physics_features(
            rf_power=rf, pressure=press, distance=dist, thickness=thick, rotation=rot, ar_flow=ar, substrate_type=stype
        )

        phase = str(exp.get("xrd_phase") or "Amorphous").strip()
        y_xrd_list.append(XRD_MAP.get(phase, 0.0))
        
        wave_key = "wavelength_shift" if "wavelength_shift" in exp else "wavelength_shift_pm"
        y_wave_list.append(float(exp[wave_key]) if exp.get(wave_key) is not None else None)
        X_physics_list.append(p_feat)

    X_physics = np.array(X_physics_list)
    y_xrd = np.array(y_xrd_list)

    valid_waves = [v for v in y_wave_list if v is not None]
    min_w, max_w = (min(valid_waves), max(valid_waves)) if valid_waves else (0.0, 200.0)
    denom = (max_w - min_w) if max_w > min_w else 1.0
    y_wave_norm = np.array([(v - min_w) / denom if v is not None else 0.5 for v in y_wave_list])

    xrd_prior_train = physics_prior_mean(X_physics)
    y_xrd_adjusted = y_xrd - xrd_prior_train

    gp_xrd = GaussianProcessRegressor(kernel=build_physics_kernel(), normalize_y=False, random_state=42).fit(X_physics, y_xrd_adjusted)
    gp_wave = GaussianProcessRegressor(kernel=build_physics_kernel(), normalize_y=True, random_state=42).fit(X_physics, y_wave_norm)

    # Transform slider input
    slider_physics = np.array([
        transform_to_physics_features(
            rf_power=slider_params[0],
            pressure=slider_params[1],
            distance=slider_params[2],
            thickness=slider_params[3] if len(slider_params) > 3 else 200.0,
            rotation=slider_params[4] if len(slider_params) > 4 else 5.0,
            ar_flow=slider_params[5] if len(slider_params) > 5 else 30.0
        )
    ])

    prior_val = physics_prior_mean(slider_physics)[0]
    pred_xrd_res, std_xrd = gp_xrd.predict(slider_physics, return_std=True)
    pred_wave_norm, std_wave = gp_wave.predict(slider_physics, return_std=True)

    predicted_shift = round(float(min_w + float(pred_wave_norm[0]) * denom), 1)
    uncertainty_pm = round(float(std_wave[0] * denom), 1)
    xrd_score = float(np.clip(pred_xrd_res[0] + prior_val, 0.0, 1.0))

    if xrd_score >= 0.8:
        phase = "Monoclinic"
    elif xrd_score >= 0.4:
        phase = "Partial"
    else:
        phase = "Amorphous"

    return {
        "predicted_wavelength_shift": predicted_shift,
        "uncertainty": uncertainty_pm,
        "expected_phase": phase,
        "xrd_score": round(xrd_score, 2)
    }

# ------------------------------------------------------------------------------
# ACTIVE PHASE MAP GENERATOR
# ------------------------------------------------------------------------------
def generate_phase_map(user_experiments: list, target_material: str, param_x: str = "rf_power", param_y: str = "working_pressure", resolution: int = 15) -> dict:
    X_physics_list = []
    y_xrd_list = []

    priors = MATERIAL_PRIORS.get(target_material, MATERIAL_PRIORS["Generic"])
    for p in priors:
        p_feat = transform_to_physics_features(
            rf_power=p[0], pressure=p[1], distance=p[2], thickness=p[3], rotation=p[4], ar_flow=p[5]
        )
        X_physics_list.append(p_feat)
        y_xrd_list.append(p[6])

    for exp in user_experiments:
        if exp.get("target_material", target_material) != target_material and target_material != "Generic":
            continue
        rf = float(exp.get("rf_power") or exp.get("rf_power_w") or 120.0)
        press = float(exp.get("working_pressure") or exp.get("working_pressure_mtorr") or 5.0)
        dist = float(exp.get("target_distance") or exp.get("target_substrate_distance_cm") or 7.0)
        thick = float(exp.get("film_thickness") or exp.get("film_thickness_nm") or 200.0)
        rot = float(exp.get("rotation_speed") or exp.get("rotation_speed_rpm") or 5.0)
        ar = float(exp.get("ar_flow") or exp.get("ar_flow_sccm") or 30.0)

        p_feat = transform_to_physics_features(
            rf_power=rf, pressure=press, distance=dist, thickness=thick, rotation=rot, ar_flow=ar
        )
        phase = str(exp.get("xrd_phase") or "Amorphous").strip()
        y_xrd_list.append(XRD_MAP.get(phase, 0.0))
        X_physics_list.append(p_feat)

    X_physics = np.array(X_physics_list)
    y_xrd = np.array(y_xrd_list)

    xrd_prior_train = physics_prior_mean(X_physics)
    y_xrd_adjusted = y_xrd - xrd_prior_train

    gp_xrd = GaussianProcessRegressor(kernel=build_physics_kernel(), normalize_y=False, random_state=42).fit(X_physics, y_xrd_adjusted)

    param_indices = {"rf_power": 0, "working_pressure": 1, "target_distance": 2, "film_thickness": 3, "rotation_speed": 4, "ar_flow": 5}
    idx_x = param_indices.get(param_x, 0)
    idx_y = param_indices.get(param_y, 1)

    defaults = [120.0, 5.0, 5.0, 200.0, 5.0, 30.0]
    if user_experiments:
        best_run = max(user_experiments, key=lambda e: float(e.get("quality_score") or 0.0))
        defaults[0] = float(best_run.get("rf_power") or best_run.get("rf_power_w") or 120.0)
        defaults[1] = float(best_run.get("working_pressure") or best_run.get("working_pressure_mtorr") or 5.0)
        defaults[2] = float(best_run.get("target_distance") or best_run.get("target_substrate_distance_cm") or 5.0)
        defaults[3] = float(best_run.get("film_thickness") or best_run.get("film_thickness_nm") or 200.0)
        defaults[5] = float(best_run.get("ar_flow") or best_run.get("ar_flow_sccm") or 30.0)

    x_min, x_max = (80.0, 150.0) if idx_x == 0 else (3.0, 10.0)
    y_min, y_max = (80.0, 150.0) if idx_y == 0 else (3.0, 10.0)

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)

    grid_z = []
    for y_v in y_vals:
        row = []
        for x_v in x_vals:
            pt = list(defaults)
            pt[idx_x] = x_v
            pt[idx_y] = y_v
            p_feat = np.array([transform_to_physics_features(pt[0], pt[1], pt[2], pt[3], pt[4], pt[5])])
            prior_val = physics_prior_mean(p_feat)[0]
            pred_res = gp_xrd.predict(p_feat)[0]
            pred = np.clip(pred_res + prior_val, 0.0, 1.0)
            row.append(round(float(pred), 2))
        grid_z.append(row)

    return {
        "x": x_vals.tolist(),
        "y": y_vals.tolist(),
        "z": grid_z,
        "param_x": param_x,
        "param_y": param_y
    }

# ------------------------------------------------------------------------------
# AUTOMATED NOISE CALIBRATION
# ------------------------------------------------------------------------------
def calibrate_noise_variance(user_experiments: list) -> dict:
    if not user_experiments or len(user_experiments) < 3:
        return {"calibrated_noise": 0.05, "message": "Insufficient experiments for robust calibration (need at least 3). Defaulting to 0.05."}
    
    shifts = [float(e.get("wavelength_shift") or e.get("wavelength_shift_pm") or 0.0) for e in user_experiments]
    variance = float(np.var(shifts))
    calibrated_noise = round(float(np.clip(variance / (np.mean(shifts) + 1e-6), 0.001, 0.5)), 4)
    
    return {
        "calibrated_noise": calibrated_noise,
        "sample_variance": round(variance, 2),
        "message": f"Successfully calibrated WhiteKernel noise level to {calibrated_noise} based on experimental variance."
    }

# ------------------------------------------------------------------------------
# AUTOMATED SYNTHESIS RECIPE GENERATOR
# ------------------------------------------------------------------------------
def generate_synthesis_recipe(params: dict, target_material: str) -> str:
    rf = params.get("rf_power", 120.0)
    press = params.get("working_pressure", 5.0)
    ar = params.get("ar_flow", 30.0)
    o2 = params.get("o2_flow", 5.0)
    dist = params.get("target_distance", 7.0)
    thick = params.get("film_thickness", 100.0)
    rot = params.get("rotation_speed", 5.0)

    recipe = f"""# MatrixAI PVD Deposition Recipe ({target_material})
## Target Thickness: {thick} nm | Substrate Rotation: {rot} RPM

### Step 1: Chamber Evacuation & Base Pressure
* **Target Base Pressure:** < $5.0 \\times 10^{-6}$ Torr
* **Substrate Pre-heating:** Room temperature baseline (No in-situ heater). Post-anneal max: 230 °C.

### Step 2: Gas Stabilization
* **Argon (Ar) Flow:** {ar} SCCM
* **Oxygen ($O_2$) Flow:** {o2} SCCM
* **Working Pressure Setpoint:** {press} mTorr (Throttle valve automated stabilization: 120s delay).

### Step 3: Target Pre-Sputtering
* **Shutter Status:** CLOSED
* **RF Power Ramp:** Ramp to {rf} W at 20 W/min to prevent ceramic target thermal shock.
* **Duration:** 5 minutes (Target cleaning phase).

### Step 4: Thin-Film Deposition
* **Shutter Status:** OPEN
* **Target-Substrate Distance:** {dist} cm
* **Estimated Deposition Time:** {int(thick * 0.35)} minutes (Based on kinetic energy density rate model).

### Step 5: Post-Deposition Cool Down
* **RF Power:** Ramp down to 0 W.
* **Gas Flow:** Maintain Ar flow for 10 minutes during chamber vent cool down.
"""
    return recipe
