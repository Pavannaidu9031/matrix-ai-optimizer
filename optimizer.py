import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

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

XRD_MAP = {"Monoclinic": 1.0, "Partial": 0.5, "Amorphous": 0.0}
PARAM_NAMES = ["RF Power", "Pressure", "Target Distance", "Film Thickness", "Rotation Speed", "Ar Flow"]

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

    if pred_xrd >= 0.7: phase = "Monoclinic"
    elif pred_xrd >= 0.3: phase = "Partial"
    else: phase = "Amorphous"
    
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
# MAIN OPTIMIZATION LOGIC (Feature 9: Smart Parameter Bounds Included)
# ------------------------------------------------------------------------------
def generate_bayesian_suggestion(user_experiments: list, recent_suggestions: list = None, target_material: str = "Generic") -> dict:
    real_count = len(user_experiments)

    kappa = 1.5 if real_count <= 15 else 0.5  

    X_list, y_xrd_list, y_wave_list = [], [], []

    priors = MATERIAL_PRIORS.get(target_material, MATERIAL_PRIORS["Generic"])
    for p in priors:
        X_list.append(p[:6])
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

        phase = str(exp.get("xrd_phase") or "Amorphous").strip()
        y_xrd_list.append(XRD_MAP.get(phase, 0.0))
        
        wave_key = "wavelength_shift" if "wavelength_shift" in exp else "wavelength_shift_pm"
        y_wave_list.append(float(exp[wave_key]) if exp.get(wave_key) is not None else None)
        X_list.append([rf, press, dist, thick, rot, ar])

    X = np.array(X_list)
    y_xrd = np.array(y_xrd_list)

    valid_waves = [v for v in y_wave_list if v is not None]
    min_w, max_w = (min(valid_waves), max(valid_waves)) if valid_waves else (0.0, 200.0)
    denom = (max_w - min_w) if max_w > min_w else 1.0

    y_wave_norm = np.array([(v - min_w) / denom if v is not None else 0.5 for v in y_wave_list])

    physical_length_scales = [10.0, 1.0, 1.0, 50.0, 2.0, 5.0]

    kernel_xrd = ConstantKernel(1.0) * RBF(length_scale=physical_length_scales, length_scale_bounds=(0.1, 1000)) + WhiteKernel(noise_level=0.1)
    gp_xrd = GaussianProcessRegressor(kernel=kernel_xrd, n_restarts_optimizer=10, normalize_y=True, random_state=42)
    gp_xrd.fit(X, y_xrd)

    kernel_wave = ConstantKernel(1.0) * RBF(length_scale=physical_length_scales, length_scale_bounds=(0.1, 1000)) + WhiteKernel(noise_level=0.1)
    gp_wave = GaussianProcessRegressor(kernel=kernel_wave, n_restarts_optimizer=10, normalize_y=True, random_state=42)
    gp_wave.fit(X, y_wave_norm)
    
    anomaly_detected = False
    if real_count > 0:
        pred_last, _ = gp_wave.predict([X_list[-1]], return_std=True)
        if abs(pred_last[0] - y_wave_norm[-1]) > 0.4:  
            anomaly_detected = True

    # --------------------------------------------------------------------------
    # FEATURE 9: SMART PARAMETER BOUNDS LEARNING & DEAD ZONE DETECTION
    # --------------------------------------------------------------------------
    sentences = []
    if user_experiments:
        best_run = max(user_experiments, key=lambda e: float(e.get("quality_score") or 0.0))
        b_rf = float(best_run.get("rf_power") or best_run.get("rf_power_w") or 120.0)
        b_press = float(best_run.get("working_pressure") or best_run.get("working_pressure_mtorr") or 5.0)
        b_dist = float(best_run.get("target_distance") or best_run.get("target_substrate_distance_cm") or 5.0)
        b_thick = float(best_run.get("film_thickness") or best_run.get("film_thickness_nm") or 200.0)
        b_ar = float(best_run.get("ar_flow") or best_run.get("ar_flow_sccm") or 30.0)
    else:
        b_rf, b_press, b_dist, b_thick, b_ar = 120.0, 5.0, 5.0, 200.0, 30.0

    # Phased bounds tightening based on accumulated data
    if real_count <= 5:
        # Phase 1: Full Exploration
        bounds = [(15.0, 200.0), (3.0, 10.0), (3.0, 7.0), (100.0, 500.0), [1.0, 5.0, 10.0], (20.0, 40.0)]
    elif real_count <= 12:
        # Phase 2: ±30% around current best (Targeted Exploration)
        bounds = [
            (max(15.0, b_rf * 0.7), min(200.0, b_rf * 1.3)),
            (max(3.0, b_press * 0.7), min(10.0, b_press * 1.3)),
            (max(3.0, b_dist * 0.7), min(7.0, b_dist * 1.3)),
            (max(100.0, b_thick * 0.7), min(500.0, b_thick * 1.3)),
            [1.0, 5.0, 10.0],
            (max(20.0, b_ar * 0.7), min(40.0, b_ar * 1.3))
        ]
        sentences.append("Phase 2: Tightening optimization bounds to ±30% around the optimal parameter region.")
    else:
        # Phase 3: ±15% around current best (Fine-Tuning)
        bounds = [
            (max(15.0, b_rf * 0.85), min(200.0, b_rf * 1.15)),
            (max(3.0, b_press * 0.85), min(10.0, b_press * 1.15)),
            (max(3.0, b_dist * 0.85), min(7.0, b_dist * 1.15)),
            (max(100.0, b_thick * 0.85), min(500.0, b_thick * 1.15)),
            [1.0, 5.0, 10.0],
            (max(20.0, b_ar * 0.85), min(40.0, b_ar * 1.15))
        ]
        sentences.append("Phase 3: Deep convergence mode. Bounding search to ±15% for fine-tuning.")

    # Dead Zone Detection (Filter out consistent failure regions)
    if real_count >= 3:
        amorphous_runs = [e for e in user_experiments if str(e.get("xrd_phase", "")).strip() == "Amorphous"]
        if len(amorphous_runs) >= 3:
            max_amorphous_rf = max([float(e.get("rf_power", 120.0)) for e in amorphous_runs])
            # If all amorphous results happened below a certain RF, exclude that region completely
            if max_amorphous_rf < 90.0:
                bounds[0] = (max(bounds[0][0], 90.0), bounds[0][1])
                sentences.append("Dead Zone Excluded: RF Power < 90W consistently yielded Amorphous films.")

    np.random.seed(42)
    num_candidates = 2000
    candidates = np.column_stack([
        np.random.uniform(bounds[0][0], bounds[0][1], num_candidates),
        np.random.uniform(bounds[1][0], bounds[1][1], num_candidates),
        np.random.uniform(bounds[2][0], bounds[2][1], num_candidates),
        np.random.uniform(bounds[3][0], bounds[3][1], num_candidates),
        np.random.choice(bounds[4], num_candidates),
        np.random.uniform(bounds[5][0], bounds[5][1], num_candidates)
    ])

    mean_xrd, std_xrd = gp_xrd.predict(candidates, return_std=True)
    mean_wave, std_wave = gp_wave.predict(candidates, return_std=True)
    
    mean_combined = 0.7 * mean_xrd + 0.3 * mean_wave
    std_combined = 0.7 * std_xrd + 0.3 * std_wave
    acquisition = mean_combined + (kappa * std_combined)

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
    pred_xrd = float(mean_xrd[idx_1])
    pred_wave_pm = round(float(min_w + float(mean_wave[idx_1]) * denom), 1)

    if pred_xrd >= 0.7: expected_phase = "Monoclinic"
    elif pred_xrd >= 0.3: expected_phase = "Partial"
    else: expected_phase = "Amorphous"

    def generate_sandbox_curve(param_index, base_params, bounds_tuple):
        sandbox_X = []
        test_vals = np.linspace(bounds_tuple[0], bounds_tuple[1], 20)
        for val in test_vals:
            pt = list(base_params)
            pt[param_index] = val
            sandbox_X.append(pt)
        mean_w, std_w = gp_wave.predict(np.array(sandbox_X), return_std=True)
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
        last3 = [[float(r["suggested_rf_power"]), float(r["suggested_pressure"]), float(r["suggested_distance"]), float(r["suggested_thickness"]), float(r["suggested_rotation"]), float(r["suggested_ar_flow"])] for r in recent_suggestions]
        last3.append(best_candidate.tolist())
        variances = np.var(last3, axis=0)
        if variances[0] < 25.0 and variances[1] < 0.25 and variances[2] < 0.09 and variances[3] < 225.0 and variances[4] < 1.0:
            converged = True

    if s_rf > b_rf + 5.0 and real_count <= 12: sentences.append(f"Increasing RF power to {round(s_rf, 1)}W to provide higher adatom energy.")
    elif s_rf < b_rf - 5.0 and real_count <= 12: sentences.append(f"Lowering RF power to {round(s_rf, 1)}W to reduce plasma damage.")
    if anomaly_detected: sentences.append("WARNING: High residual variance detected in recent runs. Check target wear.")
    if not sentences: sentences.append(f"Balancing working pressure at {round(s_press, 1)} mTorr and Ar flow at {round(s_ar, 1)} sccm.")

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
            "score": int(np.clip((33 if real_count < 6 else 90) + (10 if converged else 0), 10, 99)),
            "label": "HIGH" if real_count >= 6 else "MEDIUM",
        },
        "convergence": {
            "converged": converged,
            "convergence_score": min(100, int((real_count / 20.0) * 100)),
            "runs_to_convergence_estimate": 0 if converged else max(1, 15 - real_count),
        },
        "parameter_uncertainties": uncertainties,
        "explanation": " ".join(sentences),
        "digital_twin": sandbox_data
    }

# ------------------------------------------------------------------------------
# DIGITAL TWIN SANDBOX SIMULATOR
# ------------------------------------------------------------------------------
def simulate_sandbox_point(user_experiments: list, target_material: str, slider_params: list) -> dict:
    X_list, y_xrd_list, y_wave_list = [], [], []

    priors = MATERIAL_PRIORS.get(target_material, MATERIAL_PRIORS["Generic"])
    for p in priors:
        X_list.append(p[:6])
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

        phase = str(exp.get("xrd_phase") or "Amorphous").strip()
        y_xrd_list.append(XRD_MAP.get(phase, 0.0))
        
        wave_key = "wavelength_shift" if "wavelength_shift" in exp else "wavelength_shift_pm"
        y_wave_list.append(float(exp[wave_key]) if exp.get(wave_key) is not None else None)
        X_list.append([rf, press, dist, thick, rot, ar])

    X = np.array(X_list)
    y_xrd = np.array(y_xrd_list)

    valid_waves = [v for v in y_wave_list if v is not None]
    min_w, max_w = (min(valid_waves), max(valid_waves)) if valid_waves else (0.0, 200.0)
    denom = (max_w - min_w) if max_w > min_w else 1.0
    y_wave_norm = np.array([(v - min_w) / denom if v is not None else 0.5 for v in y_wave_list])

    physical_length_scales = [10.0, 1.0, 1.0, 50.0, 2.0, 5.0]
    kernel_xrd = ConstantKernel(1.0) * RBF(length_scale=physical_length_scales) + WhiteKernel(noise_level=0.1)
    kernel_wave = ConstantKernel(1.0) * RBF(length_scale=physical_length_scales) + WhiteKernel(noise_level=0.1)

    gp_xrd = GaussianProcessRegressor(kernel=kernel_xrd, normalize_y=True, random_state=42).fit(X, y_xrd)
    gp_wave = GaussianProcessRegressor(kernel=kernel_wave, normalize_y=True, random_state=42).fit(X, y_wave_norm)

    point = np.array([slider_params])
    pred_xrd, std_xrd = gp_xrd.predict(point, return_std=True)
    pred_wave_norm, std_wave = gp_wave.predict(point, return_std=True)

    predicted_shift = round(float(min_w + float(pred_wave_norm[0]) * denom), 1)
    uncertainty_pm = round(float(std_wave[0] * denom), 1)
    xrd_score = float(pred_xrd[0])

    if xrd_score >= 0.7: phase = "Monoclinic"
    elif xrd_score >= 0.3: phase = "Partial"
    else: phase = "Amorphous"

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
    X_list, y_xrd_list, y_wave_list = [], [], []

    priors = MATERIAL_PRIORS.get(target_material, MATERIAL_PRIORS["Generic"])
    for p in priors:
        X_list.append(p[:6])
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

        phase = str(exp.get("xrd_phase") or "Amorphous").strip()
        y_xrd_list.append(XRD_MAP.get(phase, 0.0))
        wave_key = "wavelength_shift" if "wavelength_shift" in exp else "wavelength_shift_pm"
        y_wave_list.append(float(exp[wave_key]) if exp.get(wave_key) is not None else None)
        X_list.append([rf, press, dist, thick, rot, ar])

    X = np.array(X_list)
    y_xrd = np.array(y_xrd_list)

    physical_length_scales = [10.0, 1.0, 1.0, 50.0, 2.0, 5.0]
    kernel_xrd = ConstantKernel(1.0) * RBF(length_scale=physical_length_scales) + WhiteKernel(noise_level=0.1)
    gp_xrd = GaussianProcessRegressor(kernel=kernel_xrd, normalize_y=True, random_state=42).fit(X, y_xrd)

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

    x_min, x_max = (15.0, 200.0) if idx_x == 0 else (3.0, 10.0)
    y_min, y_max = (15.0, 200.0) if idx_y == 0 else (3.0, 10.0)

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)

    grid_z = []
    for y_v in y_vals:
        row = []
        for x_v in x_vals:
            pt = list(defaults)
            pt[idx_x] = x_v
            pt[idx_y] = y_v
            pred = gp_xrd.predict([pt])[0]
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
# AUTOMATED NOISE CALIBRATION (REPETITIVE RUN VARIANCE TESTING)
# ------------------------------------------------------------------------------
def calibrate_noise_variance(user_experiments: list) -> dict:
    if not user_experiments or len(user_experiments) < 3:
        return {"calibrated_noise": 0.1, "message": "Insufficient experiments for robust calibration (need at least 3). Defaulting to 0.1."}
    
    shifts = [float(e.get("wavelength_shift") or e.get("wavelength_shift_pm") or 0.0) for e in user_experiments]
    variance = float(np.var(shifts))
    calibrated_noise = round(float(np.clip(variance / (np.mean(shifts) + 1e-6), 0.01, 0.5)), 4)
    
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
* **Substrate Pre-heating:** 300 °C for 15 minutes.

### Step 2: Gas Stabilization
* **Argon (Ar) Flow:** {ar} SCCM
* **Oxygen ($O_2$) Flow:** {o2} SCCM
* **Working Pressure Setpoint:** {press} mTorr (Throttle valve automated stabilization: 120s delay).

### Step 3: Target Pre-Sputtering
* **Shutter Status:** CLOSED
* **RF Power Ramp:** Ramp to {rf} W at 20 W/min to prevent thermal shock.
* **Duration:** 5 minutes (Target cleaning phase).

### Step 4: Thin-Film Deposition
* **Shutter Status:** OPEN
* **Target-Substrate Distance:** {dist} cm
* **Estimated Deposition Time:** {int(thick * 0.3)} minutes (Assuming standard rate ~0.3 nm/s at {rf}W).

### Step 5: Post-Deposition Cool Down
* **RF Power:** Ramp down to 0 W.
* **Gas Flow:** Maintain Ar flow for 10 minutes during cool down below 100 °C.
"""
    return recipe
