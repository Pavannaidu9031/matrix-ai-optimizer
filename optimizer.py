import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# ------------------------------------------------------------------------------
# UPGRADE 3: Literature Pre-Seeding Prior Knowledge
# ------------------------------------------------------------------------------
LITERATURE_PRIORS = [
    [100.0, 5.0, 7.0, 200.0, 5.0, 30.0, 0.0, 45.0],
    [120.0, 4.0, 5.0, 200.0, 5.0, 30.0, 0.5, 78.0],
    [80.0,  8.0, 7.0, 300.0, 1.0, 25.0, 0.0, 22.0],
    [150.0, 3.0, 3.0, 200.0, 10.0, 30.0, 1.0, 142.0],
    [100.0, 5.0, 5.0, 100.0, 5.0, 30.0, 0.0, 18.0],
    [130.0, 4.0, 4.0, 300.0, 10.0, 35.0, 0.5, 95.0],
    [150.0, 3.0, 3.0, 250.0, 10.0, 40.0, 1.0, 168.0],
]

XRD_MAP = {"Monoclinic": 1.0, "Partial": 0.5, "Amorphous": 0.0}
PARAM_NAMES = ["RF Power", "Pressure", "Target Distance", "Film Thickness", "Rotation Speed", "Ar Flow"]

# ------------------------------------------------------------------------------
# UPGRADE 7: Experiment Quality Score Calculation
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# MAIN OPTIMIZATION LOGIC
# ------------------------------------------------------------------------------
def generate_bayesian_suggestion(user_experiments: list, recent_suggestions: list = None) -> dict:
    real_count = len(user_experiments)

    # --------------------------------------------------------------------------
    # UPGRADE 1: Dynamic Kappa UCB Schedule
    # --------------------------------------------------------------------------
    if real_count <= 7:
        kappa = 1.5  # Lowered from 2.5 to reduce over-exploration in Phase 1
    elif real_count <= 15:
        kappa = 1.5  
    else:
        kappa = 0.5  

    # Prepare datasets
    X_list, y_xrd_list, y_wave_list = [], [], []

    for p in LITERATURE_PRIORS:
        X_list.append(p[:6])
        y_xrd_list.append(p[6])
        y_wave_list.append(p[7])

    for exp in user_experiments:
        rf = float(exp.get("rf_power") or exp.get("rf_power_w") or 120.0)
        press = float(exp.get("working_pressure") or exp.get("working_pressure_mtorr") or 5.0)
        dist = float(exp.get("target_distance") or exp.get("target_substrate_distance_cm") or 7.0)
        thick = float(exp.get("film_thickness") or exp.get("film_thickness_nm") or 200.0)
        rot = float(exp.get("rotation_speed") or exp.get("rotation_speed_rpm") or 5.0)
        ar = float(exp.get("ar_flow") or exp.get("ar_flow_sccm") or 30.0)

        phase = str(exp.get("xrd_phase") or "Amorphous").strip()
        xrd_val = XRD_MAP.get(phase, 0.0)
        
        wave_key = "wavelength_shift" if "wavelength_shift" in exp else "wavelength_shift_pm"
        wave_val = float(exp[wave_key]) if exp.get(wave_key) is not None else None

        X_list.append([rf, press, dist, thick, rot, ar])
        y_xrd_list.append(xrd_val)
        y_wave_list.append(wave_val)

    X = np.array(X_list)
    y_xrd = np.array(y_xrd_list)

    valid_waves = [v for v in y_wave_list if v is not None]
    min_w = min(valid_waves) if valid_waves else 0.0
    max_w = max(valid_waves) if valid_waves else 200.0
    denom = (max_w - min_w) if max_w > min_w else 1.0

    y_wave_norm = np.array([
        (v - min_w) / denom if v is not None else 0.5 for v in y_wave_list
    ])

    # --------------------------------------------------------------------------
    # UPGRADE 2: Anisotropic Gaussian Processes with Physical Intuition
    # --------------------------------------------------------------------------
    physical_length_scales = [10.0, 1.0, 1.0, 50.0, 2.0, 5.0]

    kernel_xrd = ConstantKernel(1.0) * RBF(
        length_scale=physical_length_scales,
        length_scale_bounds=(0.1, 1000)
    ) + WhiteKernel(noise_level=0.1)

    gp_xrd = GaussianProcessRegressor(
        kernel=kernel_xrd, 
        n_restarts_optimizer=10, 
        normalize_y=True,
        random_state=42
    )
    gp_xrd.fit(X, y_xrd)

    kernel_wave = ConstantKernel(1.0) * RBF(
        length_scale=physical_length_scales,
        length_scale_bounds=(0.1, 1000)
    ) + WhiteKernel(noise_level=0.1)
    
    gp_wave = GaussianProcessRegressor(
        kernel=kernel_wave, 
        n_restarts_optimizer=10, 
        normalize_y=True,
        random_state=42
    )
    gp_wave.fit(X, y_wave_norm)

    # --------------------------------------------------------------------------
    # SANITY CHECK (Test known Monoclinic parameters)
    # --------------------------------------------------------------------------
    test_point = np.array([[150.0, 3.0, 3.0, 200.0, 10.0, 30.0]])
    sanity_pred, _ = gp_xrd.predict(test_point, return_std=True)
    model_sanity_check = round(float(sanity_pred[0]), 3)

    # --------------------------------------------------------------------------
    # UPGRADE 6: Adaptive Parameter Bounds
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

    if real_count <= 8:
        bounds = [(80.0, 150.0), (3.0, 10.0), (3.0, 7.0), (100.0, 500.0), [1.0, 5.0, 10.0], (20.0, 40.0)]
    elif real_count <= 15:
        bounds = [
            (max(80.0, b_rf * 0.7), min(150.0, b_rf * 1.3)),
            (max(3.0, b_press * 0.7), min(10.0, b_press * 1.3)),
            (max(3.0, b_dist * 0.7), min(7.0, b_dist * 1.3)),
            (max(100.0, b_thick * 0.7), min(500.0, b_thick * 1.3)),
            [1.0, 5.0, 10.0],
            (max(20.0, b_ar * 0.7), min(40.0, b_ar * 1.3))
        ]
    else:
        bounds = [
            (max(80.0, b_rf * 0.85), min(150.0, b_rf * 1.15)),
            (max(3.0, b_press * 0.85), min(10.0, b_press * 1.15)),
            (max(3.0, b_dist * 0.85), min(7.0, b_dist * 1.15)),
            (max(100.0, b_thick * 0.85), min(500.0, b_thick * 1.15)),
            [1.0, 5.0, 10.0],
            (max(20.0, b_ar * 0.85), min(40.0, b_ar * 1.15))
        ]

    # Sample Candidates
    np.random.seed(42)
    num_candidates = 1000
    c_rf = np.random.uniform(bounds[0][0], bounds[0][1], num_candidates)
    c_press = np.random.uniform(bounds[1][0], bounds[1][1], num_candidates)
    c_dist = np.random.uniform(bounds[2][0], bounds[2][1], num_candidates)
    c_thick = np.random.uniform(bounds[3][0], bounds[3][1], num_candidates)
    c_rot = np.random.choice(bounds[4], num_candidates)
    c_ar = np.random.uniform(bounds[5][0], bounds[5][1], num_candidates)

    candidates = np.column_stack([c_rf, c_press, c_dist, c_thick, c_rot, c_ar])

    mean_xrd, std_xrd = gp_xrd.predict(candidates, return_std=True)
    mean_wave, std_wave = gp_wave.predict(candidates, return_std=True)

    mean_combined = 0.7 * mean_xrd + 0.3 * mean_wave
    std_combined = 0.7 * std_xrd + 0.3 * std_wave
    acquisition = mean_combined + (kappa * std_combined)

    best_idx = int(np.argmax(acquisition))
    best_candidate = candidates[best_idx]

    pred_xrd = float(mean_xrd[best_idx])
    pred_wave_norm = float(mean_wave[best_idx])
    pred_wave_pm = round(float(min_w + pred_wave_norm * denom), 1)
    combined_score = round(float(mean_combined[best_idx]), 2)

    if pred_xrd >= 0.7:
        expected_phase = "Monoclinic"
    elif pred_xrd >= 0.3:
        expected_phase = "Partial"
    else:
        expected_phase = "Amorphous"

    # --------------------------------------------------------------------------
    # UPGRADE 4: Per-Parameter Length Scale Uncertainty Display
    # --------------------------------------------------------------------------
    try:
        # Extract RBF length scales from the nested ConstantKernel * RBF structure
        rbf_k = gp_xrd.kernel_.k1.k2
        l_scales = rbf_k.length_scale
        scaled_ls = (l_scales - np.min(l_scales)) / (np.max(l_scales) - np.min(l_scales) + 1e-6)
    except Exception:
        try:
            # Fallback if structure varies slightly
            l_scales = gp_xrd.kernel_.k1.length_scale
            scaled_ls = (l_scales - np.min(l_scales)) / (np.max(l_scales) - np.min(l_scales) + 1e-6)
        except Exception:
            scaled_ls = np.ones(6) * 0.5

    uncertainties = {}
    for idx, name in enumerate(PARAM_NAMES):
        uncertainties[name] = round(float(np.clip(scaled_ls[idx], 0.1, 1.0)), 2)

    most_unc = PARAM_NAMES[int(np.argmax(scaled_ls))]
    most_cert = PARAM_NAMES[int(np.argmin(scaled_ls))]

    # --------------------------------------------------------------------------
    # UPGRADE 5: Multi-Run Convergence Detection
    # --------------------------------------------------------------------------
    converged = False
    convergence_score = min(100, int((real_count / 20.0) * 100))
    runs_to_conv = max(1, 15 - real_count)

    if recent_suggestions and len(recent_suggestions) >= 2:
        last3 = [
            [
                float(r["suggested_rf_power"]),
                float(r["suggested_pressure"]),
                float(r["suggested_distance"]),
                float(r["suggested_thickness"]),
                float(r["suggested_rotation"]),
                float(r["suggested_ar_flow"]),
            ]
            for r in recent_suggestions
        ]
        last3.append(best_candidate.tolist())

        variances = np.var(last3, axis=0)

        if (
            variances[0] < 25.0
            and variances[1] < 0.25
            and variances[2] < 0.09
            and variances[3] < 225.0
            and variances[4] < 1.0
        ):
            converged = True
            convergence_score = 100
            runs_to_conv = 0

    # --------------------------------------------------------------------------
    # UPGRADE 8: Smart Scientific Explanation Generator
    # --------------------------------------------------------------------------
    s_rf, s_press, s_dist, s_thick, s_rot, s_ar = best_candidate
    sentences = []

    if s_rf > b_rf + 5.0:
        sentences.append(f"Increasing RF power to {round(s_rf, 1)}W to provide higher adatom energy for monoclinic phase nucleation.")
    elif s_rf < b_rf - 5.0:
        sentences.append(f"Lowering RF power to {round(s_rf, 1)}W to reduce plasma damage during film growth.")

    if s_dist < b_dist - 0.5:
        sentences.append(f"Reducing target distance to {round(s_dist, 1)}cm enhances kinetic energy transfer to the growing film.")

    if s_rot >= 5.0:
        sentences.append(f"Substrate rotation at {int(s_rot)} rpm ensures circumferential coating uniformity across non-planar surfaces.")

    if not sentences:
        sentences.append(f"Balancing working pressure at {round(s_press, 1)} mTorr and Ar flow at {round(s_ar, 1)} sccm to maintain optimal stoichiometry.")

    explanation = " ".join(sentences)

    base_conf = 33 if real_count < 6 else (66 if real_count <= 10 else 90)
    conf_score = int(np.clip(base_conf + (7 if not converged else 10), 10, 99))
    conf_label = "HIGH" if conf_score >= 75 else ("MEDIUM" if conf_score >= 45 else "LOW")

    return {
        "app_name": "MatrixAI",
        "run_number": real_count + 1,
        "kappa_used": round(float(kappa), 2),
        "model_sanity_check": model_sanity_check,
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
            "xrd_score": round(pred_xrd, 2),
            "wavelength_shift_estimate": pred_wave_pm,
            "combined_score": combined_score,
        },
        "confidence": {
            "score": conf_score,
            "label": conf_label,
            "data_points_used": real_count + len(LITERATURE_PRIORS),
            "literature_points": len(LITERATURE_PRIORS),
            "real_points": real_count,
        },
        "convergence": {
            "converged": converged,
            "convergence_score": convergence_score,
            "runs_to_convergence_estimate": runs_to_conv,
        },
        "parameter_uncertainties": uncertainties,
        "most_uncertain_parameter": most_unc,
        "most_certain_parameter": most_cert,
        "explanation": explanation,
    }
