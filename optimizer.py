import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

XRD_MAP = {"Monoclinic": 1.0, "Partial": 0.5, "Amorphous": 0.0}

PARAM_DISPLAY_NAMES = {
    0: "RF Power",
    1: "Working Pressure",
    2: "Target Distance",
    3: "Film Thickness",
    4: "Rotation Speed",
    5: "Ar Flow"
}

def generate_bayesian_suggestion(experiments: list) -> dict:
    if len(experiments) < 3:
        return {
            "error": "need_more_data",
            "message": "Log at least 3 experiments to enable AI suggestions"
        }

    X, y = [], []
    for exp in experiments:
        rf = float(exp.get("rf_power_w") or 100.0)
        press = float(exp.get("working_pressure_mtorr") or 5.0)
        dist = float(exp.get("target_substrate_distance_cm") or exp.get("target_substrate_distance_mm", 7.0) or 7.0)
        thick = float(exp.get("film_thickness_nm") or 200.0)
        rot = float(exp.get("rotation_speed_rpm") or 5.0)
        ar = float(exp.get("ar_flow_sccm") or 30.0)
        
        phase = str(exp.get("xrd_phase", "Amorphous"))
        score = XRD_MAP.get(phase, 0.0)
        
        X.append([rf, press, dist, thick, rot, ar])
        y.append(score)

    X = np.array(X)
    y = np.array(y)

    kernel = RBF(length_scale=np.ones(6), length_scale_bounds=(1e-1, 1e2)) + WhiteKernel(noise_level=1e-2)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gp.fit(X, y)

    np.random.seed(42)
    num_candidates = 500
    
    c_rf = np.random.uniform(80.0, 150.0, num_candidates)
    c_press = np.random.uniform(3.0, 10.0, num_candidates)
    c_dist = np.random.uniform(3.0, 7.0, num_candidates)
    c_thick = np.random.uniform(100.0, 500.0, num_candidates)
    c_rot = np.random.choice([1.0, 5.0, 10.0], num_candidates)
    c_ar = np.random.uniform(20.0, 40.0, num_candidates)

    candidates = np.column_stack([c_rf, c_press, c_dist, c_thick, c_rot, c_ar])
    means, stds = gp.predict(candidates, return_std=True)

    ucb = means + 1.96 * stds
    best_idx = int(np.argmax(ucb))
    best_candidate = candidates[best_idx]
    best_mean = float(means[best_idx])
    best_std = float(stds[best_idx])

    if best_mean >= 0.75:
        expected_phase = "Monoclinic"
    elif best_mean >= 0.25:
        expected_phase = "Partial"
    else:
        expected_phase = "Amorphous"

    count = len(experiments)
    base_confidence = 33 if count < 6 else (66 if count <= 10 else 90)
    adjusted_score = int(max(10, min(99, base_confidence - (best_std * 20))))
    conf_label = "HIGH" if adjusted_score >= 75 else ("MEDIUM" if adjusted_score >= 45 else "LOW")

    try:
        rbf_kernel = gp.kernel_.k1
        highest_unc_idx = int(np.argmax(rbf_kernel.length_scale))
    except Exception:
        highest_unc_idx = int(np.argmax(np.std(X, axis=0)))

    uncertain_param_name = PARAM_DISPLAY_NAMES.get(highest_unc_idx, "Film Thickness")

    shifts = [float(e["wavelength_shift_pm"]) for e in experiments if e.get("wavelength_shift_pm") is not None]
    wavelength_est = round(float(np.mean(shifts) * 1.15), 1) if shifts else 145.0

    return {
        "run_number": count + 1,
        "suggested": {
            "rf_power": round(float(best_candidate[0]), 1),
            "working_pressure": round(float(best_candidate[1]), 1),
            "target_distance": round(float(best_candidate[2]), 1),
            "film_thickness": round(float(best_candidate[3]), 1),
            "rotation_speed": float(best_candidate[4]),
            "ar_flow": round(float(best_candidate[5]), 1)
        },
        "expected": {
            "xrd_phase": expected_phase,
            "xrd_score": round(best_mean, 2),
            "wavelength_shift_estimate": wavelength_est
        },
        "confidence": {
            "score": adjusted_score,
            "label": conf_label
        },
        "uncertainty_parameter": uncertain_param_name,
        "explanation": f"The GP model targets {expected_phase} crystal phase with high parameter exploration on {uncertain_param_name}."
    }
