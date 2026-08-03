import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# 1. VERIFIED ENCODING: Monoclinic = 1.0 (TARGET), Partial = 0.5, Amorphous = 0.0
XRD_MAP = {
    "Monoclinic": 1.0,
    "Partial": 0.5,
    "Amorphous": 0.0
}

PARAM_DISPLAY_NAMES = {
    0: "RF Power",
    1: "Working Pressure",
    2: "Target Distance",
    3: "Film Thickness",
    4: "Rotation Speed",
    5: "Ar Flow"
}

def generate_bayesian_suggestion(experiments: list) -> dict:
    if not experiments or len(experiments) < 3:
        return {
            "error": "need_more_data",
            "message": "Log at least 3 experiments to enable AI suggestions"
        }

    X, y = [], []
    for exp in experiments:
        rf = float(exp.get("rf_power_w") if exp.get("rf_power_w") is not None else 120.0)
        press = float(exp.get("working_pressure_mtorr") if exp.get("working_pressure_mtorr") is not None else 5.0)
        dist = float(exp.get("target_substrate_distance_cm") if exp.get("target_substrate_distance_cm") is not None else exp.get("target_substrate_distance_mm", 7.0) or 7.0)
        thick = float(exp.get("film_thickness_nm") if exp.get("film_thickness_nm") is not None else 200.0)
        rot = float(exp.get("rotation_speed_rpm") if exp.get("rotation_speed_rpm") is not None else 5.0)
        ar = float(exp.get("ar_flow_sccm") if exp.get("ar_flow_sccm") is not None else 30.0)
        
        # Target optimization target: Monoclinic = 1.0
        phase = str(exp.get("xrd_phase") or "Amorphous").strip()
        score = XRD_MAP.get(phase, 0.0)
        
        X.append([rf, press, dist, thick, rot, ar])
        y.append(score)

    X = np.array(X)
    y = np.array(y)

    # Gaussian Process Model fit
    kernel = RBF(length_scale=np.ones(6), length_scale_bounds=(1e-1, 1e2)) + WhiteKernel(noise_level=1e-2)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gp.fit(X, y)

    # Candidate Sampling (1000 candidates across the search bounds)
    np.random.seed(42)
    num_candidates = 1000
    
    c_rf = np.random.uniform(80.0, 150.0, num_candidates)
    c_press = np.random.uniform(3.0, 10.0, num_candidates)
    c_dist = np.random.uniform(3.0, 7.0, num_candidates)
    c_thick = np.random.uniform(100.0, 500.0, num_candidates)
    c_rot = np.random.choice([1.0, 5.0, 10.0], num_candidates)
    c_ar = np.random.uniform(20.0, 40.0, num_candidates)

    candidates = np.column_stack([c_rf, c_press, c_dist, c_thick, c_rot, c_ar])
    y_pred, y_std = gp.predict(candidates, return_std=True)

    # 2 & 3. UCB ACQUISITION FUNCTION (MAXIMIZATION WITH EXPLORATION)
    # y_pred + 2.576 * y_std balances exploitation and exploration (99% confidence threshold)
    beta = 2.576
    acquisition_score = y_pred + (beta * y_std)
    
    # Select candidate that MAXIMIZES the UCB acquisition score
    best_idx = int(np.argmax(acquisition_score))
    best_candidate = candidates[best_idx]
    predicted_score = float(y_pred[best_idx])
    predicted_std = float(y_std[best_idx])

    # 4. EXPECTED XRD PHASE THRESHOLD VERIFICATION
    # Score > 0.7 = Monoclinic, 0.3 to 0.7 = Partial, < 0.3 = Amorphous
    if predicted_score > 0.7 or (predicted_score + 1.96 * predicted_std) > 0.8:
        expected_phase = "Monoclinic"
    elif predicted_score >= 0.3:
        expected_phase = "Partial"
    else:
        expected_phase = "Amorphous"

    count = len(experiments)
    base_confidence = 33 if count < 6 else (66 if count <= 10 else 90)
    adjusted_score = int(max(10, min(99, base_confidence - (predicted_std * 20))))
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
            "xrd_score": round(predicted_score, 2),
            "wavelength_shift_estimate": wavelength_est
        },
        "confidence": {
            "score": adjusted_score,
            "label": conf_label
        },
        "uncertainty_parameter": uncertain_param_name,
        "explanation": f"UCB exploration prioritizes process parameters with high likelihood of achieving {expected_phase} phase crystallization."
    }
