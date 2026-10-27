"""Optimizer tests on synthetic WO3 runs. Run with: python -m pytest tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optimizer  # noqa: E402


def make_runs(n, material="WO3", seed=0, with_intensity=True):
    """Fake runs where intensity peaks near 120 W / 150 nm, plus noise."""
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n):
        rf = rng.uniform(80, 150)
        thick = rng.uniform(100, 400)
        intensity = 50 - ((rf - 120) / 10) ** 2 - ((thick - 150) / 50) ** 2 + rng.normal(0, 1.0)
        runs.append({
            "id": i + 1,
            "target_material": material,
            "rf_power": rf,
            "working_pressure": 8.0,
            "target_distance": rng.uniform(3, 7),
            "film_thickness": thick,
            "rotation_speed": 5.0,
            "ar_flow": rng.uniform(10, 40),
            "substrate_type": "Si Wafer",
            "xrd_phase": "MONOCLINIC" if rf > 115 else "amorphous",
            "light_intensity_uw": float(intensity) if with_intensity else None,
            "quality_score": float(intensity),
        })
    return runs


def test_normalize_xrd_phase_is_case_insensitive():
    assert optimizer.normalize_xrd_phase("MONOCLINIC") == "Monoclinic"
    assert optimizer.normalize_xrd_phase(" monoclinic ") == "Monoclinic"
    assert optimizer.normalize_xrd_phase("partial") == "Partial"
    assert optimizer.normalize_xrd_phase(None) == "Amorphous"


def test_quality_score_reads_uppercase_phase():
    assert optimizer.calculate_quality_score("MONOCLINIC", None, None, None, []) > \
        optimizer.calculate_quality_score("Amorphous", None, None, None, [])


def test_training_set_filters_material_and_skips_missing_intensity():
    runs = make_runs(6, "WO3") + make_runs(4, "TiO2", seed=1)
    runs[0]["light_intensity_uw"] = None
    ts = optimizer.build_training_set(runs, "WO3")
    assert ts["n_real"] == 6
    assert ts["n_real_intensity"] == 5
    assert not ts["uses_prior_wave"]
    assert len(ts["y_wave_norm"]) == 5          # priors dropped once real data exists


def test_priors_fade_as_runs_accumulate():
    few = optimizer.build_training_set(make_runs(1), "WO3")["prior_noise"]
    many = optimizer.build_training_set(make_runs(20), "WO3")["prior_noise"]
    assert many > few


@pytest.mark.parametrize("strategy", ["ucb", "ts", "hybrid"])
def test_suggestion_runs_for_every_acquisition(strategy):
    out = optimizer.generate_bayesian_suggestion(make_runs(10), [], "WO3", acquisition_strategy=strategy)
    s = out["suggested"]
    assert 80 <= s["rf_power"] <= 150
    assert s["working_pressure"] == 8.0
    assert out["acquisition_used"] in ("ucb", "ts")
    assert 5 <= out["confidence"]["score"] <= 99
    assert len(out["batch_options"]) == 3
    assert set(out["feature_relevance"]) == set(optimizer.PHYSICS_FEATURE_NAMES)


def test_confidence_rises_with_data():
    low = optimizer.generate_bayesian_suggestion(make_runs(2), [], "WO3")["confidence"]["score"]
    high = optimizer.generate_bayesian_suggestion(make_runs(25), [], "WO3")["confidence"]["score"]
    assert high >= low


def test_batch_options_are_distinct():
    opts = optimizer.generate_bayesian_suggestion(make_runs(10), [], "WO3")["batch_options"]
    recipes = {(o["data"]["rf_power"], o["data"]["film_thickness"], o["data"]["target_distance"]) for o in opts}
    assert len(recipes) == 3


def test_suggestion_without_any_runs():
    out = optimizer.generate_bayesian_suggestion([], [], "WO3_Pd")
    assert 3.0 <= out["suggested"]["pd_thickness"] <= 15.0
    assert out["training_data"]["literature_priors_for_intensity"]


def test_sandbox_uses_real_data_and_accepts_seven_params():
    runs = make_runs(12)
    near_peak = optimizer.simulate_sandbox_point(runs, "WO3", [120, 8, 5, 150, 5, 25, 8])
    far = optimizer.simulate_sandbox_point(runs, "WO3", [82, 8, 5, 390, 5, 25, 8])
    assert near_peak["real_runs_used"] == 12
    assert near_peak["predicted_light_intensity_uw"] > far["predicted_light_intensity_uw"]


def test_phase_map_uses_real_axis_ranges():
    m = optimizer.generate_phase_map(make_runs(8), "WO3", "rf_power", "film_thickness", resolution=5)
    assert min(m["y"]) == 30.0 and max(m["y"]) == 500.0
    assert len(m["z"]) == 5 and len(m["z"][0]) == 5
    assert len(m["points"]) == 8


def test_noise_calibration_reports_microwatts():
    c = optimizer.calibrate_noise_variance(make_runs(15), "WO3")
    assert c["noise_uw"] is not None and c["noise_uw"] > 0
    assert optimizer.calibrate_noise_variance(make_runs(2), "WO3")["calibrated_noise"] is None


def test_leave_one_out_beats_baseline_on_learnable_data():
    v = optimizer.leave_one_out_validation(make_runs(20), "WO3")
    assert v["n"] == 20
    assert v["mae_uw"] < v["baseline_mae_uw"]
    assert 0.0 <= v["coverage_2sigma"] <= 1.0
    assert optimizer.leave_one_out_validation(make_runs(3), "WO3")["status"] == "INSUFFICIENT"


def test_suggestion_track_record_scores_stored_predictions():
    runs = make_runs(6)
    runs[0].update(suggestion_id=1, suggestion_option="Max Quality", predicted_intensity_uw=runs[0]["light_intensity_uw"] + 1,
                   predicted_intensity_sigma=1.0, predicted_phase="Monoclinic", followed_suggestion=True)
    runs[1].update(suggestion_id=2, suggestion_option="Pure Exploration", predicted_intensity_uw=runs[1]["light_intensity_uw"] + 10,
                   predicted_intensity_sigma=1.0, predicted_phase="Amorphous", followed_suggestion=False)
    runs[2].update(suggestion_id=3, predicted_intensity_uw=5.0, light_intensity_uw=None)   # not measured yet
    tr = optimizer.suggestion_track_record(runs)
    assert tr["linked_runs"] == 3 and tr["measured_linked_runs"] == 2 and tr["manual_runs"] == 3
    assert tr["mae_uw"] == pytest.approx(5.5)
    assert tr["coverage_2sigma"] == 0.5
    assert {p["option"] for p in tr["points"]} == {"Max Quality", "Pure Exploration"}
