import numpy as np
import plotly.graph_objects as go
from skopt import gp_minimize
from skopt.space import Real

PARAM_BOUNDS = [
    Real(60.0, 200.0, name="rf_power_w"),
    Real(3.0, 10.0, name="working_pressure_mtorr"),
    Real(0.0, 100.0, name="ar_flow_sccm"),
    Real(0.0, 100.0, name="o2_flow_sccm"),
    Real(25.0, 300.0, name="substrate_temp_c"),
    Real(70.0, 150.0, name="target_substrate_distance_mm"),
    Real(5.0, 90.0, name="sputtering_time_min"),
]

PARAM_NAMES = [
    "RF Power",
    "Pressure",
    "Ar Flow",
    "O₂ Flow",
    "Substrate Temp",
    "Target Dist",
    "Sputter Time",
]


def suggest_next_parameters(experiments: list) -> dict:
    if len(experiments) < 5:
        suggested = [bound.rvs()[0] for bound in PARAM_BOUNDS]
        return _format_suggestion(suggested, is_random=True)

    X, y = _extract_xy(experiments)

    def dummy_objective(x):
        return 0.0

    res = gp_minimize(
        func=dummy_objective,
        dimensions=PARAM_BOUNDS,
        x0=X,
        y0=y,
        n_calls=1,
        random_state=42,
    )

    return _format_suggestion(res.x, is_random=False)


def calculate_feature_importance(experiments: list) -> dict:
    """Calculates approximate relative importance of parameters using variance analysis."""
    if len(experiments) < 3:
        return {}

    X, y = _extract_xy(experiments)
    X_arr = np.array(X)
    y_arr = np.array(y)

    importance = {}
    y_std = np.std(y_arr) if np.std(y_arr) > 0 else 1.0

    for idx, name in enumerate(PARAM_NAMES):
        x_col = X_arr[:, idx]
        if np.std(x_col) > 0:
            corr = abs(np.corrcoef(x_col, y_arr)[0, 1])
            importance[name] = (
                round(float(corr * 100), 1) if not np.isnan(corr) else 10.0
            )
        else:
            importance[name] = 10.0

    return importance


def generate_trend_chart_html(experiments: list) -> str:
    """Generates an interactive Plotly dark-mode trend chart HTML snippet."""
    if not experiments:
        return "<p style='color:#94a3b8;'>No data logged yet for charting.</p>"

    # Reverse list so plot displays chronologically from trial #1 onwards
    exps = list(reversed(experiments))
    ids = [exp["id"] for exp in exps]
    times = [exp["h2_response_time_s"] for exp in exps]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ids,
            y=times,
            mode="lines+markers",
            name="Response Time (s)",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=8, color="#0284c7"),
        )
    )

    fig.update_layout(
        title="H₂ Detection Response Time Trend (Lower is Better)",
        xaxis_title="Experiment Trial ID",
        yaxis_title="H₂ Response Time (seconds)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#f8fafc"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _extract_xy(experiments: list):
    X = []
    y = []
    for exp in experiments:
        X.append([
            float(exp["rf_power_w"]),
            float(exp["working_pressure_mtorr"]),
            float(exp["ar_flow_sccm"]),
            float(exp["o2_flow_sccm"]),
            float(exp["substrate_temp_c"]),
            float(exp["target_substrate_distance_mm"]),
            float(exp["sputtering_time_min"]),
        ])
        y.append(float(exp["h2_response_time_s"]))
    return X, y


def _format_suggestion(values: list, is_random: bool) -> dict:
    return {
        "rf_power_w": round(values[0], 1),
        "working_pressure_mtorr": round(values[1], 2),
        "ar_flow_sccm": round(values[2], 1),
        "o2_flow_sccm": round(values[3], 1),
        "substrate_temp_c": round(values[4], 1),
        "target_substrate_distance_mm": round(values[5], 1),
        "sputtering_time_min": round(values[6], 1),
        "is_random": is_random,
    }