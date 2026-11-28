"""The experiment record: every column, its type, and how to read it from
the ledger form, the JSON API, a CSV import or a literature extraction.

This is the single list of experiment columns. The database migrations,
the INSERT statement and the parsers all follow it, so a new field is
added in one place (plus one migration) instead of six hand-written
column lists that drift apart.
"""
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .science import optimizer

SUBSTRATES = ("Si Wafer", "Glass", "Optical Fiber")
SPUTTER_MODES = ("RF", "DC")
SUGGESTION_OPTIONS = ("Max Quality", "High Efficiency", "Pure Exploration")
MAX_RESPONSE_CURVE_CHARS = 200_000


def to_float(value) -> float | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_text(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def clean_response_curve(raw) -> str | None:
    """The analyzer's JSON (curve + metrics) from the browser, re-serialized,
    or None if missing, oversized or malformed."""
    if not raw:
        return None
    if not isinstance(raw, str):
        try:
            raw = json.dumps(raw)
        except (TypeError, ValueError):
            return None
    if len(raw) > MAX_RESPONSE_CURVE_CHARS:
        return None
    try:
        data = json.loads(raw)
        curve = data.get("curve") or {}
        if not isinstance(curve.get("t"), list) or len(curve["t"]) != len(curve.get("y") or []):
            return None
        return json.dumps(data, separators=(",", ":"))
    except (ValueError, AttributeError, TypeError):
        return None


def _choice(options, default):
    def pick(v):
        v = clean_text(v)
        return v if v in options else default
    return pick


def _upper_choice(options, default):
    def pick(v):
        v = (clean_text(v) or "").upper()
        return v if v in options else default
    return pick


@dataclass(frozen=True)
class Field:
    name: str
    sql_type: str                       # used by the baseline migration
    parse: Callable[[Any], Any] = to_float
    aliases: tuple = ()                 # other names the forms / CSVs use
    default: Any = None                 # applied when the input is missing or blank


# Process parameters keep the defaults the old ledger used; measurements
# default to None (NULL) -- an unmeasured value must never be stored as a
# made-up number, because the optimizer would learn from it.
EXPERIMENT_FIELDS = (
    Field("target_material", "VARCHAR DEFAULT 'Generic'", clean_text, default="Generic"),
    Field("branch_name", "VARCHAR DEFAULT 'main'", clean_text, ("branch",), "main"),
    Field("sputter_mode", "VARCHAR DEFAULT 'RF'", _upper_choice(SPUTTER_MODES, "RF"), default="RF"),
    Field("rf_power", "FLOAT", aliases=("rf_power_w",), default=120.0),
    Field("working_pressure", "FLOAT", aliases=("working_pressure_mtorr", "pressure"), default=5.0),
    Field("ar_flow", "FLOAT", aliases=("ar_flow_sccm",), default=30.0),
    Field("o2_flow", "FLOAT", aliases=("o2_flow_sccm",), default=5.0),
    Field("substrate_temp", "FLOAT", aliases=("substrate_temp_c",), default=25.0),
    Field("target_distance", "FLOAT", aliases=("target_substrate_distance_cm",), default=7.0),
    Field("sputter_time_s", "FLOAT", aliases=("sputtering_time_s",)),   # min+sec handled in parse_experiment
    Field("rotation_speed", "FLOAT", aliases=("rotation_speed_rpm",), default=5.0),
    Field("substrate_type", "VARCHAR DEFAULT 'Si Wafer'", _choice(SUBSTRATES, "Si Wafer"), default="Si Wafer"),
    Field("core_diameter_um", "FLOAT"),
    # Light power through the fiber measured before and after the coating,
    # in µW. Optional; blank means not measured.
    Field("fiber_power_before_uw", "FLOAT", aliases=("fiber_power_before_coating_uw", "power_before_uw")),
    Field("fiber_power_after_uw", "FLOAT", aliases=("fiber_power_after_coating_uw", "power_after_uw")),
    # Pd catalyst step: a separate sputtering run (different target), logged
    # in full so the process is reproducible from the ledger.
    Field("pd_thickness", "FLOAT DEFAULT 0", default=0.0),
    Field("pd_dc_power", "FLOAT"),
    Field("pd_distance_cm", "FLOAT"),
    Field("pd_sputter_time_s", "FLOAT"),
    Field("pd_pressure_mtorr", "FLOAT"),
    Field("pd_ar_flow", "FLOAT"),
    Field("pd_o2_flow", "FLOAT DEFAULT 0", default=0.0),
    # Measurements
    Field("film_thickness", "FLOAT", aliases=("film_thickness_nm",)),
    Field("xrd_phase", "VARCHAR", optimizer.normalize_xrd_phase, ("crystal_phase",), "Amorphous"),
    Field("grain_size", "FLOAT", aliases=("grain_size_nm",)),
    Field("light_intensity_uw", "FLOAT", aliases=("wavelength_shift_pm",)),
    Field("h2_response_time", "FLOAT", aliases=("h2_response_time_s", "h2_response")),
    Field("recovery_time_s", "FLOAT"),
    Field("response_curve", "TEXT", clean_response_curve),
    Field("batch_notes", "TEXT", clean_text, ("notes",), "Manual Entry"),
    # Which suggestion the run came from and what the model predicted then,
    # so predictions can later be checked against what was measured.
    Field("suggestion_id", "BIGINT", lambda v: int(f) if (f := to_float(v)) is not None else None),
    Field("suggestion_option", "VARCHAR", _choice(SUGGESTION_OPTIONS, None)),
    Field("predicted_intensity_uw", "FLOAT"),
    Field("predicted_intensity_sigma", "FLOAT"),
    Field("predicted_phase", "VARCHAR", lambda v: optimizer.normalize_xrd_phase(v) if clean_text(v) else None),
    Field("followed_suggestion", "BOOLEAN",
          lambda v: {"1": True, "0": False, "true": True, "false": False}.get(str(v).lower()) if v is not None else None),
    # Computed by the server, never read from input
    Field("quality_score", "FLOAT", lambda v: None),
)

FIELDS_BY_NAME = {f.name: f for f in EXPERIMENT_FIELDS}
INPUT_FIELDS = tuple(f for f in EXPERIMENT_FIELDS if f.name != "quality_score")
SUGGESTION_LINK_FIELDS = ("suggestion_option", "predicted_intensity_uw", "predicted_intensity_sigma",
                          "predicted_phase", "followed_suggestion")
DEFAULT_SPUTTER_TIME_S = 1800.0


def _raw(source: Mapping, field: Field):
    for key in (field.name, *field.aliases):
        v = source.get(key)
        if v is not None and not (isinstance(v, str) and v.strip() == ""):
            return v
    return None


def parse_experiment(source: Mapping, overrides: Mapping | None = None) -> dict:
    """Turns any input mapping (form, JSON body, CSV row, AI extraction) into
    a clean experiment dict with every input column present."""
    out = {}
    for f in INPUT_FIELDS:
        raw = _raw(source, f)
        value = f.parse(raw) if raw is not None else None
        out[f.name] = value if value is not None else f.default

    if out["sputter_time_s"] is None or out["sputter_time_s"] <= 0:
        minutes = to_float(source.get("sputtering_time_min") or source.get("sputter_time_min")) or 0.0
        seconds = to_float(source.get("sputtering_time_sec") or source.get("sputter_time_sec")) or 0.0
        total = minutes * 60.0 + seconds
        out["sputter_time_s"] = total if total > 0 else DEFAULT_SPUTTER_TIME_S

    # Intensity change = fiber power before coating - after coating, when
    # both were measured and no intensity was typed in.
    if out["light_intensity_uw"] is None and out["fiber_power_before_uw"] is not None             and out["fiber_power_after_uw"] is not None:
        out["light_intensity_uw"] = round(out["fiber_power_before_uw"] - out["fiber_power_after_uw"], 4)

    # Prediction fields only mean something when tied to a stored suggestion.
    if out["suggestion_id"] is None:
        for name in SUGGESTION_LINK_FIELDS:
            out[name] = None

    if overrides:
        out.update(overrides)
    return out


def score_run(run: Mapping, branch_runs: list) -> float:
    """Quality score 0-100 relative to the user's other runs on the branch."""
    def as_scoring(e):
        return {
            "wavelength_shift_pm": e.get("light_intensity_uw"),
            "h2_response_time_s": e.get("h2_response_time"),
            "grain_size_nm": e.get("grain_size"),
        }
    try:
        return optimizer.calculate_quality_score(
            run.get("xrd_phase"), run.get("light_intensity_uw"), run.get("h2_response_time"),
            run.get("grain_size"), [as_scoring(e) for e in branch_runs],
        )
    except Exception as err:          # scoring must never block saving a run
        print(f"Quality score failed, using 50: {err}")
        return 50.0
