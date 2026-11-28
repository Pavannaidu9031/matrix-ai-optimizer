"""H2 response-curve analysis for the optical-fiber sensor.

Takes the raw export from a power meter / data logger (intensity over time
while H2 is switched on and off) and extracts the numbers the optimizer
learns from, measured the same way for every run:

  * intensity change  |plateau - baseline|, in uW
  * response time     t90: H2 on -> 90 % of the full change
  * recovery time     H2 off -> back within 10 % of the change

Pure numpy, no web dependencies, so it is unit-tested on its own.
"""
import csv
import io
import math
import re

import numpy as np

# Multipliers to microwatts. dBm is handled separately (logarithmic).
UNIT_TO_UW = {"w": 1e6, "mw": 1e3, "uw": 1.0, "µw": 1.0, "nw": 1e-3, "pw": 1e-6}

MAX_CURVE_POINTS = 600
MAX_FILE_BYTES = 5 * 1024 * 1024

_TIME_HINTS = ("time", "sec", "elapsed", "t(s)", "t [s]", "t_s")
_POWER_HINTS = ("power", "intensity", "signal", "uw", "µw", "mw", "nw", "dbm", "(w)", "[w]", "watt")


class CurveError(ValueError):
    """Raised with a message that is safe to show the user."""


# ------------------------------------------------------------------------------
# PARSING
# ------------------------------------------------------------------------------
def _to_number(cell: str):
    cell = cell.strip().strip('"').strip()
    if not cell:
        return None
    # hh:mm:ss(.fff) clock times -> seconds since midnight
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2}(?:[.,]\d+)?)", cell)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3).replace(",", "."))
    try:
        return float(cell)
    except ValueError:
        pass
    # European decimal comma: "1,234" (only when it is the sole separator)
    if cell.count(",") == 1 and "." not in cell:
        try:
            return float(cell.replace(",", "."))
        except ValueError:
            return None
    return None


def _split(line: str, delimiter):
    if delimiter is None:
        return line.split()
    return next(csv.reader([line], delimiter=delimiter))


def _guess_delimiter(lines):
    sample = [l for l in lines if l.strip()][:50]
    for d in ("\t", ";", ","):
        counts = [l.count(d) for l in sample]
        # The delimiter appears the same (non-zero) number of times on most lines.
        if counts and max(counts) > 0 and counts.count(max(set(counts), key=counts.count)) >= 0.6 * len(counts):
            if d == "," and all(l.count(";") > 0 for l in sample[-5:]):
                continue    # "1,5;2,3" style: semicolon-separated with decimal commas
            return d
    return None             # whitespace-separated


def parse_curve_file(raw: bytes):
    """Returns (headers, columns) where columns is a list of float arrays
    (NaN where a cell wasn't numeric). Metadata lines before the data block,
    which most instrument exports have, are skipped."""
    if not raw:
        raise CurveError("The file is empty.")
    if len(raw) > MAX_FILE_BYTES:
        raise CurveError("File is larger than 5 MB. Export a shorter window or fewer points.")
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc)
            if "\x00" not in text:
                break
        except UnicodeDecodeError:
            continue
    lines = text.splitlines()
    delimiter = _guess_delimiter(lines)

    rows = [_split(l, delimiter) for l in lines if l.strip()]
    numeric = [sum(_to_number(c) is not None for c in r) for r in rows]

    # The data block = longest run of consecutive rows with >= 2 numeric cells.
    best_start, best_len, start = 0, 0, None
    for i, n in enumerate(numeric + [0]):
        if n >= 2 and start is None:
            start = i
        elif n < 2 and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    if best_len < 10:
        raise CurveError("Couldn't find a block of at least 10 numeric rows (time + intensity).")

    data = rows[best_start:best_start + best_len]
    width = max(len(r) for r in data)
    header_row = rows[best_start - 1] if best_start > 0 else []
    headers = [(header_row[i].strip() if i < len(header_row) else "") or f"Column {i + 1}" for i in range(width)]

    columns = []
    for i in range(width):
        vals = [(_to_number(r[i]) if i < len(r) else None) for r in data]
        columns.append(np.array([np.nan if v is None else v for v in vals], dtype=float))
    # Drop columns that are mostly non-numeric (dates, labels).
    keep = [i for i, c in enumerate(columns) if np.isfinite(c).mean() > 0.8]
    if len(keep) < 2:
        raise CurveError("Need two numeric columns (time and intensity); found fewer.")
    return [headers[i] for i in keep], [columns[i] for i in keep]


def guess_columns(headers, columns):
    """Indexes of (time, intensity) columns, by header name first, then shape."""
    low = [h.lower() for h in headers]
    t_idx = next((i for i, h in enumerate(low) if any(k in h for k in _TIME_HINTS)), None)
    if t_idx is None:
        # A time axis is (almost) strictly increasing.
        mono = [float(np.mean(np.diff(c[np.isfinite(c)]) > 0)) if np.isfinite(c).sum() > 2 else 0 for c in columns]
        t_idx = int(np.argmax(mono)) if max(mono) > 0.95 else 0
    p_idx = next((i for i, h in enumerate(low) if i != t_idx and any(k in h for k in _POWER_HINTS)), None)
    if p_idx is None:
        p_idx = next(i for i in range(len(columns)) if i != t_idx)
    return t_idx, p_idx


def guess_unit(header: str) -> str:
    h = header.lower().replace(" ", "")
    if "dbm" in h:
        return "dbm"
    for unit in ("µw", "uw", "nw", "pw", "mw"):
        if unit in h:
            return unit.replace("µ", "u")
    if re.search(r"[\(\[]w[\)\]]", h) or h.endswith("w"):
        return "w"
    return "uw"


def to_microwatts(values: np.ndarray, unit: str) -> np.ndarray:
    unit = (unit or "uw").lower()
    if unit == "dbm":
        return 1e3 * np.power(10.0, values / 10.0)     # dBm -> mW -> uW
    if unit not in UNIT_TO_UW:
        raise CurveError(f"Unknown unit '{unit}'.")
    return values * UNIT_TO_UW[unit]


# ------------------------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------------------------
def _median_filter(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) < k:
        return y.copy()
    pad = k // 2
    yp = np.pad(y, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(yp, k)
    return np.median(windows, axis=1)


def _robust_noise(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    return float(1.4826 * np.median(np.abs(y - np.median(y))))


def _first_crossing(t, mask, after):
    idx = np.nonzero(mask & (t >= after))[0]
    return int(idx[0]) if len(idx) else None


def analyze_response(t, y_uw, h2_on_s=None, h2_off_s=None) -> dict:
    """Analyzes the first H2 exposure cycle. h2_on_s / h2_off_s are optional
    and count seconds from the first sample (clock-time exports are shifted
    to start at 0); without them the switch points are detected from the
    signal itself."""
    ok = np.isfinite(t) & np.isfinite(y_uw)
    t, y = np.asarray(t, float)[ok], np.asarray(y_uw, float)[ok]
    if len(t) < 10:
        raise CurveError("Fewer than 10 usable points after removing blanks.")
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    t = t - t[0]                                     # elapsed seconds from first sample
    warnings = []

    k = max(3, min(31, (len(y) // 100) | 1))          # odd window, ~1 % of the trace
    ys = _median_filter(y, k)

    # --- H2 on & baseline -----------------------------------------------------------
    if h2_on_s is not None:
        h2_on, on_detected = float(h2_on_s), False
    else:
        # Rough baseline from a short lead-in (5 % of the trace), then the first
        # sustained excursion from it. Requiring k consecutive points keeps a
        # single spike from being taken as the H2 switch.
        lead = max(5, len(ys) // 20)
        rough = float(np.median(ys[:lead]))
        rough_noise = max(_robust_noise(y[:lead]), 1e-12)
        dev = np.abs(ys - rough)
        threshold = max(5 * rough_noise, 0.02 * float(dev.max() or 1))
        above = dev > threshold
        run = np.convolve(above.astype(int), np.ones(k, dtype=int), mode="valid")
        starts = np.nonzero(run == k)[0]
        if not len(starts):
            raise CurveError("No clear response found. Is H2 actually switched on in this recording?")
        h2_on = float(t[max(0, int(starts[0]) - 1)])
        on_detected = True

    pre_mask = t < h2_on
    if pre_mask.sum() < 3:
        raise CurveError("Fewer than 3 points before H2 was switched on; record some baseline first.")
    pre_idx = np.nonzero(pre_mask)[0]
    settled = pre_idx[-max(3, len(pre_idx) // 2):]  # settled part just before switching
    baseline = float(np.median(ys[settled]))
    noise = max(_robust_noise(y[pre_idx]), 1e-12)

    delta = ys - baseline

    # --- plateau ------------------------------------------------------------------
    window = (t >= h2_on) & ((t <= h2_off_s) if h2_off_s is not None else True)
    if window.sum() < 3:
        raise CurveError("Too few points between H2 on and H2 off.")
    peak_idx = int(np.nonzero(window)[0][np.argmax(np.abs(delta[window]))])
    sign = 1.0 if delta[peak_idx] >= 0 else -1.0
    if h2_off_s is not None:
        # Plateau = settled level over the last 20 % of the exposure window.
        w_idx = np.nonzero(window)[0]
        plateau_delta = float(np.median(delta[w_idx[-max(3, len(w_idx) // 5):]]))
    else:
        around = slice(max(0, peak_idx - k), peak_idx + k + 1)
        plateau_delta = float(np.median(delta[around]))
    magnitude = abs(plateau_delta)
    if magnitude <= 0:
        raise CurveError("Signal never left the baseline.")

    # --- response time (t90) ---------------------------------------------------------
    reached = sign * delta >= 0.9 * magnitude
    r_idx = _first_crossing(t, reached, h2_on)
    response_time = round(float(t[r_idx] - h2_on), 2) if r_idx is not None else None
    if response_time is None:
        warnings.append("Signal never reached 90 % of its final change; response time not measured.")

    # --- H2 off & recovery ---------------------------------------------------------------
    if h2_off_s is None:
        # Detected as the moment after the peak where the signal starts falling back.
        leaving = sign * delta < 0.9 * magnitude
        off_idx = _first_crossing(t, leaving, t[peak_idx])
        h2_off = float(t[max(peak_idx, off_idx - 1)]) if off_idx is not None else None
        off_detected = True
    else:
        h2_off, off_detected = float(h2_off_s), False

    recovery_time = None
    if h2_off is not None:
        back = sign * delta <= 0.1 * magnitude
        b_idx = _first_crossing(t, back, h2_off)
        if b_idx is not None:
            recovery_time = round(float(t[b_idx] - h2_off), 2)
        else:
            warnings.append("Signal did not return within 10 % of baseline before the recording ended; recovery not measured.")
    else:
        warnings.append("No H2-off step found in this recording; recovery not measured.")

    snr = magnitude / noise
    if snr < 5:
        warnings.append(f"Low signal-to-noise ({snr:.1f}:1). Treat this result with caution.")
    if on_detected:
        warnings.append("H2-on time was detected automatically. Enter it if you logged it.")
    if off_detected and h2_off is not None and recovery_time is not None:
        warnings.append("H2-off time was detected automatically, which reads a few seconds late, "
                        "so recovery may be slightly understated. Enter the switch time for an exact value.")

    # Count exposure cycles (each fresh rise through 50 % of the change).
    half = (sign * delta >= 0.5 * magnitude).astype(int)
    cycles = int(np.sum(np.diff(half) == 1) + (half[0] == 1))
    if cycles > 1:
        warnings.append(f"{cycles} exposure cycles found; only the first was analyzed.")

    # Downsample for display / storage.
    step = max(1, math.ceil(len(t) / MAX_CURVE_POINTS))
    return {
        "baseline_uw": round(baseline, 4),
        "plateau_uw": round(baseline + plateau_delta, 4),
        "intensity_change_uw": round(magnitude, 4),
        "direction": "decrease" if sign < 0 else "increase",
        "relative_change_pct": round(100.0 * magnitude / abs(baseline), 2) if baseline else None,
        "response_time_s": response_time,
        "recovery_time_s": recovery_time,
        "h2_on_s": round(h2_on, 2),
        "h2_off_s": round(h2_off, 2) if h2_off is not None else None,
        "h2_on_detected": on_detected,
        "h2_off_detected": off_detected,
        "noise_uw": round(noise, 4),
        "snr": round(snr, 1),
        "cycles": max(cycles, 1),
        "n_points": int(len(t)),
        "duration_s": round(float(t[-1]), 2),
        "warnings": warnings,
        "curve": {"t": np.round(t[::step], 3).tolist(), "y": np.round(y[::step], 5).tolist()},
    }


def analyze_file(raw: bytes, time_col=None, intensity_col=None, unit=None,
                 time_unit="s", h2_on_s=None, h2_off_s=None) -> dict:
    headers, columns = parse_curve_file(raw)
    g_t, g_p = guess_columns(headers, columns)
    t_idx = g_t if time_col is None else int(time_col)
    p_idx = g_p if intensity_col is None else int(intensity_col)
    if not (0 <= t_idx < len(columns) and 0 <= p_idx < len(columns)) or t_idx == p_idx:
        raise CurveError("Pick two different columns for time and intensity.")
    unit = unit or guess_unit(headers[p_idx])
    scale = {"s": 1.0, "ms": 1e-3, "min": 60.0}.get(time_unit or "s")
    if scale is None:
        raise CurveError(f"Unknown time unit '{time_unit}'.")
    result = analyze_response(columns[t_idx] * scale, to_microwatts(columns[p_idx], unit), h2_on_s, h2_off_s)
    result.update({"columns": headers, "time_col": t_idx, "intensity_col": p_idx, "unit": unit, "time_unit": time_unit or "s"})
    return result
