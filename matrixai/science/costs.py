"""Rough running-cost estimate for the Resource Burn page."""

RATE_KWH = 0.18          # $ / kWh
RATE_GAS_L = 0.08        # $ / litre
COST_SUBSTRATE = 15.00   # $ / substrate
RATE_TARGET_WH = 0.05    # $ / Wh of target erosion


def _num(value, fallback):
    try:
        return float(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def resource_burn(experiments: list) -> dict:
    kwh = gas_l = 0.0
    erosion_wh = {"WO3": 0.0, "TiO2": 0.0, "ZnO": 0.0}
    for exp in experiments:
        hours = _num(exp.get("sputter_time_s"), 600.0) / 3600.0
        power = _num(exp.get("rf_power"), 100.0)
        temp = _num(exp.get("substrate_temp"), 25.0)
        heater_w = max(0.0, (temp - 25.0) * 4.0)
        kwh += (power + heater_w) * hours / 1000.0
        gas_l += (_num(exp.get("ar_flow"), 20.0) + _num(exp.get("o2_flow"), 2.0)) * 60.0 / 1000.0 * hours
        material = str(exp.get("target_material") or "Generic").strip()
        erosion_wh[material] = erosion_wh.get(material, 0.0) + power * hours

    runs = len(experiments)
    electricity = kwh * RATE_KWH
    gas = gas_l * RATE_GAS_L
    substrates = runs * COST_SUBSTRATE
    target = sum(erosion_wh.values()) * RATE_TARGET_WH
    total = electricity + gas + substrates + target
    return {
        "total_runs": runs,
        "total_electricity_kwh": round(kwh, 2),
        "total_gas_l": round(gas_l, 2),
        "electricity_cost": round(electricity, 2),
        "gas_cost": round(gas, 2),
        "substrates_cost": round(substrates, 2),
        "target_cost": round(target, 2),
        "total_cost": round(total, 2),
        "avg_run_cost": round(total / runs, 2) if runs else 0.0,
        "target_erosion_wh": erosion_wh,
    }
