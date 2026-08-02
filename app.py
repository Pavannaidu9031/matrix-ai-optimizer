import os
from fastapi import FastAPI, Request, Form, Depends, HTTPException
# ... other imports ...

# Initialize FastAPI App (Render looks specifically for this variable name)
app = FastAPI(title="Rock AI — WO3 Sputtering Optimizer")


@app.post("/add")
def add_experiment(
    request: Request,
    rf_power_w: float = Form(...),
    working_pressure_mtorr: float = Form(...),
    ar_flow_sccm: float = Form(...),
    o2_flow_sccm: float = Form(...),
    substrate_temp_c: float = Form(...),
    target_substrate_distance_cm: float = Form(...),
    sputtering_time_min: float = Form(...),
    film_thickness_nm: float = Form(None),
    rotation_speed_rpm: float = Form(5.0),
    substrate_type: str = Form("Si Wafer"),
    xrd_phase: str = Form("Amorphous"),
    grain_size_nm: float = Form(None),
    h2_response_time_s: float = Form(...),
    wavelength_shift_pm: float = Form(None),
    notes: str = Form(None)
):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    database.add_experiment(
        user_id=user["id"],
        rf_power_w=rf_power_w,
        working_pressure_mtorr=working_pressure_mtorr,
        ar_flow_sccm=ar_flow_sccm,
        o2_flow_sccm=o2_flow_sccm,
        substrate_temp_c=substrate_temp_c,
        target_substrate_distance_cm=target_substrate_distance_cm,
        sputtering_time_min=sputtering_time_min,
        film_thickness_nm=film_thickness_nm,
        rotation_speed_rpm=rotation_speed_rpm,
        substrate_type=substrate_type,
        xrd_phase=xrd_phase,
        grain_size_nm=grain_size_nm,
        h2_response_time_s=h2_response_time_s,
        wavelength_shift_pm=wavelength_shift_pm,
        notes=notes
    )
    return RedirectResponse("/", status_code=303)
