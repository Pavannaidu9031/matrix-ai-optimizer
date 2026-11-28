"""Gemini / Groq features: co-pilot chat, image reading, paper extraction, thesis text."""
import asyncio
import json
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..auth import current_user, require_founder
from ..models import SUBSTRATES, parse_experiment
from ..science import advanced_features
from ..web import json_body, settings, store
from .experiments import save_runs

router = APIRouter()

CHAT_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"
# Tried in order; the next one is used when a model is overloaded (503).
EXTRACTION_MODELS = ("gemini-3.7-flash", "gemini-3.5-flash", "gemini-2.5-flash")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
CONTEXT_COLUMNS = ("target_material", "rf_power", "working_pressure", "target_distance", "sputter_time_s",
                   "film_thickness", "rotation_speed", "ar_flow", "xrd_phase", "light_intensity_uw", "quality_score")

LITERATURE_PROMPT = (
    "You are an expert materials scientist. Read this research paper and extract the optimal or primary "
    "experimental parameters for thin-film deposition. Return strictly a valid JSON object with these keys "
    "(numbers without units; use null for anything the paper does not report): "
    "'target_material' (string), 'rf_power' (W), 'working_pressure' (mTorr), 'ar_flow' (sccm), "
    "'o2_flow' (sccm), 'substrate_temp' (C), 'target_distance' (cm), 'sputter_time_s' (s), "
    "'film_thickness' (nm), 'rotation_speed' (rpm), 'xrd_phase' (Monoclinic, Partial or Amorphous), "
    "'grain_size' (nm), 'h2_response_time' (s), 'light_intensity_uw' (uW), "
    "'pd_thickness' (nm, Pd catalyst layer, 0 if none), 'pd_dc_power' (W), 'pd_distance_cm', "
    "'pd_sputter_time_s', 'pd_pressure_mtorr', 'pd_ar_flow' (sccm), 'pd_o2_flow' (sccm, usually 0), "
    "'substrate_type' (exactly 'Si Wafer', 'Glass' or 'Optical Fiber': 'Optical Fiber' only if the film was "
    "deposited on a fiber; 'Glass' for glass, quartz, fused silica or ITO/FTO glass; otherwise 'Si Wafer')."
)


def _gemini(request):
    key = settings(request).gemini_api_key
    if not key:
        raise HTTPException(500, "GEMINI_API_KEY not configured.")
    from google import genai
    return genai, genai.Client(api_key=key)


def _first_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(re.sub(r"```json|```", "", match.group(0)).strip())
    except ValueError:
        return None


@router.post("/api/chat")
async def chat(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    message, provider = (body.get("message") or "").strip(), body.get("provider")
    if not message:
        raise HTTPException(422, "message is required")
    try:
        recent = [{k: r.get(k) for k in CONTEXT_COLUMNS} for r in store(request).experiments(user["email"], limit=5)]
    except Exception as err:
        print(f"Chat context unavailable: {err}")
        recent = []
    prompt = ("You are an expert materials-science assistant inside MatrixAI. The user optimizes sputtered "
              "WO3/Pd thin films for optical-fiber H2 sensors. Give concise, scientific, actionable answers.")
    if recent:
        prompt += f"\nTheir most recent experiments: {recent}"
    prompt += f"\n\nUser question: {message}"

    try:
        if provider == "gemini":
            _, client = _gemini(request)
            response = await asyncio.to_thread(client.models.generate_content, model=CHAT_MODEL, contents=prompt)
            return {"reply": response.text}
        if provider == "groq":
            key = settings(request).groq_api_key
            if not key:
                raise HTTPException(500, "GROQ_API_KEY not configured.")
            from groq import AsyncGroq
            completion = await AsyncGroq(api_key=key).chat.completions.create(
                messages=[{"role": "user", "content": prompt}], model=GROQ_MODEL)
            return {"reply": completion.choices[0].message.content}
    except HTTPException:
        raise
    except Exception as err:
        return JSONResponse({"error": str(err)}, status_code=502)
    raise HTTPException(400, "Invalid AI provider selected.")


@router.post("/api/vision")
async def vision(request: Request, image: UploadFile = File(...), analysis_type: str = Form(...),
                 user: dict = Depends(current_user)):
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 20 MB.")
    genai, client = _gemini(request)
    prompt = (f"Analyze this {analysis_type} image for materials science. If it is an XRD plot, identify the "
              "dominant phase (exactly: Monoclinic, Partial or Amorphous). If it is an SEM/FESEM micrograph, "
              "estimate the average grain size in nm (just a number). Return strictly a JSON object with a "
              "single key 'extracted_value'.")
    try:
        response = await asyncio.to_thread(
            client.models.generate_content, model=CHAT_MODEL,
            contents=[prompt, genai.types.Part.from_bytes(data=data, mime_type=image.content_type or "image/png")])
    except Exception as err:
        return JSONResponse({"error": str(err)}, status_code=502)
    parsed = _first_json_object(response.text)
    if not parsed:
        return {"success": False, "message": "Failed to parse AI output. Try again."}
    return {"success": True, "extracted_value": parsed.get("extracted_value")}


@router.post("/api/literature/upload")
async def literature_upload(request: Request, file: UploadFile = File(...), user: dict = Depends(require_founder)):
    pdf = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(pdf) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "PDF is larger than 20 MB.")
    genai, client = _gemini(request)
    part = genai.types.Part.from_bytes(data=pdf, mime_type="application/pdf")
    response, last_err = None, None
    for model in EXTRACTION_MODELS:
        try:
            response = await asyncio.to_thread(client.models.generate_content, model=model, contents=[LITERATURE_PROMPT, part])
            if response and response.text:
                break
        except Exception as err:
            last_err = err
            print(f"Model {model} unavailable ({err}); trying the next one.")
    if not response or not response.text:
        return JSONResponse({"error": f"Extraction failed (503): {last_err or 'all models busy'}"}, status_code=503)

    data = _first_json_object(response.text)
    if data is None:
        return JSONResponse({"error": "Failed to parse the AI's JSON response."}, status_code=502)
    if data.get("substrate_type") not in SUBSTRATES:
        data["substrate_type"] = "Si Wafer"
    record = parse_experiment(data, overrides={"batch_notes": "Literature Prior (AI Extracted)"})
    save_runs(request, user["email"], [record])
    return {"success": True, "extracted_data": data}


@router.post("/api/thesis/generate")
async def thesis_generate(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    runs = store(request).experiments(user["email"])
    content = await asyncio.to_thread(advanced_features.generate_thesis_content, runs, body.get("section"))
    return {"success": True, "content": content}
