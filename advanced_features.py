import numpy as np
import pandas as pd
from google import genai
import os

def analyze_failure(experiments, current_run):
    """Smart Failure Analysis Engine: Checks if current run deviates from successful clusters."""
    if not experiments: return None
    
    df = pd.DataFrame(experiments)
    df['quality'] = pd.to_numeric(df['quality_score'], errors='coerce').fillna(0)
    successful_runs = df[df['quality'] > 70]
    
    if successful_runs.empty or current_run['quality_score'] >= 50:
        return None
        
    causes = []
    # Drift Indicator 1: Deposition Rate
    current_dep_rate = current_run['film_thickness'] / max(current_run['sputter_time'], 1)
    avg_dep_rate = (successful_runs['film_thickness'] / successful_runs['sputter_time'].clip(lower=1)).mean()
    
    if current_dep_rate < avg_dep_rate * 0.85:
        causes.append({
            "cause": "Target erosion changing deposition rate",
            "action": "Run a deposition rate calibration wafer at identical conditions to verify.",
            "severity": "HIGH"
        })
        
    # Chamber Outgassing / Pressure anomaly
    avg_press = successful_runs['working_pressure'].mean()
    if current_run['working_pressure'] > avg_press * 1.5:
        causes.append({
            "cause": "Chamber outgassing (base pressure not achieved)",
            "action": "Check base pressure log for this run. If >1e-4 Torr, bake out chamber.",
            "severity": "CRITICAL"
        })

    return {
        "is_failure": True,
        "causes": causes if causes else [{"cause": "Unknown parameter drift", "action": "Run baseline calibration", "severity": "MEDIUM"}]
    }

def schedule_experiments(experiments, available_hours, available_days):
    """Experiment Scheduler: Groups Bayesian suggestions into 5.5hr blocks."""
    cycle_time_hrs = 5.5 # Deposition(3) + XRD(1) + FESEM(1) + Logging(0.5)
    total_capacity = int(available_hours // cycle_time_hrs)
    
    schedule = []
    current_day = 1
    runs_scheduled = 0
    
    while runs_scheduled < total_capacity and current_day <= available_days:
        schedule.append({
            "day": f"Day {current_day}",
            "task": f"Run {len(experiments) + runs_scheduled + 1}: Sputtering + XRD/FESEM Characterization",
            "estimated_time": f"{cycle_time_hrs} hrs"
        })
        runs_scheduled += 1
        current_day += 1 if runs_scheduled % 2 == 0 else 0 # Max 2 runs per day
        
    return schedule

def generate_thesis_content(experiments, section):
    """Thesis Writing Assistant & Novelty Checker using Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not experiments: return "API Key missing or insufficient data."
    
    client = genai.Client(api_key=api_key)
    df = pd.DataFrame(experiments)
    
    context = f"Total Runs: {len(df)}. Best Shift: {df['wavelength_shift'].max() if 'wavelength_shift' in df else 'N/A'}. Materials: {df['target_material'].unique() if 'target_material' in df else 'WO3'}."
    
    prompts = {
        "results": f"Write an academic results chapter paragraph summarizing this PVD optimization data: {context}",
        "discussion": f"Generate 5 academic discussion points for this PVD sputtering data: {context}",
        "abstract": f"Write a 250-word academic abstract for a paper optimizing thin films using Bayesian ML: {context}",
        "novelty": f"Analyze the novelty of these findings compared to standard WO3 PVD literature and suggest 5 journals: {context}"
    }
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompts.get(section, prompts["results"])
        )
        return response.text
    except Exception as e:
        return str(e)
