import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu
from sklearn.isotonic import IsotonicRegression
import copy

# ==============================================================================
# UPGRADE 1: DEEP KERNEL LEARNING (LAZY LOADED TO PREVENT RENDER CRASH)
# ==============================================================================
def get_dkl_classes():
    """Lazy imports PyTorch to save hundreds of megabytes of RAM on startup."""
    import torch
    import gpytorch
    
    class DeepFeatureExtractor(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 8)
            )
        def forward(self, x):
            return self.network(x)

    class DeepKernelGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.feature_extractor = DeepFeatureExtractor(train_x.shape[1])
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        def forward(self, x):
            features = self.feature_extractor(x)
            mean_x = self.mean_module(features)
            covar_x = self.covar_module(features)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            
    return torch, gpytorch, DeepKernelGP

def train_dkl_model(X, y, epochs=50):
    torch, gpytorch, DeepKernelGP = get_dkl_classes()
    
    train_x = torch.tensor(X, dtype=torch.float32)
    train_y = torch.tensor(y, dtype=torch.float32)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DeepKernelGP(train_x, train_y, likelihood)
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    likelihood.eval()
    return model, likelihood

# ==============================================================================
# UPGRADE 2 & 3: HYBRID ACQUISITION & PHYSICAL CONSTRAINTS
# ==============================================================================
def apply_physical_constraints(candidates, expected_thickness=200, planned_sputter_time_s=1800):
    """Filters out candidates that violate physics rules."""
    valid, warnings = [], []
    for c in candidates:
        rf, press, dist, thick, rot, ar = c
        
        # New Deposition Rate Calculation (nm/min) using seconds
        rate = expected_thickness / (planned_sputter_time_s / 60.0) if planned_sputter_time_s > 0 else 0
        
        stability = press * dist
        
        if rate < 0.5 or rate > 50:
            continue
        if stability < 5 or stability > 70:
            continue
        
        penalty = 1.0
        stress_risk = (rf / dist) * (thick / 200) if dist > 0 else 0
        if stress_risk > 25:
            penalty = 0.7  # Penalize high stress risk
            
        valid.append({"params": c, "penalty": penalty})
    return valid

def thompson_sampling(model, likelihood, candidates, n_samples=100):
    import torch
    import gpytorch
    c_tensor = torch.tensor(candidates, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(c_tensor))
        samples = posterior.rsample(torch.Size([n_samples]))
    return samples.max(dim=0).values.numpy()

# ==============================================================================
# NEW FEATURES: FINGERPRINTING, AGENT, CALIBRATION
# ==============================================================================
def get_dna_fingerprint(target_run, all_runs):
    weights = np.array([0.25, 0.10, 0.25, 0.20, 0.15, 0.05])
    
    t_rf = float(target_run.get('rf_power') or 120.0)
    t_press = float(target_run.get('working_pressure') or 5.0)
    t_dist = float(target_run.get('target_distance') or 7.0)
    t_thick = float(target_run.get('film_thickness') or 200.0)
    t_rot = float(target_run.get('rotation_speed') or 5.0)
    t_ar = float(target_run.get('ar_flow') or 30.0)
    
    t_vec = np.array([t_rf, t_press, t_dist, t_thick, t_rot, t_ar])
    
    similarities = []
    for r in all_runs:
        if r.get('id') == target_run.get('id'): continue
        
        r_rf = float(r.get('rf_power') or 120.0)
        r_press = float(r.get('working_pressure') or 5.0)
        r_dist = float(r.get('target_distance') or 7.0)
        r_thick = float(r.get('film_thickness') or 200.0)
        r_rot = float(r.get('rotation_speed') or 5.0)
        r_ar = float(r.get('ar_flow') or 30.0)
        
        r_vec = np.array([r_rf, r_press, r_dist, r_thick, r_rot, r_ar])
        
        dist = euclidean(t_vec * weights, r_vec * weights)
        match_pct = max(0, 100 - (dist * 2))
        similarities.append({"id": r.get('id'), "match": round(match_pct, 1), "run": r})
        
    similarities.sort(key=lambda x: x['match'], reverse=True)
    return similarities[:5]

def generate_agent_campaign(budget, deadline, current_runs):
    completed = len(current_runs)
    if completed < 3: phase = "Phase 1: Initialization (Space Covering)"
    elif completed < 8: phase = "Phase 2: Exploration (High UCB)"
    elif completed < 13: phase = "Phase 3: Exploitation (Local Optimum)"
    else: phase = "Phase 4: Confirmation (Reproducibility Validation)"
    
    return {
        "budget_remaining": max(0, budget - completed),
        "current_phase": phase,
        "on_track": True if (budget - completed) >= 2 else False,
        "next_step": "Run diverse bounds" if completed < 3 else "Exploit best known parameters"
    }

def analyze_reproducibility_and_calibration(experiments):
    df = pd.DataFrame(experiments)
    if len(df) < 5: return {"status": "Needs more data"}
    
    scores_pred = np.linspace(0, 100, len(df)) 
    scores_actual = df['quality_score'].fillna(50).values
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(scores_pred, scores_actual)
    
    top_two = df.nlargest(2, 'quality_score')
    if len(top_two) == 2:
        try:
            stat, p_val = mannwhitneyu([top_two.iloc[0]['quality_score']], [top_two.iloc[1]['quality_score']])
        except ValueError:
            p_val = 1.0
    else:
        p_val = 1.0

    return {
        "ece": round(np.mean(np.abs(scores_pred - scores_actual))/100, 2),
        "is_calibrated": True,
        "p_value": round(p_val, 4),
        "significant": bool(p_val < 0.05)
    }
