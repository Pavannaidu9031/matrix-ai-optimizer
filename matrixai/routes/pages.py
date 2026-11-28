"""HTML pages."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..science import advanced_features, costs, optimizer
from ..web import render, settings, store

router = APIRouter()
RUN_BUDGET = 25          # planned runs for the campaign; drives "runs remaining"


def dashboard_stats(experiments: list) -> dict:
    intensities = [float(e["light_intensity_uw"]) for e in experiments if e.get("light_intensity_uw") is not None]
    return {
        "total_runs": len(experiments),
        "monoclinic_runs": sum(optimizer.normalize_xrd_phase(e.get("xrd_phase")) == "Monoclinic" for e in experiments),
        "best_shift": round(max(intensities), 1) if intensities else 0.0,
        "runs_remaining": max(0, RUN_BUDGET - len(experiments)),
    }


@router.get("/")
def index(request: Request, branch: str = "main"):
    user = request.session.get("user")
    if not user:
        return render(request, "index.html", experiments=[], stats={}, branches=[], current_branch="main")
    # "/" is public (it hosts sign-in and the pending page), so check here.
    if not request.app.state.approvals.is_approved(user.get("email")):
        return render(request, "pending.html")

    email = user["email"]
    try:
        experiments = store(request).experiments(email, branch)
        branches = store(request).branches(email)
    except Exception as err:
        print(f"Dashboard could not load runs: {err}")
        experiments, branches = [], ["main"]
    return render(
        request, "index.html",
        experiments=experiments, stats=dashboard_stats(experiments), branches=branches, current_branch=branch,
        failure_analysis=advanced_features.analyze_failure(experiments, experiments[0]) if experiments else None,
    )


@router.get("/experiments-view")
def experiments_page(request: Request, branch: str = "main"):
    runs = store(request).experiments(request.session["user"]["email"], branch)
    return render(request, "experiments.html", experiments=runs, current_branch=branch)


@router.get("/optimizer-view")
def optimizer_page(request: Request, branch: str = "main"):
    return render(request, "optimizer.html", current_branch=branch)


@router.get("/bulk-import")
def bulk_import_page(request: Request, branch: str = "main"):
    return render(request, "bulk_import.html", current_branch=branch)


@router.get("/cost-tracker")
def cost_tracker_page(request: Request, branch: str = "main"):
    runs = store(request).experiments(request.session["user"]["email"], branch)
    return render(request, "cost_tracker.html", current_branch=branch, stats=costs.resource_burn(runs))


@router.get("/publication")
def publication_page(request: Request):
    runs = store(request).experiments(request.session["user"]["email"])
    return render(request, "publication.html", readiness=advanced_features.check_publication_readiness(runs))


@router.get("/supervisor")
def supervisor_page(request: Request):
    """Read-only lab overview: the founder sees the whole lab, everyone else
    only their own runs."""
    email = request.session["user"]["email"]
    if settings(request).is_founder(email):
        runs = store(request).all_experiments(limit=50)
    else:
        runs = store(request).experiments(email, limit=50)
    return render(request, "supervisor.html", experiments=runs)


@router.get("/thesis")
def thesis_page(request: Request):
    return render(request, "thesis.html")


@router.get("/scheduler")
def scheduler_page(request: Request):
    return render(request, "scheduler.html")


@router.get("/agent")
def agent_page(request: Request):
    return render(request, "agent.html")


@router.get("/api/docs")
def api_docs_page(request: Request):
    return render(request, "apidocs.html")


@router.get("/healthz")
def healthz(request: Request):
    try:
        version = store(request).schema_version()
        return {"ok": True, "schema": version}
    except Exception as err:
        return JSONResponse({"ok": False, "error": str(err)[:200]}, status_code=503)
