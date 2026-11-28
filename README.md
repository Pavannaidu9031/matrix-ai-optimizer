# MatrixAI: Bayesian Optimization for Thin-Film Sputtering

An open-source web tool that helps researchers choose the next sputtering experiment, so they reach a good thin-film recipe in fewer runs.

**Live app:** https://rock-ai-optimizer.onrender.com

> **Status: early-stage.** The software works end to end, but it has not yet been validated against a full set of real sputtering runs. Validation is the next milestone. Feedback and issues are welcome.

## The problem

Sputtering recipes (power, Ar flow, pressure, thickness, substrate rotation) are usually tuned by trial and error. Each run costs hours of lab time and expensive target material. General-purpose optimizers exist, but none are packaged for a sputtering workflow, where an experimenter needs to log runs, learn from literature, and get the next recommended recipe without writing code. Small university labs rarely have commercial design-of-experiments software or a dedicated data scientist.

## What it does

- **Experiment suggestions:** a Gaussian process surrogate model with a UCB acquisition function recommends the next parameters to try.
- **Experiment ledger:** log every run, including sputter mode (RF or DC) and, for optical fibers, the measured core diameter.
- **Multiple materials:** WO3, Pd and ITO tracks, plus a WO3+Pd bilayer workflow.
- **Multiple substrates:** Si wafer, glass thin film and optical fiber.
- **What-if simulator and phase map:** explore parameter space before spending lab time.
- **Literature parser:** extract process parameters from research-paper PDFs.

## Tech stack

Python, FastAPI, scikit-learn, SQLAlchemy, PostgreSQL (Supabase), deployed on Render.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Web application and routes |
| `optimizer.py` | Optimization logic |
| `matrix_ml_engine.py` | Machine-learning engine |
| `advanced_features.py` | Additional features |
| `database.py` | Database access |
| `templates/`, `static/` | Front-end pages and assets |

## Run locally

```bash
git clone https://github.com/Pavannaidu9031/matrix-ai-optimizer.git
cd matrix-ai-optimizer
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set these environment variables **[confirm the exact names your code uses]**:

```bash
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/DBNAME"
export SECRET_KEY="a-long-random-string"
# Windows (PowerShell): $env:DATABASE_URL="..."; $env:SECRET_KEY="..."
```

Start the app **[confirm the command, e.g. check your Procfile]**:

```bash
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000.

## Deployment

The included `Procfile` runs the app with `uvicorn app:app --host 0.0.0.0 --port $PORT`, which works on Render and similar platforms. Set the same environment variables in your host's dashboard.

## Roadmap

- [ ] Automated tests and CI
- [ ] Validation against real sputtering experiments, with results published openly
- [ ] Bulk CSV import
- [ ] Wafer-to-fiber thickness conversion, verified experimentally
- [ ] Setup guide, methodology notes and example datasets
- [ ] Generalize toward other thin-film processes

## Contributing

Issues and pull requests are welcome. If you run sputtering experiments and want to try the tool, please open an issue with your feedback.

## Author

Built by Pavan Kalyan, MS student in Electrical Engineering at Southern Taiwan University of Science and Technology (STUST), as part of research on WO3+Pd hydrogen sensors on optical fiber.

## License

MIT. See [LICENSE](LICENSE).
