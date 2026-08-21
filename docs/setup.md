# Setup

## Local Windows setup

Use Python 3.12 and Node.js 24. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
pytest -q
```

Start the backend:

```powershell
python scripts\start_demo.py
```

Start the frontend in a second terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

The frontend expects `NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1`. If the frontend origin changes, add it to `XRL_HVAC_CORS_ORIGINS` as a comma-separated value.

## Docker

```bash
docker compose up --build
```

The Compose demo is CPU-first and exposes frontend port 3000 and API port 8000. GPU access is unnecessary for inference.

## Recreate XAI artifacts

```powershell
python scripts\generate_xai_artifacts.py
```

The script refuses to run if the frozen checkpoint hash differs from the manifest. Full trajectory JSON/CSV is written under `outputs/trajectories/xai`; the compact validation report is retained for the API and portfolio evidence.

## Training and evaluation

Full training is separate from demo startup. Configuration lives in `configs/evaluation.yaml`; outputs go to `outputs/metrics/step5` and best checkpoints to `models`. The intended laptop profile is RTX 4050 6 GB / 16 GB RAM, with small networks and no broad hyperparameter sweep.
