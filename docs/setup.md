# Setup

## Local Windows setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pytest -q
```

Start FastAPI from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\start_demo.py
```

Start Next.js in a second terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm.cmd install
npm.cmd run dev
```

Use `npm.cmd` if PowerShell blocks `npm.ps1` under its execution policy. The frontend expects `NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1`; it derives the V2 API origin from the same setting.

Open:

- Dashboard: `http://localhost:3000`
- OpenAPI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Troubleshooting `No module named 'api'`

Run the script from the repository root with the virtual-environment interpreter:

```powershell
Set-Location "C:\path\to\HVAC_control"
.\.venv\Scripts\python.exe scripts\start_demo.py
```

Do not call a globally installed `uvicorn` executable. The startup script adds the repository root to `sys.path` before Uvicorn imports `api.main`.

## Docker

```powershell
docker compose up --build
```

Compose runs CPU inference and waits for the API health check before starting the frontend. GPU access is unnecessary for the dashboard.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
Set-Location frontend
npm.cmd run typecheck
npm.cmd run build
```

## Reproducibility and evidence

V1 checkpoint integrity is declared in `models/demo_manifest.json`. V2 checkpoint hashes and development metrics are in `outputs/v2/training`; the held-out seal receipt is `outputs/v2/protocol/held_out_status.json`.

Training is intentionally separate from runtime startup. V2 scripts live in `scripts/v2`, write only to `models/v2` and `outputs/v2`, and use the locked configuration in `configs/v2`.
