FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XRL_HVAC_DEVICE=cpu

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY api api
COPY configs configs
COPY models/demo_manifest.json models/demo_manifest.json
COPY models/dqn/demo_best.pt models/dqn/demo_best.pt
COPY outputs/metrics/step5/benchmark_report.json outputs/metrics/step5/benchmark_report.json
COPY outputs/trajectories/xai/step6_xai_report.json outputs/trajectories/xai/step6_xai_report.json
COPY src src

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
