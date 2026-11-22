FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HOME=/tmp PYTHONPATH=/app
WORKDIR /app
USER 65534:65534

# Agent image: public code only (no generator, no evaluator, no answers).
FROM base AS agent
COPY audit_env/__init__.py audit_env/agents.py audit_env/checks.py audit_env/contracts.py audit_env/micro.py /app/audit_env/
COPY models/model.json /app/models/model.json
ENTRYPOINT ["python", "-m", "audit_env.agents"]
CMD ["expert"]

# Verification image: the single command the jury runs, offline.
FROM base AS verify
COPY audit_env/ audit_env/
COPY tests/ tests/
COPY configs/ configs/
COPY data/ data/
COPY schemas/ schemas/
COPY models/ models/
CMD ["python", "-m", "audit_env", "verify", "--report", "/tmp/verify-report.json"]
