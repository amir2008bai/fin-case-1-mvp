FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
USER 65534:65534

FROM base AS agent
COPY audit_env/__init__.py audit_env/agents.py audit_env/domain.py audit_env/contracts.py audit_env/micro.py audit_env/rag.py /app/audit_env/
ENTRYPOINT ["python", "-m", "audit_env.agents"]
CMD ["reconcile"]

FROM base AS verify
COPY audit_env/*.py audit_env/
COPY audit_env/scenarios/*.py audit_env/scenarios/
COPY tests/*.py tests/
COPY configs/*.json configs/
CMD ["sh", "-ec", "python -m unittest discover -s tests -v && python -m audit_env benchmark"]

FROM agent AS micro-agent
COPY build/micro/model.json /app/model.json
CMD ["micro"]
