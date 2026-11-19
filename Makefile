PYTHON ?= python3
DOCKER ?= docker
VERIFY_IMAGE ?= fintech-audit:verify
AGENT_IMAGE ?= fintech-audit:agent
MICRO_IMAGE ?= fintech-audit:micro-agent
DATASET ?= build/open-dataset
LLM_DATA ?= build/llm

.PHONY: verify test demo dataset docker-verify docker-agent train micro-demo docker-micro llm-data llm-smoke

verify: test
	$(PYTHON) -m audit_env benchmark

test:
	$(PYTHON) -m unittest discover -s tests -v

demo:
	$(PYTHON) -m audit_env demo

dataset:
	$(PYTHON) -m audit_env dataset "$(DATASET)"

docker-verify:
	$(DOCKER) build --target verify -t "$(VERIFY_IMAGE)" .
	$(DOCKER) run --rm --pull=never --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --cpus=2 --memory=4g --pids-limit=64 --tmpfs /tmp:rw,nosuid,noexec,size=32m "$(VERIFY_IMAGE)"

docker-agent:
	$(DOCKER) build --target agent -t "$(AGENT_IMAGE)" .
	$(PYTHON) -m audit_env demo --closed --agent-image "$(AGENT_IMAGE)"

train:
	$(PYTHON) -m audit_env train

micro-demo: train
	$(PYTHON) -m audit_env demo --policy micro --closed --monitor build/micro/monitor.json --trajectory build/micro/trajectory.jsonl

docker-micro: train
	$(DOCKER) build --target micro-agent -t "$(MICRO_IMAGE)" .
	$(PYTHON) -m audit_env demo --policy micro --closed --agent-image "$(MICRO_IMAGE)" --monitor build/micro/docker-monitor.json --trajectory build/micro/docker-trajectory.jsonl

llm-data:
	$(PYTHON) -m audit_env.llm_data "$(LLM_DATA)"

llm-smoke:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m audit_env.hf_smoke
