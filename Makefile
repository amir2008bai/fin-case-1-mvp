PYTHON ?= python
DOCKER ?= docker
VERIFY_IMAGE ?= fintech-audit:verify
AGENT_IMAGE ?= fintech-audit:agent

.PHONY: verify test demo dataset benchmark failures train serve docker-verify docker-agent

verify:
	$(PYTHON) -m audit_env verify

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

demo:
	$(PYTHON) -m audit_env demo --policy expert

dataset:
	$(PYTHON) -m audit_env dataset build/open-dataset

benchmark:
	$(PYTHON) -m audit_env benchmark --holdout

failures:
	$(PYTHON) -m audit_env failures

train:
	$(PYTHON) -m audit_env train

serve:
	$(PYTHON) -m audit_env serve

# The jury's command: offline verification in a read-only container with 2 CPU / 4 GB.
docker-verify:
	$(DOCKER) build --target verify -t "$(VERIFY_IMAGE)" .
	$(DOCKER) run --rm --pull=never --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --cpus=2 --memory=4g --pids-limit=64 --tmpfs /tmp:rw,nosuid,size=256m "$(VERIFY_IMAGE)"

# Closed episodes where the agent runs in its own container (no network, read-only, no answers).
docker-agent:
	$(DOCKER) build --target agent -t "$(AGENT_IMAGE)" .
	$(PYTHON) -m audit_env demo --closed --policy expert --agent-image "$(AGENT_IMAGE)"
	$(PYTHON) -m audit_env demo --closed --policy micro --agent-image "$(AGENT_IMAGE)"
