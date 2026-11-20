PYTHON ?= python
DOCKER ?= docker
VERIFY_IMAGE ?= fintech-audit:verify
AGENT_IMAGE ?= fintech-audit:agent

.PHONY: verify test demo dataset benchmark docker-verify docker-agent train

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

train:
	$(PYTHON) -m audit_env train

docker-verify:
	$(DOCKER) build --target verify -t "$(VERIFY_IMAGE)" .
	$(DOCKER) run --rm --pull=never --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --cpus=2 --memory=4g --pids-limit=64 --tmpfs /tmp:rw,nosuid,size=256m "$(VERIFY_IMAGE)"

docker-agent:
	$(DOCKER) build --target agent -t "$(AGENT_IMAGE)" .
	$(PYTHON) -m audit_env demo --closed --policy expert
