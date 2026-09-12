# Entry points. `make check` is what CI would run and what to run before a commit.
.PHONY: check test test-slow lint smoke services clean

check: lint test          ## static checks + unit tests (no services needed)

test:                     ## unit tests (fast; excludes the model-loading ones)
	uv run pytest tests/ -q

test-slow:                ## the metrics tests: loads multi-GB scorers, minutes
	uv run --group metrics pytest tests/ -q -m slow

lint:                     ## shellcheck every script, syntax-check every one
	shellcheck -S warning scripts/*.sh
	@for f in scripts/*.sh; do bash -n "$$f" || exit 1; done
	uv run python -m harness.privacy
	@echo "lint ok"

smoke:                    ## end-to-end; REQUIRES the services to be running
	./scripts/smoke.sh

services:                 ## install launchd units so the services survive a reboot
	./scripts/launchd.sh install

evals:                    ## compare candidates; REQUIRES the services running
	uv run python -m evals.run --modality $(MODALITY) \
	  --candidates $(CANDIDATES) --out .logs/evals-$$(date +%Y%m%d-%H%M%S)

MODALITY   ?= all
CANDIDATES ?= local-mid

clean:
	rm -f tools/h3probe
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
