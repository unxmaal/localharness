# Entry points. `make check` is what CI would run and what to run before a commit.
.PHONY: check test lint smoke clean

check: lint test          ## static checks + unit tests (no services needed)

test:                     ## unit tests
	uv run pytest tests/ -q

lint:                     ## shellcheck every script, syntax-check every one
	shellcheck -S warning scripts/*.sh
	@for f in scripts/*.sh; do bash -n "$$f" || exit 1; done
	@echo "lint ok"

smoke:                    ## end-to-end; REQUIRES the services to be running
	./scripts/smoke.sh

evals:                    ## compare candidates; REQUIRES the gateway running
	uv run python -m evals.run --modality all \
	  --candidates $(CANDIDATES) --out .logs/evals-$$(date +%Y%m%d-%H%M%S)

CANDIDATES ?= local-mid,local-summarize

clean:
	rm -f tools/h3probe
	rm -rf .pytest_cache voice/__pycache__ tests/__pycache__
