#: The node container that runs the cluster. `desktop-control-plane` is Docker
#: Desktop's; a kind cluster is `<name>-control-plane`. Overridable because the
#: name is the one thing that differs between the two.
CLUSTER_NODE ?= desktop-control-plane
IMAGE_TAG ?= 0.1.0
NAMESPACE ?= lh

# Entry points. `make check` is what CI would run and what to run before a commit.
.PHONY: check test test-slow test-network test-postgres coverage image secret-gh metrics lint smoke services clean

check: lint test          ## static checks + unit tests (no services needed)

test:                     ## unit tests (fast; excludes the model-loading ones)
	uv run pytest tests/ -q

test-slow:                ## the metrics tests: loads multi-GB scorers, minutes
	uv run --group metrics pytest tests/ -q -m slow

test-network:             ## ask third parties whether what they publish is still there
	uv run pytest tests/ -q -m network

coverage:                 ## REPORT coverage, never gate on it; then the diff figure
	uv run pytest tests/ -q --cov=harness --cov=evals \
	  --cov-report=term:skip-covered --cov-report=json
	@uv run python -m harness.covdiff || true

secret-gh:                ## put a GitHub token in the cluster, with no trailing newline
	@# `gh auth token` ends in a newline and `--from-file` keeps it, so the
	@# header becomes "Bearer ghp_xxx\n" and every API call dies with
	@# `invalid header field value for "Authorization"`. tr -d is the fix, and
	@# the token never appears in output, in a manifest, or in git.
	@gh auth token | tr -d '\r\n' | kubectl create secret generic localharness-gh \
	  -n $(NAMESPACE) --from-file=token=/dev/stdin --dry-run=client -o yaml \
	  | kubectl apply -f - >/dev/null
	@echo "localharness-gh updated in namespace $(NAMESPACE)"

image:                    ## build the discovery image and load it INTO the cluster
	docker build -t localharness:$(IMAGE_TAG) .
	@# Docker Desktop's Kubernetes does NOT share the docker image store: the
	@# kubelet's containerd uses the k8s.io namespace and a `docker build` is
	@# invisible to it, so a pod goes to docker.io for a name that only exists
	@# here. This is the same step `kind load docker-image` performs.
	docker save localharness:$(IMAGE_TAG) \
	  | docker exec -i $(CLUSTER_NODE) ctr -n k8s.io images import -
	@docker exec $(CLUSTER_NODE) ctr -n k8s.io images ls \
	  | grep -q "localharness:$(IMAGE_TAG)" \
	  && echo "loaded: localharness:$(IMAGE_TAG)" \
	  || { echo "NOT loaded; the pod will try docker.io" >&2; exit 1; }

metrics:                  ## the four DORA numbers, from git and gh
	uv run python -m harness.dora

test-postgres:            ## the store against a REAL database, in docker
	@docker rm -f lh-pg-test >/dev/null 2>&1 || true
	@docker run -d --name lh-pg-test -e POSTGRES_PASSWORD=test \
	  -e POSTGRES_USER=discovery -e POSTGRES_DB=discovery \
	  -p 55432:5432 postgres:16-alpine >/dev/null
	@for i in $$(seq 1 30); do \
	  docker exec lh-pg-test pg_isready -U discovery >/dev/null 2>&1 && break; \
	  sleep 1; done
	LOCALHARNESS_STORE=postgres PGHOST=127.0.0.1 PGPORT=55432 \
	  PGUSER=discovery PGPASSWORD=test PGDATABASE=discovery \
	  uv run --group cluster pytest tests/ -q -m postgres; \
	  status=$$?; docker rm -f lh-pg-test >/dev/null 2>&1; exit $$status

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
