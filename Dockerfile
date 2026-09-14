# The discovery image: the cheap tiers of #148, and nothing else.
#
# It carries no model and no GPU runtime on purpose. `harness/inspect.py`
# imports no model and calls no gateway -- it clones source, reads it, and asks
# the registry for weight sizes -- which is exactly why discovery is the first
# thing worth putting in a cluster.
#
# `git` is a RUNTIME dependency, not a build one: inspect shells out to
# `git clone --depth 1` for every candidate it reads.
FROM python:3.12-slim AS base

# uv from its own published image rather than curl-to-shell: a version is
# pinned, and a vendor installer that dies quietly without a TTY is a trap this
# project has already paid for (RULE #142).
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# Dependencies first, so editing a source file does not re-resolve the world.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --group cluster --no-install-project

COPY harness/ ./harness/
COPY evals/ ./evals/
COPY gateway/ ./gateway/
RUN uv sync --locked --no-dev --group cluster

# A writable home for anything that caches. The chart mounts a volume over the
# parts that must survive a pod, notably the GitHub response cache, which is
# deliberately permanent because a failed fetch falls back to the stale copy.
ENV LOCALHARNESS_HOME=/var/lib/localharness \
    PATH="/app/.venv/bin:$PATH"
RUN mkdir -p /var/lib/localharness && chmod 777 /var/lib/localharness

# Run as nobody: this pod clones arbitrary third-party source and reads it.
USER 65534:65534

ENTRYPOINT ["lh"]
CMD ["--help"]
