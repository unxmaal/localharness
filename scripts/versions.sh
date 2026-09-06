# shellcheck shell=bash
# shellcheck disable=SC2034  # every pin is consumed by a script that sources this
# Pinned service dependencies. Sourced by the launchers; never run directly.
#
# `uv run --with litellm[proxy]` resolves the LATEST release on every launch, so
# a restart silently changes 107 packages. That is the opposite of what this
# machine is for: a measurement from last week and one from today would have
# been produced by different software, and nothing would say so. The gateway
# restart in this session downloaded 22.5 MiB and reinstalled 107 packages
# before it could serve.
#
# These are the versions the suite was last verified against.
#
# TO BUMP ONE: change it here, restart the service, and run ./scripts/smoke.sh.
# It exits non-zero if the architecture's assumptions broke, which is exactly
# what an upgrade is most likely to do -- the /v1/messages routing regression
# this repo exists to remember came from a LiteLLM release.

LITELLM_PIN="litellm[proxy]==1.100.0"

MLX_AUDIO_PIN="mlx-audio==0.5.1"
MISAKI_PIN="misaki[en]==0.9.4"
WEBRTCVAD_PIN="webrtcvad==2.0.10"
FASTAPI_PIN="fastapi==0.141.1"
UVICORN_PIN="uvicorn==0.52.4"
MULTIPART_PIN="python-multipart==0.0.32"
# RULE #143, and a range on purpose. webrtcvad still imports pkg_resources, uv
# does not install setuptools into venvs on Python 3.12+, and setuptools >= 81
# removed pkg_resources outright, so unpinned resolves to 84.x and mlx_audio
# dies on import. Any 70-80 works; nothing above 81 does.
SETUPTOOLS_PIN="setuptools>=70,<81"
