"""Refuse a response_format the alias's server cannot enforce. See README, #315.

Loaded by LiteLLM from gateway/config.yaml (`litellm_settings.callbacks`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import gateway, serving  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "config.yaml"
# The served config adds sohot-<lane> aliases; prefer it when serve-gateway.sh wrote it. #297.
if gateway.served_path(CONFIG).exists():
    CONFIG = gateway.served_path(CONFIG)
ENFORCED = ("json_schema", "json_object")

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # the test suite has no litellm
    CustomLogger = object


def _enforcing() -> list[str]:
    return sorted(str(e["model_name"]) for e in gateway.load(CONFIG).get("model_list") or []
                  if not e.get("deprecated_for")
                  and serving.engine_for(str(e["model_name"]), config=CONFIG) == serving.LLAMACPP)


def refusal(data: dict) -> str:
    """Why this request must be refused, or ""."""
    fmt = data.get("response_format")
    if not isinstance(fmt, dict) or fmt.get("type") not in ENFORCED:
        return ""
    model = str(data.get("model") or "")
    engine = serving.engine_for(model, config=CONFIG)
    if engine == serving.LLAMACPP:
        return ""
    return (f"{model} is served by {engine}, which ignores response_format, so "
            f"the reply would not be guaranteed to match the schema. Use an "
            f"alias that enforces it: {', '.join(_enforcing())}")


class SchemaGuard(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        why = refusal(data)
        if why:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=why)
        return data


proxy_handler_instance = SchemaGuard()
