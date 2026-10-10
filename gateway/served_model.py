"""Name the model that answered in the gateway's reply headers, not the alias asked for. See README, #670.

Loaded by LiteLLM from the gateway config (`litellm_settings.callbacks`). Headers, because
LiteLLM 1.100.0 restamps the body's `model` to the requested name after every hook.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import models  # noqa: E402

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # the test suite has no litellm
    CustomLogger = object


def resolved(data: dict) -> tuple[str, str]:
    """(the real id, the alias asked for or "") for a request; ("", "") when nothing serves it."""
    model = str(data.get("model") or "")
    try:
        # GATEWAY_CONFIG names the config this gateway serves; its served copy wins.
        real = models.resolve(model)
    except Exception:  # noqa: BLE001
        return "", ""
    return real, (model if real and real != model else "")


class ServedModel(CustomLogger):
    async def async_post_call_response_headers_hook(self, data, user_api_key_dict, response,
                                                    request_headers=None, litellm_call_info=None):
        real, alias = resolved(data)
        if not real:
            return None
        headers = {"x-sohot-model": real}
        if alias:
            headers["x-sohot-served-as"] = alias
        return headers


proxy_handler_instance = ServedModel()
