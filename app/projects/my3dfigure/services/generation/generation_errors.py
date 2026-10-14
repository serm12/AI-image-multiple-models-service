"""Preserve the provider's display message separately from diagnostics."""
import json


def upstream_message_from_text(text):
    """Accept JSON responses and the historical locally wrapped JSON errors."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else payload.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return None


class UpstreamGenerationError(ValueError):
    def __init__(self, status_code, text):
        super().__init__(f"aiapiroute HTTP {status_code}: {text[:1000]}")
        # Parse the complete body before limiting the diagnostic preview.
        self.upstream_message = upstream_message_from_text(text)


def get_upstream_message(error):
    seen = set()
    current = error
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        message = getattr(current, "upstream_message", None)
        if isinstance(message, str) and message.strip():
            return message
        current = current.__cause__ or current.__context__
    return upstream_message_from_text(str(error))
