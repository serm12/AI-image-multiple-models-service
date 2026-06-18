from functools import lru_cache

from app.core.config import ArtStyleEnum
from app.core.helpers import get_style_description


@lru_cache(maxsize=1)
def build_art_styles_response() -> dict:
    """Build the static art style payload used by the /styles endpoint."""
    styles = []
    seen_values = set()

    for style in ArtStyleEnum:
        style_value = style.value
        if style_value in seen_values:
            continue

        seen_values.add(style_value)
        styles.append(
            {
                "value": style_value,
                "name": style_value.replace("_", " ").title(),
                "description": get_style_description(style_value),
            }
        )

    def sort_key(style):
        value = style["value"]
        if value.startswith("flux_"):
            return (0, value)
        if value.startswith("gemini_"):
            return (1, value)
        if value.startswith("seedream-4_"):
            return (2, value)
        return (3, value)

    styles.sort(key=sort_key)
    return {"styles": styles}
