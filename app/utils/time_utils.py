"""Application time helpers.

Persist and display business timestamps in China Standard Time regardless of
the host or container timezone.
"""

from datetime import datetime
from zoneinfo import ZoneInfo


CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_china() -> datetime:
    """Return the current timezone-aware China Standard Time."""
    return datetime.now(CHINA_TIMEZONE)

