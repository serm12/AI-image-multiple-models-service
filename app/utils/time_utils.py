"""Application time helpers.

Persist new business timestamps in UTC. Keep China time helpers for legacy
records and convert timestamps only at display/filter boundaries.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")
CHINA_TIMEZONE_NAME = "Asia/Shanghai"


def now_china() -> datetime:
    """Return the current timezone-aware China Standard Time."""
    return datetime.now(CHINA_TIMEZONE)


def now_utc() -> datetime:
    """Canonical storage clock; display timezones are applied when reading."""
    return datetime.now(timezone.utc)
