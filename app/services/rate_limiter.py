import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    window_seconds: int
    remaining: int
    retry_after_seconds: int


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> RateLimitResult:
        now = time.monotonic()
        cutoff = now - self.window_seconds

        async with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - hits[0])) + 1)
                return RateLimitResult(
                    allowed=False,
                    limit=self.limit,
                    window_seconds=self.window_seconds,
                    remaining=0,
                    retry_after_seconds=retry_after,
                )

            hits.append(now)
            remaining = max(0, self.limit - len(hits))
            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                window_seconds=self.window_seconds,
                remaining=remaining,
                retry_after_seconds=0,
            )


generate_request_limiter = SlidingWindowRateLimiter(limit=6, window_seconds=600)
