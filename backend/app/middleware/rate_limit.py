import time

import redis.asyncio as redis
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limiter keyed by client IP. Good enough for a first pass;
    swap for a token-bucket/sliding-window algorithm at higher scale."""

    def __init__(self, app):
        super().__init__(app)
        self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        window = int(time.time() // 60)
        key = f"ratelimit:{client_ip}:{window}"

        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, 60)

        if count > settings.RATE_LIMIT_PER_MINUTE:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again shortly."})

        return await call_next(request)
