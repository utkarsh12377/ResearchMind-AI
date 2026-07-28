"""Rate limiting for brute-force-sensitive endpoints (login, register).

Uses in-process memory storage, which is sufficient for a single backend
instance. Scaling to multiple instances only requires pointing `storage_uri`
at the Redis already declared in docker-compose (settings.redis_url) — the
`limits`/slowapi storage backend is swappable without touching call sites.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
