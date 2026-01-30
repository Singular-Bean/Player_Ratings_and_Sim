import os
import json
import time
import random
import hashlib
import requests
from pathlib import Path
from threading import Lock

# --------------------------------------------------------------------
# Configuration (edit these values if needed)
# --------------------------------------------------------------------

CACHE_DIR = Path("./cache")        # where cached responses are stored
CACHE_TTL = 60 * 60 * 6            # how long to keep cached data (6h)
REQUESTS_PER_SECOND = 1          # throttle speed (1 req per second)
MAX_RETRIES = 5                    # max retry attempts
BACKOFF_BASE = 2.0                 # exponential backoff base
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0 Safari/537.36"
)

# --------------------------------------------------------------------

_lock = Lock()
_last_request_time = 0

def _hash_url(url: str) -> str:
    cleaned = url.replace("https://", "").replace("http://", "")
    safe = cleaned.replace("/", "_").replace("?", "_").replace("&", "_").replace(":", "_").replace("=", "_")
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{h}"


def _cache_path(url: str) -> Path:
    """Return the path of the cache file for a URL."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{_hash_url(url)}.json"


def _load_from_cache(url: str, cache_ttl: float):
    """Try to load a cached response if it exists and is fresh."""
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            age = time.time() - cached["timestamp"]
            # If cache_ttl is 0, treat it as "cache forever"
            if cache_ttl == 0 or age < cache_ttl:
                return cached["status"], cached["headers"], cached["text"]
    except Exception:
        return None
    return None


def _save_to_cache(url: str, status, headers, text):
    """Save response data to the cache."""
    path = _cache_path(url)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "status": status,
                    "headers": headers,
                    "text": text,
                },
                f,
            )
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[safe_requests] Cache save failed for {url}: {e}")


def _rate_limit():
    """Ensure we don't exceed REQUESTS_PER_SECOND."""
    global _last_request_time
    with _lock:
        now = time.time()
        elapsed = now - _last_request_time
        delay_needed = max(0, (1.0 / REQUESTS_PER_SECOND) - elapsed)
        if delay_needed > 0:
            # jitter: +/- up to 30%
            time.sleep(delay_needed + random.uniform(0, delay_needed * 0.3))
        _last_request_time = time.time()


def safe_request(url: str, cache_ttl: float = CACHE_TTL, defaultSession=None, **kwargs):
    """
    Get a URL safely (with caching, rate limiting, and retries).
    Returns a requests.Response object.

    Parameters:
        url : str
            The request URL.
        cache_ttl : float
            Cache time in seconds. Use 0 for "cache forever".
        **kwargs :
            Passed directly to requests.get().
    """
    # Step 1: check cache
    cached = _load_from_cache(url, cache_ttl)
    if cached:
        status, headers, text = cached
        resp = requests.Response()
        resp.status_code = status
        resp._content = text.encode("utf-8")
        resp.headers = headers
        resp.url = url
        resp._from_cache = True
        print(f"[safe_requests] Cache hit: {url}")
        return resp

    # Step 2: polite live request with retries
    session = requests.Session() if defaultSession is None else defaultSession
    session.headers.update({"User-Agent": USER_AGENT})
    attempt = 0

    while attempt < MAX_RETRIES:
        _rate_limit()
        try:
            resp = session.get(url, **kwargs)
            if resp.status_code == 200:
                _save_to_cache(url, resp.status_code, dict(resp.headers), resp.text)
                print(f"[safe_requests] Fetched OK: {url}")
                return resp
            elif resp.status_code in (403, 429) or 500 <= resp.status_code < 600:
                delay = BACKOFF_BASE ** attempt + random.uniform(0, 1)
                print(
                    f"[safe_requests] Got {resp.status_code}, retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                attempt += 1
            else:
                print(f"[safe_requests] Non-retryable {resp.status_code} for {url}")
                return resp
        except requests.RequestException as e:
            delay = BACKOFF_BASE ** attempt + random.uniform(0, 1)
            print(f"[safe_requests] Exception {e}, retrying in {delay:.1f}s...")
            time.sleep(delay)
            attempt += 1
    raise RuntimeError(f"[safe_requests] Failed after {MAX_RETRIES} attempts: {url}")

