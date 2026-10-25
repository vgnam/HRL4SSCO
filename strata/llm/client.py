"""Chat client for an OpenAI-compatible endpoint, with an on-disk cache.

Ported from ``lexlaw/llm/client.py`` (branch ``paper-config``). L1 uses it to
synthesise law programs and L2 to synthesise operators.

Configuration: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
OPENAI_REASONING_EFFORT, from the environment or from the repository's
``.env``. The key is never printed, logged, cached or committed: the cache key
is a hash of (model, messages, sample) only.

``sample`` lets a caller draw several distinct proposals for the same prompt
(each index is cached separately), which the synthesis loops need; with a
fixed cache the whole pipeline replays deterministically and offline.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE = os.path.join(ROOT, "strata_cache", "llm")


def read_env(path=None):
    path = path or os.path.join(ROOT, ".env")
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class Chat:
    def __init__(self, cache_dir=None, env_path=None, temperature=None, max_retries=4,
                 offline=False, model=None):
        env = dict(read_env(env_path))
        env.update({k: v for k, v in os.environ.items() if k.startswith("OPENAI_")})
        self._key = env.get("OPENAI_API_KEY")
        self.base = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = model or env.get("OPENAI_MODEL", "")
        self.effort = env.get("OPENAI_REASONING_EFFORT")
        self.temperature, self.max_retries, self.offline = temperature, max_retries, offline
        self.cache_dir = cache_dir or DEFAULT_CACHE
        os.makedirs(self.cache_dir, exist_ok=True)
        self.calls = self.tok_in = self.tok_out = self.cache_hits = 0
        # Some gateways (e.g. OpenCode Go) reject generic HTTP-library user agents
        # and route by a stable per-client session id.
        self.session = uuid.uuid4().hex

    def __repr__(self):
        return "Chat(model=%r, base=%r, key=%s)" % (
            self.model, self.base, "set" if self._key else "missing")

    def _cache_path(self, messages, sample):
        payload = [self.model, messages] if sample == 0 else [self.model, messages, sample]
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, h[:2], h + ".json")

    def __call__(self, messages, sample=0):
        """Return the assistant text for ``messages`` (cached per ``sample`` index)."""
        path = self._cache_path(messages, sample)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.cache_hits += 1
                return json.load(f)["content"]
        if self.offline:
            raise RuntimeError("cache miss in offline mode: an API call would be needed")
        if not self._key or not self.model:
            raise RuntimeError("OPENAI_API_KEY / OPENAI_MODEL not configured (environment or .env)")
        body = {"model": self.model, "messages": messages}
        if self.effort:
            body["reasoning_effort"] = self.effort
        if self.temperature is not None:
            body["temperature"] = self.temperature
        req = urllib.request.Request(
            self.base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key,
                     "User-Agent": "strata/0.1", "x-opencode-session": self.session})
        data = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    detail = type(e).__name__
                    if isinstance(e, urllib.error.HTTPError):
                        detail += " %s: %s" % (e.code, e.read()[:300].decode("utf-8", "replace"))
                    raise RuntimeError("chat request failed: %s" % detail) from None
                time.sleep(2 ** attempt * 5)
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        self.calls += 1
        self.tok_in += usage.get("prompt_tokens", 0)
        self.tok_out += usage.get("completion_tokens", 0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"content": content, "usage": usage}, f)
        return content

    def usage(self):
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "tokens_in": self.tok_in, "tokens_out": self.tok_out}
