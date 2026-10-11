"""Chat client for an OpenAI-compatible endpoint, with an on-disk cache (method Section 10.1).

Configuration: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_REASONING_EFFORT, from the
environment or from a .env file. The key is never printed, logged, cached or committed: the cache
key is a hash of (model, messages) only.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    def __init__(self, cache_dir=None, env_path=None, temperature=None, max_retries=4, offline=False):
        env = {**read_env(env_path), **{k: v for k, v in os.environ.items() if k.startswith("OPENAI_")}}
        self._key = env.get("OPENAI_API_KEY")
        self.base = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = env.get("OPENAI_MODEL", "")
        self.effort = env.get("OPENAI_REASONING_EFFORT")
        self.temperature, self.max_retries, self.offline = temperature, max_retries, offline
        self.cache_dir = cache_dir or os.path.join(ROOT, "lexlaw_cache", "llm")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.calls = self.tok_in = self.tok_out = self.cache_hits = 0

    def __repr__(self):
        return f"Chat(model={self.model!r}, base={self.base!r}, key={'set' if self._key else 'missing'})"

    def _cache_path(self, messages):
        h = hashlib.sha256(json.dumps([self.model, messages], sort_keys=True).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, h[:2], h + ".json")

    def __call__(self, messages):
        path = self._cache_path(messages)
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
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"chat request failed: {type(e).__name__}") from None
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
