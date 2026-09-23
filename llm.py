"""Chat-model client for EGO-Prompt.

Models are addressed as
    mantle:<model-id>   Bedrock OpenAI-compatible endpoint (e.g. mantle:openai.gpt-6-luna)
    openai:<model-id>   OpenAI API (OPENAI_API_KEY)
    claude:<model-id>   Claude on Bedrock via anthropic.AnthropicBedrockMantle (e.g. claude:anthropic.claude-sonnet-5)
    litellm:<model>     any provider supported by LiteLLM, with its own model names and API-key variables
                        (e.g. litellm:gpt-5-mini, litellm:anthropic/claude-sonnet-5, litellm:gemini/gemini-2.5-flash;
                        Bedrock models not served by the Mantle endpoint: litellm:bedrock/us.anthropic.claude-sonnet-4-6)

Mantle authenticates with a short-term Bedrock API key minted from IAM credentials
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in the environment, or passed explicitly), unless a static
Bedrock API key is set in AWS_BEARER_TOKEN_BEDROCK;
the key is valid for 12 h and is re-minted every 10 h.

Every call is recorded in a thread-safe usage ledger (calls, input / output / reasoning
tokens, per role) and, unless disabled, in a content-addressed sqlite cache: the same
(model, effort, system, user) returns the stored text without an API call. Within one
run this makes re-evaluating an unchanged prompt free and lets a system-prompt candidate
reuse the causal descriptions of the incumbent causal prompt.
"""

import hashlib
import re
import json
import os
import random
import sqlite3
import threading
import time
from collections import defaultdict

MANTLE_BASE = "https://bedrock-mantle.{region}.api.aws/openai/v1"


class Usage:
    """Token ledger keyed by role ('forward_causal', 'forward_predict', 'backward_grad', ...)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.by_role = defaultdict(lambda: defaultdict(int))

    def add(self, role, model, calls=0, cached=0, inp=0, out=0, reasoning=0):
        with self.lock:
            r = self.by_role[role]
            r["model"] = model
            r["calls"] += calls
            r["cache_hits"] += cached
            r["input_tokens"] += inp
            r["output_tokens"] += out
            r["reasoning_tokens"] += reasoning

    def snapshot(self):
        with self.lock:
            return {k: dict(v) for k, v in self.by_role.items()}

    def cost(self, prices):
        """prices: {model_id: (usd_per_M_input, usd_per_M_output)}; reasoning tokens are part of output."""
        total, rows = 0.0, {}
        for role, r in self.snapshot().items():
            p = prices.get(r["model"])
            c = None if p is None else (r["input_tokens"] * p[0] + r["output_tokens"] * p[1]) / 1e6
            rows[role] = c
            total += c or 0.0
        return total, rows


class Cache:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()

    def get(self, k):
        with self.lock:
            row = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return None if row is None else row[0]

    def put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, v))
            self.db.commit()


class _Bearer:
    """Short-term Bedrock API key, shared by all LLM objects of one (key, region)."""
    _lock = threading.Lock()
    _tokens = {}

    @classmethod
    def get(cls, key, secret, region):
        with cls._lock:
            tok, t0 = cls._tokens.get((key, region), (None, 0))
            if tok is None or time.time() - t0 > 10 * 3600:
                from aws_bedrock_token_generator import BedrockTokenGenerator
                from botocore.credentials import Credentials
                tok = BedrockTokenGenerator().get_token(Credentials(key, secret), region)
                cls._tokens[(key, region)] = (tok, time.time())
            return tok


class LLM:
    def __init__(self, spec, effort=None, max_tokens=4000, region="us-east-1",
                 aws_key=None, aws_secret=None, usage=None, cache=None, timeout=180, max_attempts=6):
        self.provider, self.model = spec.split(":", 1) if ":" in spec else ("mantle", spec)
        self.spec = spec
        self.effort = effort                  # reasoning_effort (None = model default)
        self.max_tokens = max_tokens
        self.region = region
        self.aws_key = aws_key or os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret = aws_secret or os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.usage = usage or Usage()
        self.cache = cache
        self.timeout = timeout
        self.max_attempts = max_attempts      # transient-error retries (backoff capped at 60 s per wait)
        self._local = threading.local()

    def _client(self):
        from openai import OpenAI
        if self.provider == "openai":
            if not getattr(self._local, "client", None):
                self._local.client = OpenAI(timeout=self.timeout, max_retries=0)
            return self._local.client
        # a static Bedrock API key (long-term ABSK... or short-term bedrock-api-key-...) takes precedence over IAM keys
        tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or _Bearer.get(self.aws_key, self.aws_secret, self.region)
        if getattr(self._local, "tok", None) != tok:
            self._local.client = OpenAI(api_key=tok, base_url=MANTLE_BASE.format(region=self.region),
                                        timeout=self.timeout, max_retries=0)
            self._local.tok = tok
        return self._local.client

    def _claude(self):
        if not getattr(self._local, "claude", None):
            from anthropic import AnthropicBedrockMantle
            tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            auth = dict(api_key=tok) if tok else dict(aws_access_key=self.aws_key, aws_secret_key=self.aws_secret)
            self._local.claude = AnthropicBedrockMantle(aws_region=self.region, timeout=self.timeout,
                                                        max_retries=0, **auth)
        return self._local.claude

    def _key(self, system, user, salt):
        h = hashlib.sha256()
        for part in (self.spec, str(self.effort), str(self.max_tokens), system, user, str(salt)):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def chat(self, system, user, role="misc", use_cache=True, salt=0, json_mode=False):
        """One chat completion. `salt` makes otherwise identical requests distinct (repeat draws)."""
        k = self._key(system, user, salt) if (self.cache and use_cache) else None
        if k:
            hit = self.cache.get(k)
            if hit is not None:
                self.usage.add(role, self.model, cached=1)
                return hit
        if self.provider == "claude":
            # Claude on Bedrock (Messages API via AnthropicBedrockMantle). effort 'none' = thinking disabled
            # (Sonnet 5 runs adaptive thinking when `thinking` is omitted); sampling params are not sent (400 on 5.x).
            kw = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
            if self.effort in (None, "none"):
                kw["thinking"] = {"type": "disabled"}
            else:
                kw["thinking"] = {"type": "adaptive"}
                kw["output_config"] = {"effort": self.effort}
        elif self.provider == "litellm":
            # LiteLLM maps the OpenAI-style request to the provider; drop_params removes what a provider does not
            # take. effort 'none' is sent only to OpenAI reasoning models (other providers: model default).
            kw = dict(model=self.model, max_tokens=self.max_tokens, drop_params=True, timeout=self.timeout,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            if self.effort and (self.effort != "none" or re.search(r"gpt-5|gpt-6|\bo\d", self.model)):
                kw["reasoning_effort"] = self.effort
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
        else:
            kw = dict(model=self.model, max_completion_tokens=self.max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            if self.effort:
                kw["reasoning_effort"] = self.effort
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
        last = None
        for attempt in range(self.max_attempts):
            try:
                if self.provider == "claude":
                    resp = self._claude().messages.create(**kw)
                    text = "".join(b.text for b in resp.content if b.type == "text")
                    self.usage.add(role, self.model, calls=1, inp=resp.usage.input_tokens,
                                   out=resp.usage.output_tokens)
                    if k and text:
                        self.cache.put(k, text)
                    return text
                if self.provider == "litellm":
                    import litellm
                    resp = litellm.completion(**kw)
                else:
                    resp = self._client().chat.completions.create(**kw)
                text = resp.choices[0].message.content or ""
                u = resp.usage
                reasoning = 0
                if u is not None and getattr(u, "completion_tokens_details", None) is not None:
                    reasoning = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
                self.usage.add(role, self.model, calls=1,
                               inp=getattr(u, "prompt_tokens", 0) or 0,
                               out=getattr(u, "completion_tokens", 0) or 0, reasoning=reasoning)
                if k and text:
                    self.cache.put(k, text)
                return text
            except Exception as e:  # noqa: BLE001 - classify below
                last = e
                msg = str(e).lower()
                if "reasoning_effort" in msg and "unsupported" in msg or "invalid value" in msg and "effort" in msg:
                    kw.pop("reasoning_effort", None)          # model does not take this effort value
                    continue
                if "response_format" in msg:
                    kw.pop("response_format", None)
                    continue
                status = getattr(e, "status_code", None)
                transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or any(
                    s in msg for s in ("throttl", "rate limit", "too many requests", "timed out", "timeout",
                                       "service unavailable", "serviceunavailable", "connection error", "overloaded"))
                if transient:
                    time.sleep(min(60, 2 ** attempt * 2) * (0.5 + random.random()))
                    continue
                raise
        raise last


def extract_json(text):
    """First JSON object in a model response (tolerates ``` fences and leading prose)."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    start = t.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = t[start:i + 1]
                    # The prompts contain tags like <\Causal Description>; models often copy the backslash
                    # into JSON strings unescaped ("\C" is an invalid escape). Without the repair the outer
                    # object fails and the search falls through to an inner object (e.g. one candidate).
                    for s in (chunk, re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", chunk)):
                        try:
                            return json.loads(s, strict=False)
                        except json.JSONDecodeError:
                            pass
                    break
        start = t.find("{", start + 1)
    return None
