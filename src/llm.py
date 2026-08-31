"""Optional local-LLM fallback for turn understanding.

The rule parser is more accurate than the LLM on every phrasing it covers
(100% vs 96.7% category recognition across four rewrite levels). It only fails
on wordings outside its pattern list, where the LLM recovers 93.3%. So the LLM
is a *fallback*, never a replacement: it is consulted only when the rules
extract no category at all.

Everything here degrades to a no-op. If the service is absent, times out, or
answers with anything unparseable, the agent keeps the rule-based result. The
scored offline path therefore never depends on this module being available.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

#: Opt-in. Unset or empty disables the LLM path entirely.
ENDPOINT_ENV = "TECHJAM_LLM_ENDPOINT"
MODEL_ENV = "TECHJAM_LLM_MODEL"
DEFAULT_MODEL = "qwen3:latest"

PROBE_TIMEOUT = 2.0
CALL_TIMEOUT = 15.0
MAX_MESSAGE_CHARS = 400

PROMPT = (
    "Extract the product category the shopper wants, and their stated requirements.\n"
    "Output ONLY JSON, no explanation.\n"
    'Message: "{message}"\n'
    'Format: {{"category": "...", "constraints": ["..."]}}'
)


class LocalLLM:
    """Ollama-compatible generate endpoint. Disabled unless explicitly configured."""

    def __init__(self, endpoint: str | None = None, model: str | None = None) -> None:
        self.endpoint = endpoint or os.environ.get(ENDPOINT_ENV) or ""
        self.model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        self.available = bool(self.endpoint) and self._probe()
        self.calls = 0
        self.failures = 0
        self.seconds = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _probe(self) -> bool:
        try:
            request = urllib.request.Request(self.endpoint.replace("/api/generate", "/api/tags"))
            with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT):
                return True
        except Exception:
            return False

    def extract_category(self, message: str) -> str | None:
        """Best-effort category extraction; returns None on any problem."""
        if not self.available or not message.strip():
            return None
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": PROMPT.format(message=message[:MAX_MESSAGE_CHARS]),
                "stream": False,
                # Reasoning mode is both slower and less accurate here: 25.1 s and
                # a wrong answer, against 0.9 s and a right one with it off.
                "think": False,
                "options": {"temperature": 0},
            }
        ).encode()
        started = time.time()
        try:
            request = urllib.request.Request(
                self.endpoint, payload, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=CALL_TIMEOUT) as response:
                data = json.load(response)
        except Exception:
            self.failures += 1
            self.seconds += time.time() - started
            return None
        self.calls += 1
        self.seconds += time.time() - started
        self.prompt_tokens += int(data.get("prompt_eval_count") or 0)
        self.completion_tokens += int(data.get("eval_count") or 0)
        match = re.search(r"\{.*\}", str(data.get("response", "")), re.S)
        if not match:
            return None
        try:
            category = json.loads(match.group(0)).get("category")
        except Exception:
            return None
        if isinstance(category, str) and category.strip():
            return category.strip()[:120]
        return None

    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "seconds": round(self.seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
