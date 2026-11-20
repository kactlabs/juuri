"""LLM backends. Each backend is any object with `complete(prompt) -> str`."""

import json
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    """Raised when an LLM backend call fails."""


class OllamaLLM:
    """Calls a local Ollama server (https://ollama.com) over HTTP."""

    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 120.0) -> None:
        if not model:
            raise ValueError("model is required")
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama request to {self.host} failed: {exc}") from exc
        if "response" not in payload:
            raise LLMError(f"Ollama returned no 'response' field: {payload.get('error', payload)}")
        return payload["response"]
