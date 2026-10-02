"""LLM backends for the translator around the simulation.

Three interchangeable backends:
  * claude-cli  - the local `claude` CLI in headless mode (uses the user's Claude Code login, no API key)
  * anthropic   - the Anthropic Python SDK (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
  * ollama      - a local Ollama server (e.g. a small Qwen), for a fully offline pet
Selection: FLYPET_LLM env var, else auto (anthropic if a key is set, else claude-cli).
"""

from __future__ import annotations
import json, os, re, shutil, subprocess, urllib.request


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return m.group(1) if m else text


class LLM:
    name = "base"

    def complete(
        self, system: str, prompt: str, schema: dict | None = None, max_tokens: int = 2000
    ) -> str | dict:
        raise NotImplementedError


class ClaudeCLI(LLM):
    name = "claude-cli"

    def __init__(self, model: str | None = None, extra_flags: list[str] | None = None):
        self.model = model or os.environ.get("FLYPET_CLAUDE_MODEL", "sonnet")
        self.flags = (
            extra_flags
            if extra_flags is not None
            else os.environ.get(
                "FLYPET_CLAUDE_FLAGS",
                "--tools= --setting-sources= --exclude-dynamic-system-prompt-sections --no-session-persistence",
            ).split()
        )
        if not shutil.which("claude"):
            raise RuntimeError("claude CLI not found in PATH")

    def complete(self, system, prompt, schema=None, max_tokens=2000):
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        cmd = [
            "claude",
            "-p",
            prompt,
            "--system-prompt",
            system,
            "--model",
            self.model,
            "--output-format",
            "json",
        ]
        for f in self.flags:
            cmd += (
                f.split("=", 1)
                if f.startswith("--tools=") or f.startswith("--setting-sources=")
                else [f]
            )
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=int(os.environ.get("FLYPET_CLI_TIMEOUT", "300")),
        )
        if out.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {out.stderr[:500]} {out.stdout[:500]}")
        d = json.loads(out.stdout)
        self.last_model_ids = sorted((d.get("modelUsage") or {}).keys())
        self.last_usage = {
            "cost_usd": d.get("total_cost_usd"),
            **{
                k: v
                for k, v in (d.get("usage") or {}).items()
                if k
                in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                    "output_tokens",
                )
            },
        }
        if schema:
            so = d.get("structured_output")
            if so is not None:
                return so
            return json.loads(_strip_fences(d.get("result", "")))
        return d.get("result", "")


class AnthropicSDK(LLM):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("FLYPET_ANTHROPIC_MODEL", "claude-opus-5")

    def complete(self, system, prompt, schema=None, max_tokens=2000):
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
        )
        if schema:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": schema}
        resp = self.client.messages.create(**kwargs)
        self.last_model_ids = [str(resp.model)]
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(resp.stop_details, 'explanation', '')}")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        self.last_usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_input_tokens": resp.usage.cache_read_input_tokens,
        }
        return json.loads(text) if schema else text


class Ollama(LLM):
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("FLYPET_OLLAMA_MODEL", "qwen3:4b")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def complete(self, system, prompt, schema=None, max_tokens=2000):
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if os.environ.get("FLYPET_OLLAMA_KEEP_ALIVE"):
            body["keep_alive"] = os.environ["FLYPET_OLLAMA_KEEP_ALIVE"]
        if getattr(self, "context_length", None):
            body["options"]["num_ctx"] = self.context_length
        if os.environ.get("FLYPET_OLLAMA_NUM_THREAD"):
            body["options"]["num_thread"] = int(os.environ["FLYPET_OLLAMA_NUM_THREAD"])
        if schema:
            body["format"] = schema
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        self.last_model_ids = [d.get("model", self.model)]
        text = d["message"]["content"]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        self.last_usage = {
            "input_tokens": d.get("prompt_eval_count"),
            "output_tokens": d.get("eval_count"),
            "load_s": d.get("load_duration", 0) / 1e9,
            "prompt_s": d.get("prompt_eval_duration", 0) / 1e9,
            "generation_s": d.get("eval_duration", 0) / 1e9,
        }
        return json.loads(_strip_fences(text)) if schema else text


def get_llm(backend: str | None = None, model: str | None = None) -> LLM:
    backend = (
        backend
        or os.environ.get("FLYPET_LLM")
        or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "claude-cli")
    )
    if backend == "anthropic":
        return AnthropicSDK(model)
    if backend == "ollama":
        return Ollama(model)
    if backend == "claude-cli":
        return ClaudeCLI(model)
    raise ValueError(f"unknown backend {backend}")
