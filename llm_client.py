"""
Shared LLM client for AgentForge.

The application uses an OpenAI-compatible API so the model provider can be
changed through .env without changing Builder/Reflector/Runner code.

Default provider: NVIDIA hosted API
Default model: openai/gpt-oss-20b
"""

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.environ.get("AGENTFORGE_MODEL", "openai/gpt-oss-20b")
BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")

_client = None


def get_client() -> OpenAI:
    """Return one shared OpenAI-compatible client."""
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "Missing LLM_API_KEY (or NVIDIA_API_KEY). "
                "Set it in .env before running AgentForge."
            )
        _client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
    return _client


def _extract_json(text: str) -> Any:
    """Parse JSON even when the model adds markdown fences or brief prose."""
    text = (text or "").strip()

    # Remove a ```json ... ``` wrapper.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Recover the outermost JSON object/array if the model added prose.
    starts = [(text.find("{"), "}"), (text.find("["), "]")]
    candidates = [(s, end) for s, end in starts if s >= 0]
    if not candidates:
        raise ValueError(f"Could not find JSON in model response:\n{text}")

    start, closing = min(candidates, key=lambda x: x[0])
    end = text.rfind(closing)
    if end < start:
        raise ValueError(f"Could not extract JSON from model response:\n{text}")

    return json.loads(text[start:end + 1])


def call_for_json(prompt: str, max_tokens: int = 4096) -> dict:
    """
    Call the configured OpenAI-compatible model and return a parsed JSON dict.

    A single retry is used because hosted models can occasionally return
    fenced JSON or a small amount of prose despite the JSON-only instruction.
    """
    client = get_client()

    def _attempt(extra_instruction: str = "") -> dict:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt + extra_instruction,
                }
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        raw = message.content or ""

        # GPT-OSS can expose its harmony/reasoning channel markers in
        # message.content on some hosted NIM responses. Keep only the final
        # answer portion when those markers are present.
        if "<|channel|>final" in raw:
            raw = raw.split("<|channel|>final", 1)[1]
        raw = raw.replace("<|message|>", "").strip()

        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object from the model.")
        return parsed

    try:
        return _attempt()
    except (json.JSONDecodeError, ValueError) as first_error:
        try:
            return _attempt(
                "\n\nIMPORTANT: your previous response was not valid JSON. "
                "Return ONLY the requested JSON object. Do not explain your answer, "
                "do not use markdown fences, and keep the JSON concise."
            )
        except Exception as second_error:
            raise ValueError(
                "The LLM returned invalid/empty JSON twice. "
                f"First error: {first_error}. Second error: {second_error}"
            ) from second_error


# NVIDIA's hosted gpt-oss-20b endpoint currently exposes a free endpoint.
# Keep these configurable so the project can also be pointed at another
# OpenAI-compatible provider later.
INPUT_COST_PER_MTOK = float(os.environ.get("LLM_INPUT_COST_PER_MTOK", "0"))
OUTPUT_COST_PER_MTOK = float(os.environ.get("LLM_OUTPUT_COST_PER_MTOK", "0"))


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * INPUT_COST_PER_MTOK
        + (output_tokens / 1_000_000) * OUTPUT_COST_PER_MTOK
    )
