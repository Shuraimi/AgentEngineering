"""Quick smoke test for the configured OpenAI-compatible LLM provider."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
base_url = os.environ.get(
    "LLM_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
model = os.environ.get("AGENTFORGE_MODEL", "openai/gpt-oss-20b")

if not api_key:
    raise SystemExit("Set LLM_API_KEY in .env first.")

client = OpenAI(api_key=api_key, base_url=base_url)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: NVIDIA AgentForge connection works.",
        }
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
