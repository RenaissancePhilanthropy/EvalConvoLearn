import os

from openai import OpenAI

_CLAUDE_ALIASES = {
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4": "claude-sonnet-4-0529",
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "opus-4.7": "claude-opus-4-7",
}


def resolve_model(model: str) -> str:
    return _CLAUDE_ALIASES.get(model, model)


def make_client(model: str) -> OpenAI:
    full_model = resolve_model(model)
    if full_model.startswith("claude-"):
        return OpenAI(
            base_url="https://api.anthropic.com/v1",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    return OpenAI()
