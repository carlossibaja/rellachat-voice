"""Short Spanish summaries of a call transcript via the Anthropic Messages API."""

from anthropic import AsyncAnthropic
from loguru import logger

from rellachat.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    TRANSFER_SUMMARY_SYSTEM_PROMPT,
    format_transcript,
)
from rellachat.usage import CallUsage


async def _summarize(
    api_key: str, model: str, system: str, transcript: list[dict], usage: CallUsage | None
) -> str:
    text = format_transcript(transcript)
    if not text:
        return "La llamada terminó sin conversación registrada."
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; skipping summary")
        return ""
    try:
        client = AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": f"Transcripción:\n{text}"}],
        )
        if usage is not None:
            # The call's own LLM usage: billed with the rest of the call
            usage.add_anthropic_usage(model, getattr(response, "usage", None))
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return ""


async def summarize_call(
    api_key: str, model: str, transcript: list[dict], usage: CallUsage | None = None
) -> str:
    """Summary stored on the call when it ends; its tokens are added to ``usage``."""
    return await _summarize(api_key, model, SUMMARY_SYSTEM_PROMPT, transcript, usage)


async def summarize_for_transfer(
    api_key: str, model: str, transcript: list[dict], usage: CallUsage | None = None
) -> str:
    """Summary shown to advisors while the transfer rings; its tokens are added to ``usage``."""
    summary = await _summarize(api_key, model, TRANSFER_SUMMARY_SYSTEM_PROMPT, transcript, usage)
    return summary or "El cliente pidió hablar con un asesor."
