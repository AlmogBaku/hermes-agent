"""Model switch without a prompt rebuild.

A ``/model`` switch used to rebuild the system prompt so its ``Model:`` / ``Provider:`` lines named
the new route, and the TUI gateway deleted the previous switch marker from the transcript. Both edit
what the conversation already sent: the cached prefix breaks at the edit (AGENTS.md: mutating past
context or rebuilding the system prompt mid-conversation), and Claude models that check preserved
thinking reject every earlier thinking block once the prompt or an earlier turn changes ("The block
is bound to a different conversation"), so the first request after a switch failed.

The stored prompt is therefore kept and the switch is announced in the transcript, append-only: the
TUI gateway appends its marker row at the switch, and any other path (CLI, messaging, a one-turn
model's restore) gets the same announcement as a one-shot note on the next user turn, stamped into
the byte-stable ``api_content`` sidecar like the surface-switch note (#104414). The newest
announcement is what the model was last told; older ones stay where they are and are summarized away
with the rest of the older turns at the next compaction, where the prompt converges too.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agent.message_content import flatten_message_text
from agent.surface_switch import identity_line_value

logger = logging.getLogger("run_agent")

MODEL_SWITCH_NOTE_PREFIX = "[System: The active model for this chat has changed to "
_PROVIDER_JOIN = " via provider "
# Model ids contain dots ("gpt-5.5"), so this sentence, not ".", ends the route in a parse.
_ROUTE_END = ". From this point forward"
_NOTE_SCAN_TAIL = 200  # only the newest announcement matters; see surface_switch._NOTE_SCAN_TAIL


def announces_in_transcript(agent: Any) -> bool:
    """MoA and codex_app_server turns never stamp the ``api_content`` sidecar, so a note could not be
    read back and would repeat every turn; those routes keep rebuilding the prompt on a switch."""
    return getattr(agent, "provider", None) != "moa" and getattr(agent, "api_mode", None) != "codex_app_server"


def keep_prompt_across_switch(agent: Any) -> None:
    """``switch_model``'s prompt step. The cached prompt stays, so the next turn sends the bytes the
    conversation already sent and announces the switch; its static tier is re-derived for the new
    route's caching policy. A route that cannot carry the announcement rebuilds next turn instead."""
    if announces_in_transcript(agent):
        from agent.system_prompt import reconstruct_static_prefix
        reconstruct_static_prefix(agent, log_label="model switch")
    else:
        agent._cached_system_prompt = None


def model_switch_note(model: str, provider: str, guidance: tuple[str, ...] = ()) -> str:
    provider_part = f"{_PROVIDER_JOIN}{provider}" if provider else ""
    note = (f"{MODEL_SWITCH_NOTE_PREFIX}{model}{provider_part}{_ROUTE_END}, use this runtime metadata "
            "when answering questions about what model/provider is active.]")
    return "\n".join((note, *guidance))


def _announced_route(text: str) -> Optional[tuple[str, str]]:
    if MODEL_SWITCH_NOTE_PREFIX not in text:
        return None
    route = text.rsplit(MODEL_SWITCH_NOTE_PREFIX, 1)[1].split(_ROUTE_END, 1)[0]
    model, _, provider = route.partition(_PROVIDER_JOIN)
    return model.strip(), provider.strip()


def last_announced_route(conversation_history: Any) -> Optional[tuple[str, str]]:
    """``(model, provider)`` of the newest announcement: a TUI marker row, or a note on a user row
    (in its ``api_content`` sidecar, or a text part when the content is a multimodal list)."""
    for msg in reversed((conversation_history or [])[-_NOTE_SCAN_TAIL:]):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        sidecar = msg.get("api_content")
        text = (sidecar if isinstance(sidecar, str) else "") + "\n" + flatten_message_text(msg.get("content"))
        if (route := _announced_route(text)) is not None:
            return route
    return None


def told_route(prompt: Optional[str], conversation_history: Any) -> tuple[str, str]:
    """What the model was last told: the newest announcement, else ``prompt``'s own identity lines
    (``("", "")`` when it has none, so there is nothing to correct)."""
    prompt = prompt if isinstance(prompt, str) else ""
    return last_announced_route(conversation_history) or (
        identity_line_value(prompt, "Model"), identity_line_value(prompt, "Provider"))


def route_was_told(told: tuple[str, str], model: str, provider: str) -> bool:
    """An announcement without a provider (the marker of a provider-less route) names the model only."""
    told_model, told_provider = told
    return told_model == model and told_provider in ("", provider)


def missing_guidance(agent: Any, prompt: str) -> tuple[str, ...]:
    """The blocks the prompt build would choose for the current model that ``prompt`` lacks."""
    from agent.coding_context import edit_format_guidance
    from agent.system_prompt import model_specific_parts

    missing = [part for part in model_specific_parts(agent) if part and part not in prompt]
    edit_format, every_edit_format = edit_format_guidance(getattr(agent, "model", None))
    # The nudge lives inside the coding brief; carry it only where the kept prompt has a brief.
    if edit_format and edit_format not in prompt and any(line in prompt for line in every_edit_format):
        missing.append(edit_format)
    return tuple(missing)


def model_switch_announcement(agent: Any, prompt: str) -> str:
    """The announcement for the agent's current route, with the guidance ``prompt`` lacks."""
    model = str(getattr(agent, "model", "") or "").strip()
    provider = str(getattr(agent, "provider", "") or "").strip()
    return model_switch_note(model, provider, missing_guidance(agent, prompt or ""))


def stage_model_switch_note(agent: Any, prompt: Optional[str], conversation_history: Any) -> bool:
    """Stage a one-shot announcement when the model was last told another route: the newest
    announcement in the transcript, else ``prompt``'s own ``Model:`` / ``Provider:`` lines. A prompt
    without identity lines told the model nothing to correct. Returns whether it staged."""
    if not isinstance(prompt, str) or not prompt or not announces_in_transcript(agent):
        return False
    model = str(getattr(agent, "model", "") or "").strip()
    provider = str(getattr(agent, "provider", "") or "").strip()
    told = told_route(prompt, conversation_history)
    if not model or not told[0] or route_was_told(told, model, provider):
        return False
    agent._model_switch_note = model_switch_announcement(agent, prompt)
    logger.info(
        "Session %s switched model %s -> %s; keeping the stored system prompt and announcing the "
        "switch in the transcript.", getattr(agent, "session_id", None), told[0], model,
    )
    return True
