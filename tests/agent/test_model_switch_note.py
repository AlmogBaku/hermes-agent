"""A model switch keeps what the conversation already sent and announces the new route once.

Rebuilding the system prompt on a switch broke the cached prefix and, on Claude models that check
preserved thinking, made the first request after the switch fail ("The block is bound to a different
conversation"). See agent/model_switch_note.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.model_switch_note import model_switch_note, stage_model_switch_note
from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE

CLAUDE = ("claude-opus-5-5", "anthropic")
GPT = ("gpt-5.5", "openai")


def _prompt(route: tuple[str, str], extra: str = "") -> str:
    return (f"You are Hermes Agent.{extra}\n\nConversation started: Monday, November 16, 2026\n"
            f"Model: {route[0]}\nProvider: {route[1]}")


STORED = _prompt(("x-ai/grok-4", "openrouter"))


def _live_switch():
    """A real AIAgent with a cached prompt, switched in place (the CLI and TUI /model path)."""
    from tests.agent.test_switch_model_fallback_prune import _make_agent, _switch_to_anthropic

    agent = _make_agent([])
    agent._cached_system_prompt = STORED
    _switch_to_anthropic(agent)
    return agent, agent._cached_system_prompt != STORED


def _resume(provider: str):
    """A fresh per-turn agent restoring a stored prompt written for another model (the gateway path)."""
    from agent.conversation_loop import _restore_or_build_system_prompt
    from tests.agent.test_system_prompt_restore import _make_agent

    db = MagicMock()
    db.get_session.return_value = {"system_prompt": STORED}
    agent = _make_agent(session_db=db)
    agent.provider = provider
    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])
    return agent, agent._build_system_prompt.called or db.update_system_prompt.called


# case -> (switch, whether the prompt is rebuilt)
_SWITCHES = {
    "live_switch": (_live_switch, False),
    "resume": (lambda: _resume("openrouter"), False),
    # MoA never stamps the api_content sidecar, so it cannot carry the announcement and keeps rebuilding.
    "resume_on_moa": (lambda: _resume("moa"), True),
}


@pytest.mark.parametrize("case", list(_SWITCHES), ids=list(_SWITCHES))
def test_a_model_switch_keeps_the_system_prompt_the_conversation_already_sent(case):
    switch, rebuilds = _SWITCHES[case]
    agent, rebuilt = switch()
    assert rebuilt is rebuilds
    if not rebuilds:
        assert agent._cached_system_prompt == STORED


def _note_row(route: tuple[str, str]) -> dict:
    """A user turn that carried an announcement in its byte-stable api_content sidecar."""
    return {"role": "user", "content": "hi", "api_content": f"hi\n\n{model_switch_note(*route)}"}


def _marker_row(route: tuple[str, str]) -> dict:
    """The TUI gateway's marker row (tui_gateway.server._append_model_switch_marker)."""
    return {"role": "user", "content": model_switch_note(*route), "display_kind": "model_switch"}


# case -> (prompt, current route, transcript, announced route or None, carries the tool-use guidance)
_TOLD = {
    "switched": (_prompt(CLAUDE), GPT, [], GPT, True),
    "never_switched": (_prompt(CLAUDE), CLAUDE, [], None, False),
    "announced_by_an_earlier_note": (_prompt(CLAUDE), GPT, [_note_row(GPT)], None, False),
    "announced_by_the_tui_marker": (_prompt(CLAUDE), GPT, [_marker_row(GPT)], None, False),
    "marker_without_provider": (_prompt(CLAUDE), GPT, [_marker_row((GPT[0], ""))], None, False),
    "switched_back": (_prompt(CLAUDE), CLAUDE, [_note_row(GPT)], CLAUDE, False),
    "guidance_already_in_prompt": (_prompt(CLAUDE, "\n\n" + TOOL_USE_ENFORCEMENT_GUIDANCE), GPT, [], GPT, False),
}


@pytest.mark.parametrize("case", list(_TOLD), ids=list(_TOLD))
def test_the_newest_user_turn_tells_the_model_its_route_once(case):
    """The model is told the route it now runs on: once, on the next user turn, with the model-specific
    guidance the kept prompt lacks; never again once the transcript carries it."""
    from agent.turn_context import _merge_gateway_notes

    prompt, (model, provider), history, announced, guidance = _TOLD[case]
    agent = SimpleNamespace(
        model=model, provider=provider, api_mode="chat_completions", session_id="s",
        # Gates pinned to GPT here, so the cases do not depend on the shipped model lists.
        valid_tool_names=["terminal"], _tool_use_enforcement=[GPT[0]], _execution_guidance=False,
        _model_switch_note="",
    )
    staged = stage_model_switch_note(agent, prompt, history)

    assert staged is (announced is not None)
    if not staged:
        assert agent._model_switch_note == ""
        return
    note = _merge_gateway_notes(agent, [{"role": "user", "content": "next"}], 0, "")
    assert note.startswith(model_switch_note(*announced).split("\n")[0])
    assert (TOOL_USE_ENFORCEMENT_GUIDANCE in note) is guidance
    assert _merge_gateway_notes(agent, [{"role": "user", "content": "after"}], 0, "") == ""
    # The note is stamped into that turn's sidecar; a later turn reads it back and stages nothing.
    later = history + [{"role": "user", "content": "next", "api_content": f"next\n\n{note}"}]
    assert not stage_model_switch_note(agent, prompt, later)
