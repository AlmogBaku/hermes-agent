"""Tests for _append_model_switch_marker role fix (issue #48338).

The model switch marker must NOT use role="system" because strict providers
(vLLM, Qwen) reject system messages that appear mid-conversation. Using
role="user" is safe — the system prompt is prepended to the API message list,
so a user-role marker can appear at any later position, and the gateway's
sanitize/merge pass already coalesces consecutive user messages.
"""

from __future__ import annotations


from tui_gateway.server import _append_model_switch_marker


class TestAppendModelSwitchMarkerRole:
    """Verify the marker uses role='user', not role='system'."""

    def test_marker_uses_user_role(self) -> None:
        """The history entry must be role='user', not role='system'."""
        session: dict = {"session_key": "test-session", "history": []}
        _append_model_switch_marker(session, model="gpt-4o", provider="openai")
        assert len(session["history"]) == 1
        entry = session["history"][0]
        assert entry["role"] == "user", (
            f"Expected role='user' but got role='{entry['role']}'. "
            "Strict providers (vLLM, Qwen) reject mid-conversation system messages."
        )




class TestModelSwitchMarkerAppendOnly:
    """A switch appends its marker and never edits the transcript before it: deleting an earlier
    marker changes what the conversation already sent — the cached prefix breaks there, and Claude
    models that check preserved thinking reject every later thinking block. The newest marker names
    the active model; superseded ones are summarized away at the next compaction."""

    @staticmethod
    def _markers(session: dict) -> list:
        return [h["content"] for h in session["history"] if _is_marker(h)]

    def test_each_switch_appends_and_leaves_earlier_entries_untouched(self) -> None:
        earlier = [
            {"role": "user", "content": "q1"},
            _make_marker_entry("model-a"),
            {"role": "assistant", "content": "a1"},
        ]
        session: dict = {"session_key": "s", "history": [dict(h) for h in earlier], "history_version": 0}
        _append_model_switch_marker(session, model="model-b", provider="p")
        _append_model_switch_marker(session, model="model-c", provider="p")
        assert session["history"][:3] == earlier
        markers = self._markers(session)
        assert len(markers) == 3
        assert "model-b via provider p" in markers[1] and "model-c via provider p" in markers[2]
        assert session["history"][-1]["content"] == markers[2]
        assert session["history_version"] == 2

    def test_a_switch_to_the_route_last_announced_appends_nothing(self) -> None:
        session: dict = {"session_key": "s", "history": [], "history_version": 0}
        _append_model_switch_marker(session, model="model-b", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        assert len(self._markers(session)) == 1
        assert session["history_version"] == 1


def _make_marker_entry(model: str) -> dict:
    from tui_gateway.server import _MODEL_SWITCH_MARKER_PREFIX

    return {"role": "user", "content": f"{_MODEL_SWITCH_MARKER_PREFIX}{model}.]"}


def _is_marker(entry: dict) -> bool:
    from tui_gateway.server import _is_model_switch_marker

    return _is_model_switch_marker(entry)
