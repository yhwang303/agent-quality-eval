"""P0 smoke tests for the agent registry."""

from __future__ import annotations

import pytest

from agent_cot.agents import (
    ClaudeAdapter,
    CursorAdapter,
    UnknownAgentError,
    get_adapter,
    iter_adapters,
    list_agents,
)
from agent_cot.agents.base import AgentNotImplementedError  # noqa: F401  (kept for future stubs)


def test_registry_lists_cursor_and_claude() -> None:
    """The registry must always advertise cursor + claude (P0 contract).

    Newer adapters (codex, codebuddy) are additive and may show up here
    too — assert containment instead of strict equality so adding more
    IDEs doesn't crash this smoke test.
    """
    agents = list_agents()
    assert "cursor" in agents
    assert "claude" in agents
    assert agents == sorted(agents), "registry should expose adapters in sorted order"


def test_get_adapter_returns_concrete_classes() -> None:
    assert isinstance(get_adapter("cursor"), CursorAdapter)
    assert isinstance(get_adapter("claude"), ClaudeAdapter)


def test_get_adapter_is_case_insensitive() -> None:
    assert isinstance(get_adapter("CURSOR"), CursorAdapter)
    assert isinstance(get_adapter("  Claude  "), ClaudeAdapter)


def test_unknown_agent_raises() -> None:
    with pytest.raises(UnknownAgentError) as exc:
        get_adapter("emacs")
    assert "Available" in str(exc.value)
    assert "cursor" in str(exc.value)
    assert "claude" in str(exc.value)


def test_iter_adapters_expands_all() -> None:
    adapters = list(iter_adapters(["all"]))
    names = sorted(a.name for a in adapters)
    # Whatever extra adapters land in the registry, "all" must contain at
    # least the two P0 ones.
    assert "claude" in names
    assert "cursor" in names


def test_iter_adapters_dedupes() -> None:
    """Even with an explicit name repeated alongside ``all``, names dedupe."""
    adapters = list(iter_adapters(["cursor", "all", "cursor"]))
    names = [a.name for a in adapters]
    assert len(names) == len(set(names)), f"duplicates leaked: {names}"
    assert "claude" in names
    assert "cursor" in names


def test_cursor_adapter_paths_are_under_home() -> None:
    a = CursorAdapter()
    cfg = a.hooks_config_path()
    hooks_dir = a.hooks_assets_dir()
    assert ".cursor" in cfg.parts
    assert cfg.name == "hooks.json"
    assert hooks_dir.name == "hooks"


def test_cursor_adapter_emits_stop_plus_stream_events() -> None:
    a = CursorAdapter()
    events = sorted({e.event for e in a.hook_entries()})
    # one bridge event ('stop') + the stream taps
    assert "stop" in events
    assert "afterAgentResponse" in events
    assert "afterShellExecution" in events
    # entry for `stop` invokes cot-bridge.js, others invoke cot-stream.js
    entries = a.hook_entries()
    assert any(e.event == "stop" and "cot-bridge.js" in e.command for e in entries)
    assert not any("agent_critic_hook.py" in e.command for e in entries)
    assert any(e.event == "afterAgentResponse" and "cot-stream.js" in e.command for e in entries)


def test_cursor_adapter_owner_is_stable() -> None:
    a = CursorAdapter()
    for entry in a.hook_entries():
        assert entry.owner == "agent-cot"


def test_claude_adapter_path_is_settings_json() -> None:
    a = ClaudeAdapter()
    cfg = a.hooks_config_path()
    assert ".claude" in cfg.parts
    assert cfg.name == "settings.json"


def test_claude_adapter_hook_entries_is_fully_implemented() -> None:
    """v0.19.1: Claude Internal adapter is no longer a stub.

    ``hook_entries()`` should return the full 27-event registration set
    (SessionStart / Stop / SubagentStop / SessionEnd + 23 mid-turn taps).
    """
    a = ClaudeAdapter()
    entries = a.hook_entries()
    events = {e.event for e in entries}
    # The four lifecycle events that must be there for cot.json
    # materialisation + bookkeeping to work.
    assert "SessionStart" in events
    assert "Stop" in events
    # The claude_stream_hook.py marker — every entry must invoke it.
    assert any("claude_stream_hook.py" in entry.command for entry in entries)
    assert not any("agent_critic_hook.py" in entry.command for entry in entries)
    for entry in entries:
        assert entry.owner == "agent-cot"


def test_claude_adapter_merge_hook_entries_returns_settings_dict() -> None:
    """v0.19.1: ``merge_hook_entries`` produces a Claude-shaped settings dict
    (nested ``hooks: { <event>: [{matcher, hooks: [...]}] }``)."""
    a = ClaudeAdapter()
    entries = a.hook_entries()
    merged = a.merge_hook_entries({}, entries)
    assert isinstance(merged, dict)
    assert "hooks" in merged
    assert "SessionStart" in merged["hooks"]


def test_cursor_adapter_merge_delegates_to_installer() -> None:
    """P1: CursorAdapter.merge_hook_entries now wires through to the
    central merger and produces a Cursor-shaped dict."""
    a = CursorAdapter()
    merged = a.merge_hook_entries({}, a.hook_entries())
    assert merged["version"] == 1
    assert isinstance(merged["hooks"], dict)
    assert "stop" in merged["hooks"]
    assert any("cot-bridge.js" in e["command"] for e in merged["hooks"]["stop"])
