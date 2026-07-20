"""Unit tests for installer/hooks_merger.py.

These tests pin the **most important contract in the codebase**: every
single re-run of ``agent-cot init`` must leave third-party hook
entries (codebuddy-mem, hook-handler, the user's own scripts) untouched
and must not pile up duplicates of our own.

If anything in this file regresses, P1 is unsafe to ship.
"""

from __future__ import annotations

import json
import textwrap
from copy import deepcopy

import pytest

from agent_cot.agents.cursor import CursorAdapter
from agent_cot.agents.codebuddy import CodeBuddyAdapter
from agent_cot.installer.codebuddy_hooks_merger import (
    diff_codebuddy_hooks,
    merge_codebuddy_hooks,
)
from agent_cot.installer.hooks_merger import (
    OWNED_HOOK_SCRIPTS,
    diff_hooks,
    is_owned_command,
    merge_cursor_hooks,
)

# Real-world fixture taken verbatim from the user's machine on
# 2026-04-27 — the codebuddy-mem + hook-handler shape we MUST preserve.
REAL_FIXTURE_JSON = textwrap.dedent(
    """\
    {
      "version": 1,
      "hooks": {
        "beforeSubmitPrompt": [
          {
            "command": "cmd /c C:/Users/milkwang/.codebuddy-mem/bin/node.exe \\"C:/Users/milkwang/.cursor/hooks/codebuddy-mem.js\\" beforeSubmitPrompt",
            "timeout": 10
          },
          {
            "command": "node \\"C:/Users/milkwang/.cursor/hooks/hook-handler.js\\"",
            "timeout": 30
          }
        ],
        "afterAgentResponse": [
          {
            "command": "cmd /c C:/Users/milkwang/.codebuddy-mem/bin/node.exe \\"C:/Users/milkwang/.cursor/hooks/codebuddy-mem.js\\" afterAgentResponse",
            "timeout": 30
          },
          {
            "command": "node \\"C:/Users/milkwang/.cursor/hooks/hook-handler.js\\"",
            "timeout": 30
          },
          {
            "command": "node \\"C:/Users/milkwang/.cursor/hooks/cot-stream.js\\"",
            "timeout": 5
          }
        ],
        "stop": [
          {
            "command": "cmd /c C:/Users/milkwang/.codebuddy-mem/bin/node.exe \\"C:/Users/milkwang/.cursor/hooks/codebuddy-mem.js\\" stop",
            "timeout": 30
          },
          {
            "command": "node \\"C:/Users/milkwang/.cursor/hooks/hook-handler.js\\"",
            "timeout": 30
          },
          {
            "command": "node \\"C:/Users/milkwang/.cursor/hooks/cot-bridge.js\\"",
            "timeout": 10
          },
          {
            "command": "python \\"C:/Users/milkwang/.cursor/hooks/agent_critic_hook.py\\" --agent cursor --event stop",
            "timeout": 10
          }
        ]
      }
    }
    """
)


def _real_fixture() -> dict:
    return json.loads(REAL_FIXTURE_JSON)


# ---------------------------------------------------------------------------
# is_owned_command
# ---------------------------------------------------------------------------


def test_is_owned_command_recognises_cot_bridge() -> None:
    assert is_owned_command('node "/home/u/.cursor/hooks/cot-bridge.js"')
    assert is_owned_command('node "C:/Users/u/.cursor/hooks/cot-bridge.js"')
    assert is_owned_command(
        'node "C:\\\\Users\\\\u\\\\.cursor\\\\hooks\\\\cot-bridge.js"'
    )


def test_is_owned_command_recognises_cot_stream() -> None:
    assert is_owned_command('node "/home/u/.cursor/hooks/cot-stream.js"')


def test_is_owned_command_rejects_third_party() -> None:
    assert not is_owned_command(
        'node "C:/Users/u/.cursor/hooks/codebuddy-mem.js" stop'
    )
    assert not is_owned_command(
        'node "C:/Users/u/.cursor/hooks/hook-handler.js"'
    )
    assert not is_owned_command("")
    assert not is_owned_command("python my-cot-bridge.py")  # different file


def test_is_owned_command_handles_non_string() -> None:
    assert not is_owned_command(None)  # type: ignore[arg-type]
    assert not is_owned_command(123)  # type: ignore[arg-type]


def test_owned_filenames_are_what_we_advertise() -> None:
    # Pinning this fact here so accidental rename of the script gets caught.
    # v0.19.1: claude_stream_hook.py joins the owned set (the only .py hook).
    assert OWNED_HOOK_SCRIPTS == frozenset(
        {
            "cot-bridge.js",
            "cot-stream.js",
            "cot-stream-codebuddy.js",
            "cot-bridge-codebuddy.js",
            "claude_stream_hook.py",
            "agent_critic_hook.py",
        }
    )


# ---------------------------------------------------------------------------
# merge_cursor_hooks — preservation
# ---------------------------------------------------------------------------


def test_merge_into_empty_creates_skeleton() -> None:
    out = merge_cursor_hooks(None, CursorAdapter().hook_entries())
    assert out["version"] == 1
    assert "stop" in out["hooks"]
    assert any("cot-bridge.js" in e["command"] for e in out["hooks"]["stop"])


def test_merge_does_not_mutate_input() -> None:
    snapshot = _real_fixture()
    frozen_snapshot = deepcopy(snapshot)
    merge_cursor_hooks(snapshot, CursorAdapter().hook_entries())
    assert snapshot == frozen_snapshot


def test_merge_preserves_other_owners_byte_for_byte() -> None:
    """The single most important invariant of P1."""
    fixture = _real_fixture()
    out = merge_cursor_hooks(fixture, CursorAdapter().hook_entries())

    # Every non-owned entry from the fixture must appear, verbatim,
    # in the output.
    for event, entries in fixture["hooks"].items():
        for entry in entries:
            if is_owned_command(entry["command"]):
                continue
            assert entry in out["hooks"][event], (
                f"third-party hook lost during merge in event '{event}': {entry}"
            )


def test_merge_replaces_owned_entries_in_stop() -> None:
    fixture = _real_fixture()
    out = merge_cursor_hooks(fixture, CursorAdapter().hook_entries())
    owned_in_stop = [
        e for e in out["hooks"]["stop"] if is_owned_command(e["command"])
    ]
    # Exactly one cot-bridge.js entry survives — the freshly added one,
    # not the user's hand-written copy.
    assert len(owned_in_stop) == 1
    assert sum("cot-bridge.js" in e["command"] for e in owned_in_stop) == 1
    assert all("agent_critic_hook.py" not in e["command"] for e in owned_in_stop)


def test_merge_is_idempotent() -> None:
    """Running merge twice should produce the same result as running once."""
    fixture = _real_fixture()
    additions = CursorAdapter().hook_entries()
    once = merge_cursor_hooks(fixture, additions)
    twice = merge_cursor_hooks(once, additions)
    assert once == twice


def test_merge_idempotent_across_many_runs() -> None:
    """Slightly stronger: 5 consecutive runs converge."""
    state = _real_fixture()
    additions = CursorAdapter().hook_entries()
    history = [deepcopy(state)]
    for _ in range(5):
        state = merge_cursor_hooks(state, additions)
        history.append(deepcopy(state))
    # After the first merge, all subsequent ones are no-ops
    assert history[1] == history[2] == history[3] == history[4] == history[5]


def test_merge_does_not_touch_unrelated_events() -> None:
    fixture = _real_fixture()
    out = merge_cursor_hooks(fixture, CursorAdapter().hook_entries())
    # `beforeSubmitPrompt` has no owned entries in the fixture, no
    # additions touch it: it must be byte-equal.
    assert out["hooks"]["beforeSubmitPrompt"] == fixture["hooks"]["beforeSubmitPrompt"]


def test_merge_appends_after_existing_entries() -> None:
    """User's `cot-stream.js` in afterAgentResponse should be removed,
    new one should appear AFTER codebuddy-mem + hook-handler."""
    fixture = _real_fixture()
    out = merge_cursor_hooks(fixture, CursorAdapter().hook_entries())
    after = out["hooks"]["afterAgentResponse"]
    # First two entries unchanged
    assert "codebuddy-mem.js" in after[0]["command"]
    assert "hook-handler.js" in after[1]["command"]
    # Last entry is the freshly-added cot-stream
    assert is_owned_command(after[-1]["command"])
    assert "cot-stream.js" in after[-1]["command"]


def test_merge_rejects_malformed_hooks_block() -> None:
    bad = {"version": 1, "hooks": "not-a-dict"}
    with pytest.raises(ValueError):
        merge_cursor_hooks(bad, CursorAdapter().hook_entries())


# ---------------------------------------------------------------------------
# diff_hooks — for dry-run
# ---------------------------------------------------------------------------


def test_diff_on_empty_input_lists_only_additions() -> None:
    diff = diff_hooks(None, CursorAdapter().hook_entries())
    assert diff.has_changes
    # 1 stop hook (bridge) + 8 stream taps from cursor.py = 9 additions.
    assert len(diff.added) == 9
    assert diff.removed == []
    assert diff.untouched_other_owners == 0


def test_diff_on_already_synced_fixture_is_noop() -> None:
    """If hooks.json already contains exactly our entries, diff is empty."""
    additions = CursorAdapter().hook_entries()
    synced = merge_cursor_hooks(None, additions)
    diff = diff_hooks(synced, additions)
    assert not diff.has_changes


def test_diff_counts_third_party_as_untouched() -> None:
    fixture = _real_fixture()
    diff = diff_hooks(fixture, CursorAdapter().hook_entries())
    # codebuddy-mem (5 entries observed in fixture: beforeSubmitPrompt,
    # afterAgentResponse, stop) + hook-handler (3 entries) = 6.
    # The exact number is brittle; just assert > 0 and that it's an int.
    assert isinstance(diff.untouched_other_owners, int)
    assert diff.untouched_other_owners >= 5


def test_diff_render_when_no_changes() -> None:
    additions = CursorAdapter().hook_entries()
    synced = merge_cursor_hooks(None, additions)
    rendered = diff_hooks(synced, additions).render()
    assert "no changes" in rendered.lower()


# ---------------------------------------------------------------------------
# CodeBuddy settings.json merger
# ---------------------------------------------------------------------------


def _codebuddy_fixture() -> dict:
    return {
        "enabledPlugins": {"find-skills@codebuddy-plugins-official": True},
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": '"C:/Users/u/.codebuddy-mem/hooks/cbmem.cmd" UserPromptSubmit',
                            "timeout": 10000,
                        }
                    ]
                }
            ],
            "PostToolUse": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": 'node "C:/Users/u/.codebuddy/hooks/cot-stream-codebuddy.js" PostToolUse',
                            "timeout": 10000,
                        }
                    ]
                },
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": '"C:/Users/u/.codebuddy-mem/hooks/cbmem.cmd" PostToolUse',
                            "timeout": 10000,
                        }
                    ],
                },
            ],
        },
    }


def test_codebuddy_merge_preserves_settings_and_third_party_hooks() -> None:
    fixture = _codebuddy_fixture()
    out = merge_codebuddy_hooks(fixture, CodeBuddyAdapter().hook_entries())

    assert out["enabledPlugins"] == fixture["enabledPlugins"]
    third_party = fixture["hooks"]["UserPromptSubmit"][0]
    assert third_party in out["hooks"]["UserPromptSubmit"]
    assert any(
        "cot-stream-codebuddy.js" in hook["command"]
        for group in out["hooks"]["Stop"]
        for hook in group["hooks"]
    )


def test_codebuddy_merge_replaces_owned_nested_hooks() -> None:
    out = merge_codebuddy_hooks(_codebuddy_fixture(), CodeBuddyAdapter().hook_entries())
    owned_post_tool = [
        hook
        for group in out["hooks"]["PostToolUse"]
        for hook in group.get("hooks", [])
        if is_owned_command(hook.get("command", ""))
    ]
    assert len(owned_post_tool) == 1
    assert owned_post_tool[0]["command"].endswith("PostToolUse")


def test_codebuddy_merge_is_idempotent() -> None:
    additions = CodeBuddyAdapter().hook_entries()
    once = merge_codebuddy_hooks(_codebuddy_fixture(), additions)
    twice = merge_codebuddy_hooks(once, additions)
    assert once == twice


def test_codebuddy_diff_on_synced_fixture_is_noop() -> None:
    additions = CodeBuddyAdapter().hook_entries()
    synced = merge_codebuddy_hooks(None, additions)
    diff = diff_codebuddy_hooks(synced, additions)
    assert not diff.has_changes


# ---------------------------------------------------------------------------
# v0.20.4: Claude OTel env fill-missing merger
# ---------------------------------------------------------------------------


def test_claude_env_merger_fills_missing_keys_on_empty_settings() -> None:
    from agent_cot.agents.claude import ClaudeAdapter
    from agent_cot.installer.claude_hooks_merger import merge_claude_env

    recommended = ClaudeAdapter().recommended_env(backend_port=8765)
    merged, added, preserved = merge_claude_env(None, recommended)

    for k, v in recommended.items():
        assert merged["env"][k] == v
    assert sorted([k for k, _ in added]) == sorted(recommended.keys())
    assert preserved == []
    assert "http://127.0.0.1:8765" in merged["env"]["OTEL_EXPORTER_OTLP_ENDPOINT"]


def test_claude_env_merger_preserves_user_set_endpoint() -> None:
    """User's corp collector / langfuse / phoenix endpoint MUST NOT be clobbered."""
    from agent_cot.agents.claude import ClaudeAdapter
    from agent_cot.installer.claude_hooks_merger import merge_claude_env

    user_endpoint = "https://otel.corp.example.com:4318"
    existing = {
        "env": {
            "OTEL_EXPORTER_OTLP_ENDPOINT": user_endpoint,
            "OTEL_RESOURCE_ATTRIBUTES": "team=my-team",
        },
        "hooks": {"Stop": []},
    }
    recommended = ClaudeAdapter().recommended_env(backend_port=8766)
    merged, added, preserved = merge_claude_env(existing, recommended)

    # User's keys are kept exactly as set.
    assert merged["env"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == user_endpoint
    assert merged["env"]["OTEL_RESOURCE_ATTRIBUTES"] == "team=my-team"
    # Endpoint shows up in preserved (user's value, NOT our recommendation).
    preserved_dict = dict(preserved)
    assert preserved_dict["OTEL_EXPORTER_OTLP_ENDPOINT"] == user_endpoint
    # CLAUDE_CODE_ENABLE_TELEMETRY was absent — we DO add that.
    added_dict = dict(added)
    assert added_dict["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    # Untouched non-env top-level entries survive.
    assert merged["hooks"] == {"Stop": []}


def test_claude_env_merger_is_idempotent() -> None:
    from agent_cot.agents.claude import ClaudeAdapter
    from agent_cot.installer.claude_hooks_merger import merge_claude_env

    recommended = ClaudeAdapter().recommended_env(backend_port=8765)
    once, _a1, _p1 = merge_claude_env(None, recommended)
    twice, added2, preserved2 = merge_claude_env(once, recommended)

    assert twice == once
    assert added2 == []
    # Every recommended key is now preserved (already present).
    assert sorted([k for k, _ in preserved2]) == sorted(recommended.keys())


def test_claude_env_diff_pure_no_mutation() -> None:
    from agent_cot.agents.claude import ClaudeAdapter
    from agent_cot.installer.claude_hooks_merger import diff_claude_env

    recommended = ClaudeAdapter().recommended_env(backend_port=8765)
    existing = {"env": {"CLAUDE_CODE_ENABLE_TELEMETRY": "0"}}
    before = json.dumps(existing, sort_keys=True)
    diff = diff_claude_env(existing, recommended)
    after = json.dumps(existing, sort_keys=True)

    assert before == after, "diff must not mutate the input settings"
    assert ("CLAUDE_CODE_ENABLE_TELEMETRY", "0") in diff.preserved
    assert any(k == "OTEL_LOGS_EXPORTER" for k, _ in diff.added)
    assert diff.has_changes  # at least one key to add


def test_claude_env_merger_rejects_non_dict_env_block() -> None:
    from agent_cot.installer.claude_hooks_merger import merge_claude_env

    # Some users put a list in 'env' by accident; merger must refuse cleanly.
    with pytest.raises(ValueError, match="must be an object"):
        merge_claude_env({"env": ["bogus"]}, {"FOO": "bar"})


def test_claude_endpoint_uses_127_not_localhost() -> None:
    """Doc-as-code: matches CLAUDE_OTEL.md FAQ about Windows Defender."""
    from agent_cot.agents.claude import ClaudeAdapter

    url = ClaudeAdapter.otel_endpoint_for_port(9000)
    assert url == "http://127.0.0.1:9000"
    assert "localhost" not in url
