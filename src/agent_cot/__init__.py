"""agent-cot: local-first Chain-of-Thought observability for Cursor + CodeBuddy + Claude Internal.

This package is the distribution layer of the `ai-ide-langfuse` project.
See ../../SETUP_PLAN.md for the full design.

v0.20.6 — Claude **dual-variant** + sub-agent folding + path-resolution
self-heal.

Three independent fixes that all surfaced from one司内 user report ("0.20.5
装完跟 Cursor 内嵌 Claude Code 交互，dashboard 不显示 + OTel 空"):

1. **Claude OSS vs Internal dual-write**. Claude Code exists in two
   flavors that share the same hook + env schema but read from different
   home directories:

     * ``~/.claude/``           — Anthropic OSS ``claude`` CLI
     * ``~/.claude-internal/``  — Tencent ``@tencent/claude-code-internal``
       (the one Cursor embeds — what most司内 users actually run)

   Pre-0.20.6 ``init --apply --agent claude`` only ever wrote to
   ``~/.claude/settings.json``. Cursor-embedded users got zero hooks,
   zero OTel env, zero CoT capture, an empty OtelPanel on every
   session. 0.20.6's ``ClaudeAdapter`` now picks the variant that
   actually exists as the *primary* write target AND mirrors to the
   other one when both are present, so a single ``init --apply --agent
   claude`` reliably wires up whichever Claude(s) the user has.

2. **Claude Task() sub-agent folding**. Claude Code's ``Task()`` tool
   spawns sub-agents that write their own transcript under the parent
   session's ``subagents/agent-*.jsonl``. Pre-0.20.6 these showed up
   as separate SessionList entries (one per sub-agent, with prompts
   that looked nothing like anything the user typed — because they
   were what the parent agent typed when delegating). 0.20.6 filters
   them from SessionList and merges them as ordered ``SubStart``
   timeline nodes on the parent session, with prompt preview,
   duration, model, tool distribution, and token cost.

3. **Path-resolution self-heal**. Two latent bugs that only triggered
   for contributors with ``D:/ai-ide-langfuse/{agent-dashboard,
   cot-extractor}/`` sibling git checkouts but quietly poisoned
   editable installs for months:

     * ``cot-bridge.js`` / ``cot-stream.js`` now consult
       ``~/.agent-cot/runtime.json`` *before* the install-time-baked
       literal. Stale literal can be healed by re-running
       ``agent-cot start``.
     * ``start.py._find_backend_dir`` and
       ``init.py._find_cot_extractor_root`` now prefer the
       wheel-bundled copy over any sibling git checkout. Maintainer
       dev environments stop silently loading months-old code.

   End-user impact for fresh ``pip install`` users: none — they have
   no sibling checkouts, never hit the broken branch.

``agent-cot doctor`` adds a ``claude.variants`` check that explicitly
prints ``OSS / Internal / Dual`` mode + a hint to restart the Claude
Code process after env injection (Claude reads settings.env once at
startup; live changes don't propagate).

Upgrade: colleagues currently on 0.20.5::

    pip install --upgrade observation-agent==0.20.6
    agent-cot init --apply --agent claude   # mirrors to both ~/.claude*/
    # → close & reopen the Cursor-embedded Claude Code terminal so the
    #   OTel env block in ~/.claude-internal/settings.json actually loads

────────────────────────────────────────────────────────────────────────

v0.20.5 — hotfix: 0.20.4 shipped a packaging regression that bricked the
Cursor adapter for every fresh install. The shipped wheel's
``assets/hooks/cursor/cot-bridge.js`` and ``assets/hooks/cursor/cot-stream.js``
were each 59-byte stubs (the one-line placeholder
``const COT_ROOT = process.env.COT_EXTRACTOR_ROOT || 'OLD';``) instead of
the real 18 KB / 16 KB hook scripts. End-user impact: Cursor still fired
hooks, ``node`` still ran the file, but the file did nothing — no
``events.jsonl`` was ever appended, no extractor was ever spawned, and
the dashboard reported zero Cursor sessions. CodeBuddy / Claude / VSCode
adapters were unaffected. Root cause: ``tests/test_init_command.py``
fixtures wrote stub bytes into ``src/agent_cot/assets/hooks/cursor/``
when those files weren't already present, the real bytes got cleaned out
during a pre-build sweep, and the next ``python -m build`` happily
packaged the stubs into the released wheel.

0.20.5 closes the loop with three reinforcing fixes:

1. **Test isolation** — ``tests/test_init_command.py``'s ``isolated_home``
   fixture now monkey-patches ``_assets.hooks_dir()`` to a per-test
   ``tmp_path`` mirror. Stub-writing test fixtures still work, but the
   stubs land in ``tmp_path/assets/hooks/`` instead of the real source
   tree. The source tree is no longer reachable from any test.

2. **Pre-build sanity check** — new ``python -m agent_cot._build_assets
   check`` walks ``src/agent_cot/assets/hooks/**/*.{js,py}`` and refuses
   to certify any file < 1 KB. Wire it into the release script
   (``check && build``) and stubs can never escape into a wheel again.

3. **Install-time stub guard** — ``commands/init.apply_plan`` now
   refuses to copy a hook file whose source bytes are < 1 KB. Even if a
   future packaging glitch slips a stub past the build, the user-visible
   failure mode flips from "silent zero Cursor sessions" to a loud
   ``RuntimeError("bundled hook asset looks like a stub: ... (NN bytes).
   This is a packaging bug — reinstall agent-cot from a newer wheel.")``.

Upgrade: colleagues currently on 0.20.4 with a non-working Cursor
adapter need exactly one command::

    pip install --upgrade observation-agent==0.20.5
    agent-cot init --apply --agent cursor

Nothing else changes — no re-doing CodeBuddy / Claude, no editing
settings.json. The Cursor hook scripts on disk get rewritten with the
real bytes and the next Cursor session starts producing trace.

────────────────────────────────────────────────────────────────────────

v0.20.4 — one-click Claude OTel env auto-injection (Claude only).

  Before 0.20.4, ``agent-cot init --apply --agent claude`` only registered
  the 27 hooks into ``~/.claude/settings.json``. To get the native OTel
  channel (the "📡 Claude Code 原生 OTel" panel in the dashboard, prompt-id
  timeline, API-true cost / TTFT / cache R/W metrics), every colleague had
  to manually open settings.json and paste an ``env`` block with eight
  ``OTEL_*`` keys —— and the hardcoded ``http://localhost:8765`` we
  documented would silently drop events whenever 8765 was already taken
  on the colleague's box.

  0.20.4 closes this gap with three reinforcing layers, all gated on the
  Claude adapter (Cursor / CodeBuddy / VSCode adapters are unaffected):

  1. **init writes env** —— ``ClaudeAdapter.recommended_env(backend_port=...)``
     builds the env dict at the actual port that ``init`` just locked in via
     ``pick_port``. The new ``merge_claude_env`` performs fill-missing-only
     merging: if the colleague's settings.json already has e.g.
     ``OTEL_EXPORTER_OTLP_ENDPOINT=https://otel.corp.example.com:4318`` or a
     custom ``OTEL_RESOURCE_ATTRIBUTES`` for team tagging, those values are
     preserved verbatim and surfaced in CLI output as ``env kept`` rows.
     Default behaviour is on; ``agent-cot init --apply --agent claude
     --no-otel-env`` opts out entirely for users managing OTel out-of-band.

  2. **start self-heals the endpoint port** —— if 8765 was free at ``init``
     time but occupied by the time of ``agent-cot start`` (Zoom, VPN
     client, another local dev server, …), ``pick_port`` lands on 8766.
     Before 0.20.4 the cached env inside Claude Code would still push
     OTel data at 8765 → some other process. 0.20.4's
     ``_self_heal_claude_otel_endpoint`` reads settings.json,
     detects loopback URL with the wrong port, atomically rewrites it
     to the live port, and prints ``claude OTel : endpoint rewritten``
     in the CLI plus a yellow "fully quit Claude Code and reopen" hint.
     Non-loopback URLs (corporate collector, langfuse, phoenix) are
     detected and skipped with status ``foreign`` —— the user explicitly
     set those, agent-cot never overrides them.

  3. **doctor visibility** —— ``check_claude_otel_env`` enters the default
     check list. It walks the same decision tree as the production code:
     SKIP when Claude not installed; WARN when env block missing / partial
     keys / telemetry disabled / protocol=protobuf (claude_otel_receiver
     only parses http/json) / loopback port doesn't match config; OK with
     ``user-managed endpoint`` marker when foreign; OK when the full
     pipeline is consistent. Every WARN carries an actionable hint that
     names the exact ``init`` / ``start`` command to re-run.

  Upgrade path: a colleague on 0.20.3 running ``pip install --upgrade
  observation-agent==0.20.4`` and then ``agent-cot init --apply --agent
  claude`` will see exactly one round of env writes; the regression test
  ``test_claude_second_init_with_only_env_changes_still_writes`` pins this
  (the CLI's early-return into refresh-only mode now also requires
  ``not plan.env_additions``, so the 0.20.4 env keys actually get written
  even when the 27 hooks were already in place from 0.20.3).

  Privacy / surface-area: the recommended env enables
  ``OTEL_LOG_USER_PROMPTS=1`` and ``OTEL_LOG_TOOL_DETAILS=1`` by default
  —— this is what gives the dashboard's right-panel its rich prompt /
  tool timeline. All OTel data lands on the colleague's local box at
  ``~/.claude/state/otel/<sid>/`` and is consumed only by the local
  FastAPI receiver; nothing is uploaded anywhere unless the user
  explicitly runs ``agent-cot otlp send``. Privacy-sensitive users can
  flip either flag to ``"0"`` post-install (fill-missing semantics
  preserve that override forever) or run init with ``--no-otel-env``.

  Cursor + CodeBuddy adapters: unchanged. They don't expose
  ``recommended_env`` so the init path duck-types through cleanly with
  ``otel_env_enabled=False``; cursor's hooks.json (top-level ``hooks``
  array) and codebuddy's settings.json schemas are untouched.

v0.19.6 prior — three CodeBuddy regressions actually fixed (0.19.4 release notes claimed them but code never landed).

  Three user-visible bugs (all surfaced by a single screenshot of a fresh
  CodeBuddy session in the dashboard):

  * **(Bug 1) CodeBuddy Plan card empty + tool names are snake_case.**
    The 0.19.4 release notes claimed ``_normalize_codebuddy_tool_name`` /
    ``_coerce_codebuddy_todos`` had landed and that
    ``_extract_codebuddy_session_from_transcript`` was calling
    ``_build_plan_timeline`` before returning. Audit shows those helpers
    never existed in the code (only in CHANGELOG.md + __init__.py). Real
    fix now in ``assets/cot-extractor[-src]/cot_extractor.py``:
      - New ``_CODEBUDDY_TO_CURSOR_TOOL`` dict + ``_normalize_codebuddy_
        tool_name`` helper map ``read_file`` / ``write_to_file`` /
        ``replace_in_file`` / ``execute_command`` / ``search_content`` /
        ``search_file`` / ``list_files`` / ``todo_write`` to ``Read`` /
        ``Write`` / ``Edit`` / ``Shell`` / ``Grep`` / ``Glob`` / ``LS`` /
        ``TodoWrite``. CodeBuddy-only tools (``task``,
        ``attempt_completion`` …) keep their original name —
        per user requirement "equivalent things unified, unique things
        preserved".
      - ``_coerce_codebuddy_todos`` parses ``tool_input.todos`` from a
        JSON-encoded string into a ``List[Dict]`` when the call is
        ``TodoWrite``-equivalent — without this, ``_build_plan_timeline``
        still skips every CodeBuddy snapshot because it requires a list.
      - Both helpers wire into ``_extract_codebuddy_session_from_
        transcript`` at the tool_decision *and* tool_execution build
        sites, so D-E pairs use matching names and ``tool_calls_total``
        / ``tool_call_distribution`` come out PascalCase too. Original
        raw name kept in ``step.metadata.tool_name_raw`` for audit.
      - The CodeBuddy extractor now explicitly calls
        ``_build_plan_timeline`` + ``_build_mode_transitions`` before
        the early ``return`` (line ~5482 in run_extract bypasses the
        main pipeline that does this for Cursor/Claude), so the
        plan_timeline / mode_transitions / plan_proposals fields are
        actually populated for CodeBuddy sessions.

  * **(Bug 2) CodeBuddy "本轮输入 / 本轮输出" tokens both 0.** Not a
    code regression — the stale ``cot.json`` was produced by an earlier
    extract pass when the index.json hadn't yet been finalised with
    usage data (CodeBuddy writes ``requests[i].usage`` only after the
    LLM call settles). Re-running ``extract_cot.py --session-id <sid>``
    against the now-final index.json produces the correct
    ``turn.usage = {input_tokens: 740769, output_tokens: 10358, …}``.
    No code fix; just unblocking the path by removing stale cot.json
    so the auto-extract regenerates fresh tokens. (The hook's 30 s
    debounce was the trigger for the original write — if a user wants
    to force a refresh, ``rm ~/.agent-cot/data/cot/<sid>_cot.json``
    then re-trigger any Stop/SessionEnd event.)

  * **(Bug 3) DetailPanel right-side "hero" header shows tool name
    list ("todo_write · task · task · search_file · …") instead of the
    user's prompt.** Root cause in
    ``agent-dashboard/frontend/src/components/DetailPanel.tsx``:
    the hero title fallback chain was
    ``turn.interaction_summary || turn.tool_calls.join(' · ') || …``,
    but CodeBuddy's extractor doesn't write ``interaction_summary``
    (only Cursor/Claude paths do, via ``_summarize_interaction``).
    So the JOIN of tool_calls was getting rendered as the header for
    every CodeBuddy turn. Fix: insert ``turn.user_query`` (truncated
    to 120 chars) as the priority fallback after
    ``interaction_summary``. Cursor / Claude / Claude-Code paths are
    unchanged — they still hit the ``interaction_summary`` branch as
    before.

v0.19.5 — Claude Internal Plan rendering + SpanTree scroll-jump fix.

  Two user-visible bugs (Claude Internal only):

  * **Claude Internal Plan card never rendered.** Root cause: Claude
    Internal does NOT use ``TodoWrite``; it uses a different pair of
    tools to maintain its task list:
      - ``TaskCreate {subject, description, activeForm}``  — append a
        new task. task_id is implicit (IDE-side counter, "1"-based).
      - ``TaskUpdate {taskId, status}``  — patch an existing task.
    ``_build_plan_timeline`` previously only scanned for ``TodoWrite``,
    so Claude Internal sessions ended with empty ``plan_timeline`` and
    the frontend Plan card never rendered.

    Fix in ``assets/cot-extractor[-src]/cot_extractor.py``:
      - New ``_build_plan_timeline_from_task_tools`` mirrors the
        TodoWrite path's PlanSnapshot output but folds an incremental
        TaskCreate/TaskUpdate trace into the same shape (so the
        frontend code stays untouched).
      - Falls through from ``_build_plan_timeline`` only when the
        TodoWrite scan returns nothing (no regression risk for Cursor
        / CodeBuddy / Claude Code).
      - Per user requirement, Task* tools are NOT renamed to
        TodoWrite — they're semantically different (incremental vs
        whole-list replace), so unifying them would lose information.
      - ``_reconcile_plan_timeline`` extended to recognise Task* as
        "valid plan progress" alongside TodoWrite (otherwise Claude
        sessions would always be flagged stale).
      - ``_safe_tool_input`` hardened: when ``tool_input`` arrives as
        a JSON-encoded string (occasionally seen in streamed Claude
        / CodeBuddy traces), it now attempts ``json.loads`` instead
        of dropping the dict.

  * **SpanTree click on Subagent / PermissionRequest / Notification
    scrolls to top.** Root cause: Claude hook events are surfaced in
    SpanTree as virtual ThoughtSteps with ``step_index = -1_000_000 -
    virtIdx`` where ``virtIdx`` resets to 0 in every turn. The scroll
    logic in ``SpanTree.tsx`` used
      ``document.querySelector('[data-step-index="<n>"]')``
    which returns the first match in document order — so clicking the
    second turn's first hook event would scroll to the first turn's
    first hook event (i.e. "the top"). Fix in
    ``agent-dashboard/frontend/src/components/SpanTree.tsx``:
      ``selector = '[data-turn-index="X"] [data-step-index="Y"]'``
    scoping the query to the selected step's turn subtree, eliminating
    the cross-turn collision. Also encodes ``turn_index`` into the
    de-dupe key so repeat-click-different-turn-same-step is not
    suppressed by the ``lastSelKeyRef`` short-circuit.

v0.19.4 — structural consistency pass + CodeBuddy Plan/tool-name fixes.

  Two user-visible bugs:

  * **CodeBuddy Plan card never rendered.** Triple root cause:
    (a) the tool name was ``todo_write`` but every frontend renderer
    (SpanTree, DetailPanel) hardcodes ``TodoWrite`` (Cursor's
    PascalCase convention);
    (b) the ``todos`` argument arrived as a JSON-encoded string but
    ``_build_plan_timeline`` only accepted ``list``;
    (c) the CodeBuddy code path took an early ``return`` in
    ``extract_session_cot`` and never reached the
    ``_build_plan_timeline`` / ``_build_mode_transitions``
    invocation site at all.
    Fix layers in ``assets/cot-extractor/src/cot_extractor.py``:
    ``_normalize_codebuddy_tool_name`` + ``_coerce_codebuddy_todos``
    + an explicit ``_build_plan_timeline`` call inside
    ``_extract_codebuddy_session_from_transcript`` before it returns.

  * **Tool name drift.** ``read_file`` / ``write_to_file`` /
    ``replace_in_file`` / ``delete_file`` / ``execute_command`` /
    ``todo_write`` / ``search_content`` / ``search_file`` are now
    rewritten to Cursor's ``Read`` / ``Write`` / ``Edit`` /
    ``Delete`` / ``Shell`` / ``TodoWrite`` / ``Grep`` / ``Glob`` so
    the same operation gets the same renderer treatment. Operations
    Cursor doesn't have an equivalent for (``list_dir``) keep their
    original CodeBuddy name — user explicit rule was "equivalent
    things unified, unique things preserved". The original raw name
    is kept in ``step.metadata.tool_name_raw`` for audit.

  Structural consistency fixes (mandatory & recommended from the
  0.19.3 audit; numbered P-1..P-13):

  * P-1: ``commands/otlp_bridge.py:_candidate_cot_dirs`` now scans
    ``AGENT_COT_DATA_ROOT`` env + ``runtime.json.data_root`` + legacy
    user roots, not only the developer-checkout ``<repo>/output/
    cot/``. Without this, ``agent-cot otlp send <sid>`` had no way to
    find user-generated cot.json on a normal wheel install.
  * P-2: ``assets/cot-extractor/scripts/export_otlp.py`` gets the
    same ``_candidate_cot_dirs`` treatment so the standalone CLI
    matches the bridge.
  * P-3: ``assets/cot-extractor/scripts/backfill_results.py``
    defaults ``--cot-root`` to the resolved user data root and
    accepts the new ``<root>/events/`` layout (plus the legacy
    ``<root>/output/events/`` for source checkouts).
  * P-4: ``otlp_bridge._candidate_src_dirs`` reads
    ``runtime.json.cot_extractor_root`` and falls back to the wheel-
    bundled vendor copy.
  * P-5: ``assets/backend/main.py:_resolve_cot_extractor_src`` reads
    ``runtime.json.cot_extractor_root`` before the sibling-repo and
    vendor-fallbacks.
  * P-6: ``_load_cursor_events`` reads ``runtime.json.data_root``
    AND ``runtime.json.cot_extractor_root`` before relying on
    ``Path(__file__).parent.parent`` (which under a wheel install
    pointed inside ``agent_cot/assets/`` — nowhere near
    cot-extractor).
  * P-7: ``assets/cot-extractor/scripts/cot_hook.py`` consults
    ``runtime.json.cot_extractor_root`` for its ``sys.path``
    injection.
  * P-8: ``assets/backend/config.py:_default_data_dir`` honours the
    env + runtime.json chain even in non-bundled mode so a source
    checkout with a custom data root behaves the same as a wheel
    install.
  * P-9: ``installer/upgrader.py:apply_upgrade_plan`` now forwards
    the previously-stored ``data_root`` / ``python_executable`` into
    the new ``runtime.json`` instead of letting an upgrade silently
    overwrite a user's custom data root with the package default.
  * P-10: ``installer/uninstaller.py`` recognises the nested
    CodeBuddy / Claude / Copilot hook schemas (``{matcher, hooks:
    [{type, command}]}``), no longer leaving stale absolute paths
    inside ``settings.json`` after an uninstall.
  * P-11: ``pyproject.toml`` ships ``*.md`` from
    ``assets/backend/`` so ``CLAUDE_OTEL.md`` (and any future docs)
    arrive via wheel install instead of being a clone-only file.
  * P-12: ``doctor/checks.py`` adds
    ``check_codebuddy_settings_json`` and ``check_claude_settings_
    json`` to the default check list so the report covers all three
    IDEs by default, not only Cursor.
  * P-13: ``agent-cot upgrade`` defaults to ``--agent all`` and
    fans out to every adapter, so a fresh ``pip install -U`` on a
    multi-IDE machine refreshes every hook, not only Cursor's.

  Cursor parity: this release re-asserts the ``env > runtime.json
  (self-healing) > default`` ordering established in 0.19.3 for the
  hook layer and propagates it to every backend / CLI surface above.

v0.19.3 — fix the data-root divergence that hid CodeBuddy / Claude sessions.

  Symptom (reproduced on this developer's machine after a fresh 0.19.2 install):
  Cursor sessions kept showing up on the dashboard, but CodeBuddy and
  Claude Internal sessions never appeared even though their hooks fired,
  ``events.jsonl`` was being written, and ``extract_cot.py`` exited 0.
  Manual ``GET /api/sessions`` returned only the cursor entries.

  Root cause: a semantic mismatch in how ``data_root`` was persisted and
  consumed across processes —

  * ``commands/init.py`` wrote ``data_root = ~/.agent-cot`` into
    ``runtime.json`` (the package root), but every other component
    treated ``data_root`` as the *data* directory and expected
    ``~/.agent-cot/data``.
  * The CodeBuddy / Claude hooks dutifully read ``runtime.json`` for
    ``data_root`` (correct behaviour added in 0.19.0+ for cross-process
    consistency), so they wrote ``events.jsonl`` and ``cot.json`` to
    ``~/.agent-cot/cot`` / ``~/.agent-cot/events`` — the WRONG place.
  * ``assets/backend/config.py`` ignored ``runtime.json`` and defaulted to
    ``~/.agent-cot/data/cot`` — the RIGHT place. So the scanner never
    saw the misplaced files.
  * Cursor's ``cot-bridge.js`` "accidentally" worked because it had a
    hard-coded fallback that pointed to ``~/.agent-cot/data`` (matching
    the backend), not the broken value from ``runtime.json``. This
    explained the cross-IDE behavioural drift the user observed.

  Three reinforcing layers fix it:

  1. **Source-of-truth fix** — ``commands/init.py`` now appends ``/data``
     when persisting ``data_root`` into ``runtime.json``, so the JSON
     value matches the on-disk filesystem layout used by hooks and
     extractor.
  2. **Backend defence-in-depth** — ``assets/backend/config.py`` now
     reads ``data_root`` from ``runtime.json`` first
     (env > runtime.json > default), aligning the backend with the
     hooks. AND it exposes ``COT_SCAN_DIRS`` covering legacy locations
     (``~/.agent-cot/cot``, ``~/.agent-cot/data/cot``) so the
     ``session_scanner`` still finds files left behind by 0.19.0~0.19.2
     installs — zero-friction migration.
  3. **Hook self-healing** — ``cot-stream-codebuddy.js``, ``cot-stream.js``,
     ``claude_stream_hook.py``, AND ``cot-bridge.js`` all gain
     ``_normalizeDataRoot`` / ``_normalize_data_root``. If they read
     ``data_root = .../.agent-cot`` from ``runtime.json`` (no ``/data``
     suffix), they transparently write to ``.../.agent-cot/data``
     anyway. Future misconfigurations are absorbed silently.
  4. **Cursor consistency** — ``cot-bridge.js`` no longer hard-codes
     ``~/.agent-cot/data`` as its fallback. It now resolves
     ``AGENT_COT_DATA_ROOT`` env → ``runtime.json`` (self-healing) →
     ``~/.agent-cot/data`` default — exactly the same chain as the
     other three IDEs. This eliminates the cross-IDE drift.

  All v0.19.2 / v0.19.1 functionality is preserved.

v0.19.2 prior — close the CodeBuddy auto-extract gap.

  * ``cot-stream-codebuddy.js`` now spawns ``extract_cot.py`` in a detached
    background process on Stop / SessionEnd / SubagentStop / StopFailure,
    mirroring cursor's ``cot-bridge.js`` and v0.19.1's ``claude_stream_hook.py``.
  * Same 4-layer path resolution (env > ~/.agent-cot/runtime.json
    > ~/.cursor-cot/runtime.json > probes) and 30 s debounce
    (``<data_root>/extract_debounce_codebuddy.json``).
  * Before this fix, fresh installs without a transcript_watcher would write
    ``events.jsonl`` correctly but never produce ``cot.json``, so the dashboard
    SessionList stayed empty. This was the same class of bug as the v0.19.1
    Claude Internal fix.
  * Cursor + Claude paths are unchanged from v0.19.1.

v0.19.1 prior — incremental Claude Internal integration on top of v0.19.0.

  * ``ClaudeAdapter`` is now fully implemented (was a stub through 0.19.0):
    27-event registration into ``~/.claude/settings.json``, idempotent
    merge that leaves third-party hooks (codebuddy-mem, langfuse_hook,
    user one-liners) untouched, and a dedicated nested-schema merger.
  * Bundled ``claude_stream_hook.py`` ships the v0.19.1 self-bootstrap:
    same 4-layer fallback chain as Cursor / CodeBuddy
    (env > ``.agent-cot/runtime.json`` > ``.cursor-cot/runtime.json``
    > sys.executable probe) PLUS an in-hook ``_maybe_trigger_extract``
    that backgrounds ``extract_cot.py`` on Stop / SubagentStop /
    SessionEnd / StopFailure events with 30-second debouncing — so a
    pip-installing colleague gets cot.json materialised automatically
    without any manual ``cron`` / extractor wiring.
  * ``agent-cot start`` now self-heals all THREE IDEs (cursor + codebuddy
    + claude) on every invocation, refreshing on-disk hook bytes from
    the currently-installed wheel.
  * ``agent-cot doctor --deep`` adds ``check_claude_hook_alive`` —
    detects pre-v0.19.1 hooks (no extract trigger), missing
    ``settings.json`` registrations, and broken fallback chains.

v0.19.0 prior — self-heal parity with cursor-cot-observer 0.18.15 +
CodeBuddy hardening. Hooks write ``~/.agent-cot/runtime.json`` on every
``init`` / ``upgrade`` / ``start``, and on-disk hook scripts read it
as a secondary fallback when the install-time literal no longer
resolves. Frontend bundle ships the cursor-cot-observer 0.18.15 build
(Plan diff, Pencil chips, Thinking-phase fold, Batch-collapse).
"""

from __future__ import annotations

__version__ = "1.0.22"
__all__ = ["__version__"]
