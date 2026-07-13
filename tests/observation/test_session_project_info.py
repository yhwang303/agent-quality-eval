from __future__ import annotations

import sys
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "src" / "agent_cot" / "assets" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services import session_scanner  # noqa: E402
from services.session_scanner import _extract_project_info  # noqa: E402


def test_project_info_prefers_workspace_root() -> None:
    info = _extract_project_info({"session_meta": {"workspace_roots": [r"D:\agent-quality-eval"]}})

    assert info["project_name"] == "agent-quality-eval"
    assert info["project_path"] == "D:/agent-quality-eval"
    assert info["project_source"] == "workspace_roots"
    assert info["project_id"]


def test_project_info_falls_back_to_transcript_project_slug() -> None:
    info = _extract_project_info(
        {
            "session_meta": {
                "transcript_path": r"C:\Users\me\.claude\projects\D--SST\session.jsonl",
            }
        }
    )

    assert info["project_name"] == "SST"
    assert info["project_source"] == "transcript_path"
    assert info["project_id"]


def test_project_info_keeps_drive_only_virtual_project() -> None:
    info = _extract_project_info(
        {
            "session_meta": {
                "transcript_path": r"C:\Users\me\.claude-internal\projects\D--\session.jsonl",
            }
        }
    )

    assert info["project_name"] == "D"
    assert info["project_source"] == "transcript_path"


def test_project_info_handles_uplink_workspace_metadata() -> None:
    info = _extract_project_info(
        {
            "owner": "teammate",
            "source": "uplink",
            "session_meta": {"workspace_roots": ["/home/teammate/work/research-agent"]},
        }
    )

    assert info["project_name"] == "research-agent"
    assert info["project_source"] == "workspace_roots"


def test_project_info_resolves_codebuddy_workspace_hash(monkeypatch) -> None:
    project_path = r"D:\claudecode"
    workspace_hash = hashlib.md5(project_path.lower().encode("utf-8")).hexdigest()

    monkeypatch.setattr(
        session_scanner,
        "_local_project_hash_index",
        lambda: {workspace_hash: project_path.replace("\\", "/")},
    )

    info = _extract_project_info(
        {
            "agent_type": "codebuddy",
            "session_meta": {
                "transcript_path": rf"C:\Users\me\AppData\Local\CodeBuddyExtension\Data\u\CodeBuddyIDE\u\history\{workspace_hash}\abc123abc123abc123abc123abc123ab\index.json",
            },
        }
    )

    assert info["project_name"] == "claudecode"
    assert info["project_path"] == "D:/claudecode"
    assert info["project_source"] == "codebuddy_workspace_hash"


def test_project_info_resolves_events_only_cwd(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"event":"SessionEnd","brief_input":{"cwd":"d:/claudecode"},"payload":{"cwd":"d:/claudecode"}}\n',
        encoding="utf-8",
    )

    info = _extract_project_info({"agent_type": "codebuddy", "transcript_path": str(events)})

    assert info["project_name"] == "claudecode"
    assert info["project_path"] == "d:/claudecode"
    assert info["project_source"] == "event_payload"


def test_project_info_unknown_project_fallback() -> None:
    info = _extract_project_info({})

    assert info == {
        "project_name": "Unknown Project",
        "project_path": "",
        "project_id": "unknown-project",
        "project_source": "fallback",
    }
