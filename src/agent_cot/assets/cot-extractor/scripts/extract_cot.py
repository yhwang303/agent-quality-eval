#!/usr/bin/env python3
"""
CoT 手动提取脚本（CLI）

用于手动提取指定 session 的 CoT，生成报告，可选上传到 Langfuse。

用法：
  # 提取最近的 session
  python extract_cot.py

  # 指定 session ID
  python extract_cot.py --session-id <session_id>

  # 指定 transcript 文件路径
  python extract_cot.py --transcript <path/to/session.jsonl> --session-id <session_id>

  # 提取最近 N 个 session
  python extract_cot.py --recent 3

  # 只生成报告，不上传 Langfuse
  python extract_cot.py --no-upload

  # 查看某个 session 的 CoT 报告
  python extract_cot.py --session-id <session_id> --show
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Windows 控制台 GBK 编码兼容：把 stdout/stderr 切成 UTF-8（避免 emoji 崩溃）
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─── 路径配置 ─────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_DIR / "src"


def _resolve_output_dir() -> Path:
    """决定 cot.json / reports / sessions 写到哪。

    优先级（v0.18.2，新增前两条）:

    1. ``--output`` CLI 参数  —— main() 里再覆盖
    2. ``AGENT_COT_DATA_ROOT`` 环境变量  —— 跟 backend / agent-cot CLI 对齐
    3. ``COT_OUTPUT_DIR`` 环境变量  —— 给老脚本兜底
    4. wheel 安装态：``cot-extractor/scripts/extract_cot.py`` 位于
       ``<site-packages>/agent_cot/assets/cot-extractor/scripts/`` 下，
       ``_PROJECT_DIR/output/`` 落在 site-packages 里 —— 既不该写、写了
       backend 也不会扫 —— 兜底改 ``~/.agent-cot/data/``。
    5. 源码态：保留历史行为 ``cot-extractor/output/``。

    没有这一段，pip 安装的同事跑 hook 后 cot.json 写进 site-packages，
    backend 默认扫 ``~/.agent-cot/data/cot/`` 永远是空 → SESSIONS 0。
    """
    env_root = os.environ.get("AGENT_COT_DATA_ROOT") or os.environ.get("COT_OUTPUT_DIR")
    if env_root:
        return Path(env_root).expanduser()

    project_str = str(_PROJECT_DIR).replace("\\", "/").lower()
    looks_bundled = (
        "site-packages" in project_str
        or "/agent_cot/assets/" in project_str
        or "\\agent_cot\\assets\\" in str(_PROJECT_DIR).lower()
    )
    if looks_bundled:
        return Path.home() / ".agent-cot" / "data"

    return _PROJECT_DIR / "output"


_OUTPUT_DIR = _resolve_output_dir()

# 将 src 目录加入 Python 路径
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cot_extractor import extract_session_cot
from cot_reporter import save_reports, generate_markdown_report
from cot_uploader import create_langfuse_client, upload_session_cot

# v0.18.0: CoT Uplink — 仅当 AGENT_COT_UPLINK_URL 配置时才生效（异步 + 静默失败）
try:
    from cot_uplink import uplink_session_cot
except ImportError:
    def uplink_session_cot(*_args, **_kwargs):  # type: ignore
        return False


# ─── 工具函数 ─────────────────────────────────────────────

def _candidate_projects_dirs() -> list:
    """
    返回所有 IDE 的 projects/transcript 根目录。

    兼容：
    - Claude Code:      ~/.claude-internal/projects/<slug>/<uuid>.jsonl
    - Claude OSS:       ~/.claude/projects/<slug>/<uuid>.jsonl
    - Cursor:           ~/.cursor/projects/<slug>/agent-transcripts/<uuid>/<uuid>.jsonl
    - CodeBuddy (legacy guess): ~/.codebuddy/**/<uuid>.jsonl
      (CodeBuddyExtension's real on-disk layout is handled separately
      via :func:`_find_codebuddy_index_json`, since it lives under
      %LOCALAPPDATA%/CodeBuddyExtension/Data/.../history/.../<sid>/index.json
      not ~/.codebuddy.)
    """
    dirs = []
    for dirname in (".claude-internal", ".claude-inertnal", ".claude"):
        p = Path.home() / dirname / "projects"
        if p.exists():
            dirs.append(p)
    cursor_root = Path.home() / ".cursor" / "projects"
    if cursor_root.exists():
        for proj in cursor_root.iterdir():
            at = proj / "agent-transcripts"
            if at.exists():
                dirs.append(at)
    codebuddy_root = Path.home() / ".codebuddy"
    if codebuddy_root.exists():
        for child in ("sessions", "projects", "transcripts"):
            p = codebuddy_root / child
            if p.exists():
                dirs.append(p)
    return dirs


def _find_codebuddy_index_json(session_id: str) -> Path | None:
    """Locate a CodeBuddy native transcript ``index.json`` for ``session_id``.

    Walks the same data roots that ``codebuddy_transcript`` knows about
    (``%LOCALAPPDATA%/CodeBuddyExtension/Data/...``). Returning the
    ``index.json`` (rather than its parent dir) keeps the rest of this
    CLI's flow uniform — ``extract_session_cot`` already knows to detect
    a CodeBuddy index and dispatch to the rich parser.
    """
    try:
        from codebuddy_transcript import find_transcript_by_session_id  # type: ignore
    except Exception:
        return None
    return find_transcript_by_session_id(session_id)


def _find_events_only_path(session_id: str) -> Path | None:
    """Return an events.jsonl path when a session has hooks but no transcript."""
    events_root = _OUTPUT_DIR / "events"
    candidates = [
        events_root / session_id / "events.jsonl",
        events_root / f"codebuddy-{session_id}" / "events.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_transcript(session_id: str) -> Path:
    """在所有已知 IDE 的 transcript 目录下查找指定 session 的 jsonl 文件"""
    for projects_dir in _candidate_projects_dirs():
        for f in projects_dir.rglob(f"{session_id}.jsonl"):
            return f
    # CodeBuddyExtension keeps its history far away from ~/.codebuddy; check
    # there before we give up to events-only mode. This is what unlocks the
    # rich transcript path for CodeBuddy sessions started via the IDE.
    cb_index = _find_codebuddy_index_json(session_id)
    if cb_index is not None:
        return cb_index
    events_path = _find_events_only_path(session_id)
    if events_path is not None:
        return events_path
    raise FileNotFoundError(f"找不到 session {session_id} 的 transcript 文件")


def find_recent_transcripts(n: int = 5) -> list:
    """跨 Claude Code / Cursor 查找最近 N 个 transcript 文件"""
    all_transcripts = []
    for projects_dir in _candidate_projects_dirs():
        for f in projects_dir.rglob("*.jsonl"):
            all_transcripts.append(f)
    all_transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_transcripts[:n]


def process_session(
    transcript_path: Path,
    session_id: str,
    upload: bool = True,
    show: bool = False,
    keep_history: bool = False,
) -> bool:
    """处理单个 session"""
    print(f"\n{'='*60}")
    print(f"Session: {session_id[:16]}...")
    print(f"Transcript: {transcript_path}")
    print(f"{'='*60}")

    # 提取 CoT
    print("正在提取 CoT...")
    session_cot, _ = extract_session_cot(
        transcript_path=transcript_path,
        session_id=session_id,
        offset=0,  # CLI 模式全量提取
    )

    if session_cot is None:
        print("❌ 没有提取到任何内容（transcript 可能为空）")
        return False

    print(f"✅ 提取完成:")
    print(f"   Turns: {len(session_cot.turns)}")
    print(f"   总工具调用: {session_cot.total_tool_calls}")
    print(f"   总策略转换: {session_cot.total_strategy_shifts}")
    print(f"   总思考步骤: {session_cot.total_thinking_steps}")
    print(f"   平均复杂度: {session_cot.avg_complexity}")

    # 工具分布
    if session_cot.tool_call_distribution:
        print(f"   工具分布: {dict(sorted(session_cot.tool_call_distribution.items(), key=lambda x: x[1], reverse=True))}")

    # 保存报告
    reports_dir = _OUTPUT_DIR / "reports"
    cot_dir = _OUTPUT_DIR / "cot"
    report_paths = save_reports(session_cot, reports_dir)
    cot_dir.mkdir(parents=True, exist_ok=True)
    cot_json_path = cot_dir / f"{session_id}_cot.json"
    cot_dict = session_cot.to_dict()
    cot_payload = json.dumps(cot_dict, indent=2, ensure_ascii=False)
    cot_json_path.write_text(cot_payload, encoding="utf-8")
    print(f"📄 报告已保存:")
    print(f"   Markdown: {report_paths['md']}")
    print(f"   JSON:     {report_paths['json']}")

    # v0.18.0: 中央上行（异步、失败静默；未配 URL 则什么也不做）
    if uplink_session_cot(cot_dict, session_id):
        if os.environ.get("AGENT_COT_UPLINK_URL"):
            print(f"   Uplink:   dispatched -> {os.environ['AGENT_COT_UPLINK_URL']}")

    # 历史快照：同一 session 每次 stop 事件都留一份带时间戳的 CoT 存档，
    # 用于事后回溯这个 session 的演进过程（主文件只保留最新累积版）。
    if keep_history:
        snapshot_dir = _OUTPUT_DIR / "sessions" / session_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        snapshot_path = snapshot_dir / f"{ts}_cot.json"
        snapshot_path.write_text(cot_payload, encoding="utf-8")
        print(f"   Snapshot: {snapshot_path}")

    # 显示报告
    if show:
        print("\n" + "="*60)
        print(generate_markdown_report(session_cot))

    # 上传到 Langfuse
    if upload and os.environ.get("TRACE_TO_LANGFUSE", "").lower() == "true":
        print("正在上传 CoT 到 Langfuse...")
        client = create_langfuse_client()
        if client:
            ok = upload_session_cot(session_cot, session_id, client)
            if ok:
                print("✅ CoT 上传成功")
            else:
                print("❌ CoT 上传失败（查看日志了解详情）")
        else:
            print("⚠️  无法创建 Langfuse 客户端（检查 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY）")
    elif upload:
        print("⚠️  TRACE_TO_LANGFUSE 未设置为 true，跳过上传")

    return True


# ─── CLI 主函数 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoT 提取工具 — 从 Claude Code transcript 提取思维链",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--session-id", type=str, help="指定 session ID")
    parser.add_argument("--transcript", type=str, help="指定 transcript .jsonl 文件路径")
    parser.add_argument("--recent", type=int, default=1, help="提取最近 N 个 session（默认 1）")
    parser.add_argument("--no-upload", action="store_true", help="只生成报告，不上传 Langfuse")
    parser.add_argument("--show", action="store_true", help="在终端显示 Markdown 报告")
    parser.add_argument("--output", type=str, help="指定输出目录")
    parser.add_argument("--history", action="store_true",
                        help="每次运行额外保留一份带时间戳的 CoT 快照（output/sessions/<sid>/<ts>_cot.json）")
    args = parser.parse_args()

    global _OUTPUT_DIR
    if args.output:
        _OUTPUT_DIR = Path(args.output)

    upload = not args.no_upload

    if args.transcript and args.session_id:
        # 指定文件模式
        transcript_path = Path(args.transcript).expanduser().resolve()
        process_session(transcript_path, args.session_id, upload=upload,
                        show=args.show, keep_history=args.history)

    elif args.session_id:
        # 按 session ID 查找
        try:
            transcript_path = find_transcript(args.session_id)
            process_session(transcript_path, args.session_id, upload=upload,
                            show=args.show, keep_history=args.history)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

    else:
        # 自动发现最近 N 个 session
        transcripts = find_recent_transcripts(args.recent)
        if not transcripts:
            print("❌ 未找到任何 transcript 文件")
            sys.exit(1)

        print(f"找到 {len(transcripts)} 个 transcript 文件，开始处理...")
        success = 0
        for tp in transcripts:
            session_id = tp.stem
            ok = process_session(tp, session_id, upload=upload,
                                 show=args.show, keep_history=args.history)
            if ok:
                success += 1

        print(f"\n✅ 完成：{success}/{len(transcripts)} 个 session 处理成功")


if __name__ == "__main__":
    main()
