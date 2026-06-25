#!/usr/bin/env python3
"""
CoT Hook — Stop 事件触发的 CoT 提取 Hook

作为 Claude Code 的 Stop 事件 Hook 运行，
每次 Claude 完成一轮回复后，自动：
  1. 从 transcript 提取 CoT
  2. 上传 CoT span 到 Langfuse（复用 langfuse_hook.py 的 trace_id）
  3. 生成本地 CoT 报告（MD + JSON）

安装方式：
  1. 将本文件复制到 ~/.claude/hooks/cot_hook.py
  2. 在 ~/.claude/settings.json 的 Stop 事件中追加（在 langfuse_hook.py 之后）：
     {
       "type": "command",
       "command": "python \"$HOME/.claude/hooks/cot_hook.py\""
     }

执行顺序（Stop 事件）：
  1. langfuse_hook.py    — 上传基础数据，创建 trace/span
  2. export_transcript.py — 导出 transcript
  3. auto_check.py       — 一致性检测（异步轮询）
  4. cot_hook.py         — 提取并上传 CoT（本脚本）
"""

import json
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ─── 加载 .env 文件（优先 cot-extractor/.env，其次项目根 .env）────
def _load_dotenv_early() -> None:
    """在所有模块导入前尽早加载 .env，确保环境变量可用"""
    _script_dir = Path(__file__).parent  # scripts/
    candidates = [
        _script_dir.parent / ".env",          # cot-extractor/.env
        _script_dir.parent.parent / ".env",   # 项目根 .env
    ]
    try:
        from dotenv import load_dotenv
        for p in candidates:
            if p.exists():
                load_dotenv(p, override=False)
                return
    except ImportError:
        pass
    # 回退：手动解析
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass
        break

_load_dotenv_early()

# ─── Windows stdin 编码修复 ────────────────────────────────
if hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdin, 'buffer'):
    import io
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

# ─── 路径配置 ─────────────────────────────────────────────
#
# v0.18.9: 路径解析必须 100% 跟开发树解耦 —— 同事拿 wheel 装到任意盘符 /
# 任意 site-packages 路径都不能 break。三段式 fallback：
#
#   1. 显式 env (COT_OUTPUT_DIR / COT_EXTRACTOR_ROOT / AGENT_COT_DATA_ROOT)
#   2. 相对脚本自身位置 (``__file__/../output`` 或 ``__file__/../src``) ——
#      wheel 安装到 site-packages/agent_cot/assets/cot-extractor/scripts/
#      时也成立
#   3. 用户级 fallback (``~/.agent-cot/data``) —— 即便 wheel 内的 output/
#      不存在或不可写，也保证 events 能落到一个跨用户安全的位置

_HERE = Path(__file__).resolve().parent
_SCRIPT_REL_OUTPUT = _HERE.parent / "output"
_SCRIPT_REL_SRC = _HERE.parent / "src"
_USER_DATA = Path.home() / ".agent-cot" / "data"

# CoT 输出目录优先级链（绝对不假设 D 盘 / dev tree 存在）
def _resolve_cot_output_dir() -> Path:
    env_explicit = os.environ.get("COT_OUTPUT_DIR", "").strip()
    if env_explicit:
        return Path(env_explicit).expanduser()
    env_data_root = os.environ.get("AGENT_COT_DATA_ROOT", "").strip()
    if env_data_root:
        return Path(env_data_root).expanduser()
    env_extractor = os.environ.get("COT_EXTRACTOR_ROOT", "").strip()
    if env_extractor:
        candidate = Path(env_extractor).expanduser() / "output"
        if candidate.parent.is_dir():
            return candidate
    if _SCRIPT_REL_OUTPUT.parent.is_dir():
        return _SCRIPT_REL_OUTPUT
    return _USER_DATA

COT_OUTPUT_DIR = _resolve_cot_output_dir()

LOG_FILE = Path.home() / ".claude" / "state" / "cot_hook.log"

# ─── 将 src 目录加入 Python 路径 ──────────────────────────
# 同样按"显式 env > script-relative > user-data"三级解析，不硬编码盘符。
def _candidate_src_dirs() -> list:
    out = []
    env_extractor = os.environ.get("COT_EXTRACTOR_ROOT", "").strip()
    if env_extractor:
        out.append(Path(env_extractor).expanduser() / "src")
    env_src = os.environ.get("AGENT_COT_EXTRACTOR_SRC", "").strip()
    if env_src:
        out.append(Path(env_src).expanduser())
    out.append(_SCRIPT_REL_SRC)
    return out

for _src_candidate in _candidate_src_dirs():
    if _src_candidate.is_dir() and str(_src_candidate) not in sys.path:
        sys.path.insert(0, str(_src_candidate))
        break


# ─── 日志 ─────────────────────────────────────────────────

def _log(level: str, msg: str) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} [cot_hook] [{level}] {msg}\n")
    except Exception:
        pass


def info(msg: str) -> None:
    _log("INFO", msg)


def error(msg: str) -> None:
    _log("ERROR", msg)


# ─── 状态管理（增量读取 offset） ──────────────────────────

_STATE_FILE = COT_OUTPUT_DIR / "cot" / ".cot_state.json"


def _load_cot_state() -> Dict:
    try:
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cot_state(state: Dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, _STATE_FILE)
    except Exception as e:
        error(f"save_cot_state failed: {e}")


def _session_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()[:16]


# ─── 主逻辑 ───────────────────────────────────────────────

def run(session_id: str, transcript_path: Path) -> int:
    """
    执行 CoT 提取和上传。

    Args:
        session_id: 会话 ID
        transcript_path: transcript .jsonl 文件路径

    Returns:
        0 成功，1 失败
    """
    info(f"开始 CoT 提取: session={session_id[:16]}, transcript={transcript_path}")

    # 导入模块（延迟导入，避免 import 错误影响 hook 运行）
    try:
        from cot_extractor import extract_session_cot
        from cot_reporter import save_reports
        from cot_uploader import create_langfuse_client, upload_session_cot
    except ImportError as e:
        error(f"导入模块失败: {e}，请检查 src 目录是否存在")
        return 1

    # v0.18.0: CoT Uplink 模块（可选，未配 URL 则 no-op）
    try:
        from cot_uplink import uplink_session_cot
    except ImportError:
        def uplink_session_cot(*_args, **_kwargs):  # type: ignore
            return False

    # 加载增量状态
    state = _load_cot_state()
    key = _session_key(session_id)
    ss = state.get(key, {"offset": 0})
    offset = ss.get("offset", 0)

    # 提取 CoT
    try:
        session_cot, new_offset = extract_session_cot(
            transcript_path=transcript_path,
            session_id=session_id,
            offset=offset,
        )
    except Exception as e:
        error(f"extract_session_cot 失败: {e}")
        import traceback
        error(traceback.format_exc())
        return 1

    if session_cot is None:
        info(f"没有新内容，跳过: session={session_id[:16]}")
        return 0

    info(f"提取完成: {len(session_cot.turns)} turns, "
         f"{session_cot.total_tool_calls} tool calls, "
         f"{session_cot.total_thinking_steps} thinking steps")

    # 注意：LLM CoT 摘要功能已禁用（Gemini/智谱均因 API 限制失败，详见 PLAN.md §9.4）
    # 当前使用纯规则推断的 ReasoningDigest 替代

    # ── 保存报告（简单覆盖模式 + 带时间戳的备份） ──
    try:
        reports_dir = COT_OUTPUT_DIR / "reports"
        cot_dir = COT_OUTPUT_DIR / "cot"
        cot_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 保存 CoT JSON（主文件，直接覆盖）
        cot_data = session_cot.to_dict()
        cot_json_path = cot_dir / f"{session_id}_cot.json"
        cot_json_path.write_text(
            json.dumps(cot_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # 同时保存到 reports 目录
        report_json_path = reports_dir / f"{session_id}_cot.json"
        report_json_path.write_text(
            json.dumps(cot_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # 生成 Markdown 报告
        try:
            report_paths = save_reports(session_cot, reports_dir)
            info(f"报告已保存: {report_paths.get('md', 'N/A')}")
        except Exception as e:
            error(f"save_reports 失败: {e}")

        info(f"CoT 已保存: {session_id}")

        # v0.18.0: 中央上行（异步、失败静默；未配 URL 则什么也不做）
        try:
            if uplink_session_cot(cot_data, session_id):
                if os.environ.get("AGENT_COT_UPLINK_URL"):
                    info(f"uplink dispatched -> {os.environ['AGENT_COT_UPLINK_URL']}")
        except Exception as e:
            error(f"uplink dispatch failed (silenced): {e}")
    except Exception as e:
        error(f"保存报告失败: {e}")
        import traceback
        error(traceback.format_exc())

    # 上传到 Langfuse（只要配置了 Langfuse key 就自动上传）
    try:
        client = create_langfuse_client()
        if client:
            ok = upload_session_cot(session_cot, session_id, client)
            if ok:
                info(f"CoT 上传成功: session={session_id[:16]}")
            else:
                error(f"CoT 上传失败: session={session_id[:16]}")
        else:
            info("未配置 Langfuse key，跳过上传（仅生成本地报告）")
    except Exception as e:
        error(f"upload_session_cot 失败: {e}")
        import traceback
        error(traceback.format_exc())

    # 更新状态
    ss["offset"] = new_offset
    state[key] = ss
    _save_cot_state(state)

    info(f"CoT 提取完成: session={session_id[:16]}, offset={new_offset}")
    return 0


def _extract_topic_from_cot(session_cot) -> str:
    """从 SessionCoT 对象中提取主题"""
    for turn in session_cot.turns:
        if turn.user_query:
            clean = turn.user_query.replace("\r", "").replace("\n", " ").strip()
            return clean[:40] + ("..." if len(clean) > 40 else "")
    return "未知主题"


# ─── Hook 入口 ─────────────────────────────────────────────

def main() -> int:
    try:
        data = sys.stdin.read()
        if not data.strip():
            return 0
        payload = json.loads(data)
    except Exception as e:
        error(f"解析 payload 失败: {e}")
        return 0

    session_id = (
        payload.get("session_id")
        or payload.get("sessionId")
        or ""
    )
    transcript_raw = (
        payload.get("transcript_path")
        or payload.get("transcriptPath")
        or ""
    )

    if not session_id or not transcript_raw:
        error(f"缺少 session_id 或 transcript_path，keys={list(payload.keys())}")
        return 0

    transcript_path = Path(transcript_raw).expanduser().resolve()
    return run(session_id, transcript_path)


if __name__ == "__main__":
    sys.exit(main())
