#!/usr/bin/env python3
"""
CoT Uploader — 将 CoT 上传到 Langfuse

复用 langfuse_hook.py 已创建的 trace_id，
将 CoT 步骤作为 span 上传到 Langfuse，
实现在 Langfuse 中可视化完整的思维链。
"""

import json
import os
import sys
import time
import uuid
import hashlib
import urllib.request
import urllib.parse
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cot_extractor import SessionCoT, TurnCoT, ThoughtStep, StepType


# ─── 配置 ─────────────────────────────────────────────────

STATE_FILE = Path.home() / ".claude" / "state" / "langfuse_state.json"
LOG_FILE = Path.home() / ".claude" / "state" / "cot_hook.log"


def _log(level: str, msg: str) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} [{level}] {msg}\n")
    except Exception:
        pass


def info(msg: str) -> None:
    _log("INFO", msg)


def error(msg: str) -> None:
    _log("ERROR", msg)


# ─── 读取 langfuse_hook.py 的 state，获取 trace_id ────────

def _session_state_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def get_trace_id(session_id: str) -> Optional[str]:
    """从 langfuse_state.json 中读取当前 session 的 trace_id"""
    try:
        if not STATE_FILE.exists():
            return None
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        key = _session_state_key(session_id)
        ss = state.get(key, {})
        return ss.get("trace_id")
    except Exception as e:
        error(f"get_trace_id failed: {e}")
        return None


def get_turn_span_id(session_id: str) -> Optional[str]:
    """从 langfuse_state.json 中读取当前 session 的 turn span_id"""
    try:
        if not STATE_FILE.exists():
            return None
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        key = _session_state_key(session_id)
        ss = state.get(key, {})
        return ss.get("current_turn_span_id")
    except Exception as e:
        error(f"get_turn_span_id failed: {e}")
        return None


def get_turn_count(session_id: str) -> int:
    """从 langfuse_state.json 中读取当前 session 的 turn_count"""
    try:
        if not STATE_FILE.exists():
            return 0
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        key = _session_state_key(session_id)
        ss = state.get(key, {})
        return ss.get("turn_count", 0)
    except Exception as e:
        error(f"get_turn_count failed: {e}")
        return 0


# ─── Langfuse HTTP API（不依赖 SDK） ──────────────────────

class LangfuseHTTPClient:
    """
    轻量级 Langfuse HTTP 客户端，只使用 Python 标准库。
    避免依赖 langfuse SDK，减少安装要求。
    """

    def __init__(self, public_key: str, secret_key: str, host: str):
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host.rstrip("/")
        # Basic Auth
        credentials = f"{public_key}:{secret_key}"
        self._auth = base64.b64encode(credentials.encode()).decode()
        self._batch: List[Dict] = []

    def _post(self, endpoint: str, data: Dict) -> bool:
        """发送 POST 请求"""
        url = f"{self.host}{endpoint}"
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Basic {self._auth}",
                "User-Agent": "cot-extractor/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 201, 207)
        except Exception as e:
            error(f"HTTP POST {endpoint} failed: {e}")
            return False

    def add_span(
        self,
        trace_id: str,
        name: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: Dict = None,
        parent_observation_id: str = None,
        level: str = "DEFAULT",
    ) -> str:
        """添加一个 span 到批次"""
        import uuid
        span_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        item = {
            "id": span_id,
            "traceId": trace_id,
            "type": "SPAN",
            "name": name,
            "startTime": now,
            "endTime": now,
            "level": level,
        }
        if input_data is not None:
            item["input"] = input_data
        if output_data is not None:
            item["output"] = output_data
        if metadata:
            item["metadata"] = metadata
        if parent_observation_id:
            item["parentObservationId"] = parent_observation_id
        self._batch.append(item)
        return span_id

    def flush(self) -> bool:
        """批量发送所有 span"""
        if not self._batch:
            return True
        payload = {"batch": self._batch}
        ok = self._post("/api/public/ingestion", payload)
        if ok:
            info(f"Langfuse flush: sent {len(self._batch)} items")
        else:
            error(f"Langfuse flush failed: {len(self._batch)} items lost")
        self._batch = []
        return ok


def create_langfuse_client() -> Optional[LangfuseHTTPClient]:
    """创建 Langfuse HTTP 客户端"""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") or os.environ.get("CC_LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY") or os.environ.get("CC_LANGFUSE_SECRET_KEY")
    host = (
        os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("CC_LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )
    if not public_key or not secret_key:
        error("Missing Langfuse keys (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)")
        return None
    return LangfuseHTTPClient(public_key, secret_key, host)


# ─── CoT 上传逻辑 ─────────────────────────────────────────

STEP_TYPE_TO_SPAN_NAME = {
    StepType.USER_INPUT:        "CoT: User Input",
    StepType.TOOL_RESULT_INPUT: "CoT: Tool Result Input",
    StepType.THINKING_INTER:    "CoT: Thinking (Intermediate)",
    StepType.THINKING_EXPLICIT: "CoT: Thinking (Explicit)",
    StepType.TOOL_DECISION:     "CoT: Tool Decision",
    StepType.TOOL_EXECUTION:    "CoT: Tool Execution",
    StepType.STRATEGY_SHIFT:    "CoT: Strategy Shift",
    StepType.ERROR_RECOVERY:    "CoT: Error Recovery",
    StepType.FINAL_RESPONSE:    "CoT: Final Response",
}

STEP_TYPE_TO_LEVEL = {
    StepType.ERROR_RECOVERY:  "WARNING",
    StepType.STRATEGY_SHIFT:  "WARNING",
}


def upload_session_cot(
    session_cot: SessionCoT,
    session_id: str,
    client: LangfuseHTTPClient,
) -> bool:
    """
    将 SessionCoT 上传到 Langfuse。

    结构：
      Trace (已存在，由 langfuse_hook.py 创建)
        └── [CoT] Turn N  ← 挂到 current_turn_span_id（最后一个 turn span）
              ├── [CoT] Step 1: User Input
              ├── [CoT] Step 2: Thinking (Intermediate)
              ├── [CoT] Step 3: Tool Decision → Bash
              ├── [CoT] Step 4: Tool Execution → Bash
              ├── [CoT] Step 5: Strategy Shift
              └── [CoT] Step 6: Final Response
    """
    trace_id = get_trace_id(session_id)
    if not trace_id:
        error(f"upload_session_cot: no trace_id for session={session_id[:16]}")
        return False

    # 读取 langfuse_hook.py 记录的最新 turn span_id（作为 CoT 的父节点）
    current_turn_span_id = get_turn_span_id(session_id)
    turn_count = get_turn_count(session_id)

    info(f"upload_session_cot: session={session_id[:16]}, trace_id={trace_id[:16]}, "
         f"turns={len(session_cot.turns)}, current_turn_span={current_turn_span_id}")

    for turn in session_cot.turns:
        # 直接用 turn.turn_index 推算 span_id，与 langfuse_hook.py 的 uuid5 算法完全一致
        # 不依赖 turn_count（turn_count 在 state 里可能不准确）
        parent_span = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"turn-{session_id}-{turn.turn_index}"))
        _upload_turn_cot(turn, trace_id, session_id, client, parent_span)

    return client.flush()


def _upload_turn_cot(
    turn: TurnCoT,
    trace_id: str,
    session_id: str,
    client: LangfuseHTTPClient,
    parent_span_id: Optional[str] = None,
) -> None:
    """上传单个 turn 的 CoT，挂载到对应的 Turn span 下"""

    # 创建 CoT 容器 span（包含整个 turn 的 CoT）
    cot_container_id = client.add_span(
        trace_id=trace_id,
        name=f"CoT: Turn {turn.turn_index}",
        input_data={
            "user_query": turn.user_query[:500] if turn.user_query else "",
            "total_steps": turn.total_steps,
        },
        output_data={
            "tool_calls": turn.tool_calls,
            "strategy_shifts": turn.strategy_shifts,
            "thinking_depth": turn.thinking_depth,
            "has_error_recovery": turn.has_error_recovery,
            "complexity_score": turn.complexity_score,
            "final_response_preview": turn.final_response[:200] if turn.final_response else "",
        },
        metadata={
            "turn_index": turn.turn_index,
            "total_steps": turn.total_steps,
            "tool_call_count": len(turn.tool_calls),
            "strategy_shifts": turn.strategy_shifts,
            "thinking_depth": turn.thinking_depth,
            "has_error_recovery": turn.has_error_recovery,
            "complexity_score": turn.complexity_score,
            "usage": turn.usage,
        },
        parent_observation_id=parent_span_id,
    )

    # 为每个步骤创建子 span
    for step in turn.steps:
        span_name = STEP_TYPE_TO_SPAN_NAME.get(step.step_type, f"CoT: {step.step_type}")
        level = STEP_TYPE_TO_LEVEL.get(step.step_type, "DEFAULT")

        # 构造 input/output
        if step.step_type == StepType.TOOL_DECISION:
            span_name = f"CoT: Tool Decision → {step.tool_name}"
            input_data = {
                "tool_name": step.tool_name,
                "tool_use_id": step.tool_use_id,
                "tool_input": step.metadata.get("tool_input", {}),
            }
            output_data = None
        elif step.step_type == StepType.TOOL_EXECUTION:
            input_data = {"tool_use_id": step.tool_use_id}
            output_data = {
                "result": step.content[:1000],
                "is_error": step.metadata.get("is_error", False),
                "result_len": step.metadata.get("result_len", len(step.content)),
            }
        elif step.step_type in (StepType.THINKING_INTER, StepType.THINKING_EXPLICIT):
            input_data = {"thinking_type": step.step_type}
            output_data = {"thinking_content": step.content[:2000]}
        elif step.step_type == StepType.FINAL_RESPONSE:
            input_data = None
            output_data = {"response": step.content[:2000]}
        elif step.step_type == StepType.STRATEGY_SHIFT:
            input_data = {
                "from_tool": step.metadata.get("from_tool", ""),
                "to_tool": step.metadata.get("to_tool", ""),
            }
            output_data = {"reason": step.content}
        elif step.step_type == StepType.ERROR_RECOVERY:
            input_data = {"error_content": step.metadata.get("error_content", "")[:500]}
            output_data = {"recovery_action": step.content[:500]}
        else:
            input_data = {"content": step.content[:500]}
            output_data = None

        client.add_span(
            trace_id=trace_id,
            name=span_name,
            input_data=input_data,
            output_data=output_data,
            metadata={
                "step_index": step.step_index,
                "step_type": step.step_type,
                "turn_index": step.turn_index,
                "tool_name": step.tool_name or None,
                "tool_use_id": step.tool_use_id or None,
                "tokens": step.tokens,
            },
            parent_observation_id=cot_container_id,
            level=level,
        )

    info(f"  Turn {turn.turn_index}: {len(turn.steps)} steps queued for upload")
