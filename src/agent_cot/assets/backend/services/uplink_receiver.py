"""CoT Uplink Receiver — 接收同事推上来的 cot.json，按 user_id 分桶落盘。

v0.18.0 引入。**纯附加功能**，不改任何现有逻辑：

  * 新端点：``POST /api/uplink/cot``
  * 落盘根：``AGENT_COT_CENTRAL_ROOT`` env，缺省 ``~/.agent-cot-central``
  * 目录结构：

      ~/.agent-cot-central/
        users/
          <user_id>/
            cot/
              <session_id>_cot.json    ← session_scanner 会扫这里
            meta.jsonl                 ← 每次上传一条审计记录

  * Token 鉴权：服务端配 ``AGENT_COT_UPLINK_TOKEN`` env 时强校验 ``X-Uplink-Token``；不配则任何人能写
  * 跨机/跨用户隔离：session_id 在 SessionList 中以 ``<user_id>::<sid>`` 形式呈现，避免与本机 session 撞 id
  * 失败不阻塞同事侧：返回 200 + ``{"ok": false, "reason": "..."}`` 让客户端日志可见但不报错重试

接口
----

POST /api/uplink/cot
Headers:
  Content-Type: application/json
  X-Uplink-Token: <shared_secret>      # 服务端配 AGENT_COT_UPLINK_TOKEN 时必传
  X-Uplink-User:  <user_id>            # body.user_id 优先
  X-Uplink-Host:  <hostname>

Body:
  {
    "user_id": "zhangsan",
    "host":    "DESKTOP-XYZ",
    "session_id": "abc-...",
    "cot":     { ... session_cot.to_dict() ... },
    "client_version": "v0.18.0",
    "uploaded_at": "..."
  }

GET /api/uplink/users
  → { "users": [ {"user_id":"zhangsan","session_count":3,"last_uploaded":"..."}, ... ] }

GET /api/uplink/status
  → 自检：当前服务端是否启用 token 校验、落盘根目录在哪
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request, Response

logger = logging.getLogger(__name__)

# 落盘根目录
_DEFAULT_CENTRAL_ROOT = Path.home() / ".agent-cot-central"


def central_root() -> Path:
    """中央落盘根目录。每次读环境变量，方便 docker 等场景动态配置。"""
    env = (os.environ.get("AGENT_COT_CENTRAL_ROOT") or "").strip()
    if env:
        return Path(env).expanduser()
    return _DEFAULT_CENTRAL_ROOT


def users_root() -> Path:
    return central_root() / "users"


def _expected_token() -> Optional[str]:
    tok = (os.environ.get("AGENT_COT_UPLINK_TOKEN") or "").strip()
    return tok or None


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


# 用户 / session id 安全字符过滤——只允许 [a-zA-Z0-9_.-]，最长 80
# 防 path traversal（"../"）和奇怪字符落到文件名
_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.\-]")


def _safe_id(raw: Any, fallback: str = "unknown") -> str:
    if not raw:
        return fallback
    s = str(raw).strip()
    if not s:
        return fallback
    s = _SAFE_RE.sub("_", s)
    return s[:80] or fallback


def _user_dir(user_id: str) -> Path:
    return users_root() / _safe_id(user_id, "unknown")


def _user_cot_dir(user_id: str) -> Path:
    return _user_dir(user_id) / "cot"


def _user_meta_path(user_id: str) -> Path:
    return _user_dir(user_id) / "meta.jsonl"


def _append_meta(user_id: str, record: Dict[str, Any]) -> None:
    """每次上传写一条审计记录，方便排查谁在何时上传了什么"""
    try:
        path = _user_meta_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str))
            f.write("\n")
    except Exception as e:
        logger.warning("[uplink-receiver] meta append failed for %s: %s", user_id, e)


# ════════════════════════════════════════════════════════════
#  POST handler
# ════════════════════════════════════════════════════════════

async def receive_uplink_cot(request: Request) -> Response:
    """接收 cot.json 上传。

    任何输入错误都返回 200 + JSON ``{"ok": false, "reason": "..."}``，避免同事侧
    HTTP 错误日志刷屏；只有完全没收到 body / token 校验失败才会返 4xx。
    """
    expected = _expected_token()
    if expected:
        got = (request.headers.get("X-Uplink-Token") or "").strip()
        if got != expected:
            logger.warning("[uplink-receiver] reject: bad token from %s", request.client.host if request.client else "?")
            raise HTTPException(status_code=401, detail="bad uplink token")

    try:
        raw = await request.body()
    except Exception:
        raise HTTPException(status_code=400, detail="cannot read body")

    if not raw:
        raise HTTPException(status_code=400, detail="empty body")

    try:
        payload = json.loads(raw)
    except Exception as e:
        return Response(
            content=json.dumps({"ok": False, "reason": f"invalid json: {e}"}),
            media_type="application/json",
            status_code=200,
        )

    if not isinstance(payload, dict):
        return Response(
            content=json.dumps({"ok": False, "reason": "payload must be object"}),
            media_type="application/json",
            status_code=200,
        )

    user_id = _safe_id(
        payload.get("user_id") or request.headers.get("X-Uplink-User"),
        fallback="unknown",
    )
    host = (payload.get("host") or request.headers.get("X-Uplink-Host") or "").strip()
    session_id = _safe_id(payload.get("session_id"), fallback="")

    if not session_id:
        return Response(
            content=json.dumps({"ok": False, "reason": "missing session_id"}),
            media_type="application/json",
            status_code=200,
        )

    cot = payload.get("cot")
    if not isinstance(cot, dict):
        return Response(
            content=json.dumps({"ok": False, "reason": "missing or invalid 'cot' dict"}),
            media_type="application/json",
            status_code=200,
        )

    # 落盘
    cot_path = _user_cot_dir(user_id) / f"{session_id}_cot.json"
    try:
        cot_path.parent.mkdir(parents=True, exist_ok=True)
        # 给 cot.json 注入 owner / host / received_at，方便后端二次识别
        # 这些字段是『加』，不覆盖原 cot 里同名字段（如果有的话）
        cot_with_meta = dict(cot)
        cot_with_meta.setdefault("_uplink", {})
        cot_with_meta["_uplink"] = {
            "owner": user_id,
            "host": host or None,
            "received_at": _now_iso(),
            "client_version": payload.get("client_version") or None,
        }
        cot_path.write_text(
            json.dumps(cot_with_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.exception("[uplink-receiver] write failed: %s", e)
        return Response(
            content=json.dumps({"ok": False, "reason": f"write failed: {e}"}),
            media_type="application/json",
            status_code=200,
        )

    # 写审计
    _append_meta(user_id, {
        "ts": _now_iso(),
        "session_id": session_id,
        "host": host or None,
        "client_version": payload.get("client_version") or None,
        "size_bytes": cot_path.stat().st_size if cot_path.exists() else 0,
        "client_ip": request.client.host if request.client else None,
    })

    logger.info("[uplink-receiver] accepted user=%s sid=%s bytes=%d",
                user_id, session_id, cot_path.stat().st_size if cot_path.exists() else 0)

    return Response(
        content=json.dumps({
            "ok": True,
            "user_id": user_id,
            "session_id": session_id,
            "stored_at": str(cot_path),
        }, ensure_ascii=False),
        media_type="application/json",
        status_code=200,
    )


# ════════════════════════════════════════════════════════════
#  GET handlers
# ════════════════════════════════════════════════════════════

def list_uplink_users() -> List[Dict[str, Any]]:
    """列所有有上行数据的用户（前端用户下拉框数据源）"""
    root = users_root()
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for u in sorted(root.iterdir()):
        if not u.is_dir():
            continue
        cot_dir = u / "cot"
        if not cot_dir.exists():
            continue
        sessions = list(cot_dir.glob("*_cot.json"))
        if not sessions:
            continue
        last_mtime = max((s.stat().st_mtime for s in sessions), default=0)
        out.append({
            "user_id": u.name,
            "session_count": len(sessions),
            "last_uploaded": datetime.fromtimestamp(last_mtime, tz=timezone.utc)
                .isoformat().replace("+00:00", "Z") if last_mtime else None,
            "total_bytes": sum(s.stat().st_size for s in sessions),
        })
    out.sort(key=lambda r: r.get("last_uploaded") or "", reverse=True)
    return out


def uplink_server_status() -> Dict[str, Any]:
    """中央服务自检：用户告诉我 'token 配了没 / 落盘在哪 / 收了多少'"""
    root = central_root()
    users = list_uplink_users()
    return {
        "central_root": str(root),
        "central_root_exists": root.exists(),
        "auth_required": bool(_expected_token()),
        "user_count": len(users),
        "total_session_count": sum(u.get("session_count", 0) for u in users),
        "users": users,
    }


# ════════════════════════════════════════════════════════════
#  给 session_scanner 用的：扫所有 user/cot 目录，返回 (user_id, cot_file) 列表
# ════════════════════════════════════════════════════════════

def iter_central_cot_files() -> List[Tuple[str, Path]]:
    """返回 [(user_id, cot_file_path), ...]，session_scanner 调用后可统一处理。"""
    out: List[Tuple[str, Path]] = []
    root = users_root()
    if not root.exists():
        return out
    for u in root.iterdir():
        if not u.is_dir():
            continue
        cot_dir = u / "cot"
        if not cot_dir.exists():
            continue
        for f in cot_dir.glob("*_cot.json"):
            if f.name.startswith("."):
                continue
            out.append((u.name, f))
    return out
