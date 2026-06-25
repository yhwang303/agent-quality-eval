"""CoT Uplink — 把本地写好的 cot.json 多发一份到中央 dashboard。

v0.18.0 引入。**纯附加功能**：

  * 仅当环境变量 ``AGENT_COT_UPLINK_URL`` 配置时才启用
  * 不影响任何现有逻辑（本地 cot.json / 本地 dashboard / OTLP 导出全部不变）
  * 上传失败完全静默（log 一行，不抛异常，不重试）
  * 异步线程发送，主流程不等待

使用场景
--------

小组共享中央 dashboard：

  * 中央服务（你的电脑或 CVM）跑 ``agent-cot start``
  * 同事侧只需配 2 个环境变量：

      AGENT_COT_UPLINK_URL=http://<central_host>:8765/api/uplink/cot
      AGENT_COT_UPLINK_TOKEN=<shared_secret>     # 可选，与中央服务匹配
      AGENT_COT_USER_ID=zhangsan                 # 可选，缺省用 getuser()

  * 同事每完成一次 turn → cot-extractor 写本地 cot.json + 同时 POST 到中央
  * 中央按 user_id 分桶落盘到 ~/.agent-cot-central/users/<user>/cot/<sid>.json
  * 中央 dashboard 在 SessionList 新增 owner 列 / 用户下拉筛选

接口契约
--------

POST {AGENT_COT_UPLINK_URL}
Headers:
  Content-Type: application/json
  X-Uplink-Token: <token>          # 若服务端配了 AGENT_COT_UPLINK_TOKEN 则强校验
  X-Uplink-User: <user_id>
  X-Uplink-Host: <hostname>

Body:
  {
    "user_id":    "zhangsan",
    "host":       "DESKTOP-XYZ",
    "session_id": "abc-123-...",
    "cot":        { ... session_cot.to_dict() ... },
    "client_version": "v0.18.0",
    "uploaded_at":    "2026-05-09T06:25:00Z"
  }

服务端返回 200 即视为成功，其它一律视为失败但不重试（下次 turn 会再发一份新 cot.json，旧的丢就丢了）。
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 客户端版本（独立于 agent-cot 包版本，方便服务端粗略识别上行端能力）
UPLINK_CLIENT_VERSION = "v0.18.0"

# 单次 POST 超时（秒）。CoT 通常 < 5MB，5s 内网完全够用。
DEFAULT_TIMEOUT = 5.0


def _resolve_user_id() -> str:
    """同事身份解析。优先级：

    1. ``AGENT_COT_USER_ID``（推荐让同事填工号）
    2. ``USER`` / ``USERNAME`` 环境变量
    3. ``getpass.getuser()``
    4. 兜底 ``"unknown"``
    """
    explicit = (os.environ.get("AGENT_COT_USER_ID") or "").strip()
    if explicit:
        return explicit
    env_user = (os.environ.get("USER") or os.environ.get("USERNAME") or "").strip()
    if env_user:
        return env_user
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _resolve_host_id() -> str:
    """主机标识，用于区分『同一同事在多台电脑上』。"""
    explicit = (os.environ.get("AGENT_COT_HOST_ID") or "").strip()
    if explicit:
        return explicit
    try:
        return socket.gethostname() or "unknown-host"
    except Exception:
        return "unknown-host"


def _get_uplink_url() -> Optional[str]:
    """读 URL；空字符串 / None 都视为未配置。"""
    url = (os.environ.get("AGENT_COT_UPLINK_URL") or "").strip()
    return url or None


def _get_uplink_token() -> Optional[str]:
    """共享密钥；服务端校验通过这个 header。"""
    tok = (os.environ.get("AGENT_COT_UPLINK_TOKEN") or "").strip()
    return tok or None


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _post_uplink_sync(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> bool:
    """同步 POST。requests 不可用时降级到标准库 urllib。返回 True 表示 2xx。"""
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    headers = dict(headers)
    headers.setdefault("Content-Type", "application/json")

    # 优先用 requests（agent-cot 已经强制依赖了）
    try:
        import requests  # type: ignore
        try:
            resp = requests.post(url, data=body, headers=headers, timeout=timeout)
            ok = 200 <= resp.status_code < 300
            if not ok:
                logger.warning(
                    "[cot-uplink] POST %s -> HTTP %s; body=%s",
                    url, resp.status_code, (resp.text or "")[:200],
                )
            return ok
        except Exception as e:
            logger.warning("[cot-uplink] POST %s failed (requests): %s", url, e)
            return False
    except ImportError:
        pass

    # 兜底：urllib
    try:
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError
        req = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=timeout) as resp:
                code = resp.getcode()
                ok = 200 <= code < 300
                if not ok:
                    logger.warning("[cot-uplink] POST %s -> HTTP %s (urllib)", url, code)
                return ok
        except HTTPError as e:
            logger.warning("[cot-uplink] POST %s -> HTTP %s (urllib): %s", url, e.code, e.reason)
            return False
        except URLError as e:
            logger.warning("[cot-uplink] POST %s URL error (urllib): %s", url, e.reason)
            return False
    except Exception as e:
        logger.warning("[cot-uplink] POST %s unexpected error: %s", url, e)
        return False


def _audit_log(msg: str) -> None:
    """把上行结果追加到本地审计日志，方便同事自己排查"""
    try:
        log_path = Path.home() / ".agent-cot" / "uplink.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{_now_iso()} {msg}\n")
    except Exception:
        pass


def uplink_session_cot(
    cot_data: Dict[str, Any],
    session_id: str,
    *,
    blocking: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """把一份 cot.json 推到中央 dashboard。

    Args:
        cot_data: ``session_cot.to_dict()`` 的返回值
        session_id: 会话 id
        blocking: True = 同步等结果（CLI 模式 / 测试用）；False = fire-and-forget 异步
        timeout: 单次 POST 超时

    Returns:
        blocking=True 时返回真实成功与否；blocking=False 时立即返回 True（已派发）。
        ``AGENT_COT_UPLINK_URL`` 未配置时直接返回 False，不抛异常。
    """
    url = _get_uplink_url()
    if not url:
        return False

    user_id = _resolve_user_id()
    host_id = _resolve_host_id()
    token = _get_uplink_token()

    headers: Dict[str, str] = {
        "X-Uplink-User": user_id,
        "X-Uplink-Host": host_id,
        "X-Uplink-Client-Version": UPLINK_CLIENT_VERSION,
    }
    if token:
        headers["X-Uplink-Token"] = token

    payload: Dict[str, Any] = {
        "user_id": user_id,
        "host": host_id,
        "session_id": session_id,
        "cot": cot_data,
        "client_version": UPLINK_CLIENT_VERSION,
        "uploaded_at": _now_iso(),
    }

    def _do_post() -> bool:
        try:
            ok = _post_uplink_sync(url, headers, payload, timeout)
            _audit_log(
                f"[uplink] user={user_id} host={host_id} sid={session_id[:8]} "
                f"url={url} ok={ok}"
            )
            return ok
        except Exception as e:
            logger.warning("[cot-uplink] uncaught: %s", e)
            _audit_log(
                f"[uplink-error] user={user_id} sid={session_id[:8]} err={type(e).__name__}: {e}"
            )
            return False

    if blocking:
        return _do_post()

    # 异步派发，不阻塞主流程
    try:
        t = threading.Thread(target=_do_post, daemon=True, name="cot-uplink")
        t.start()
    except Exception as e:
        logger.warning("[cot-uplink] thread spawn failed: %s", e)
        return False
    return True


def uplink_status() -> Dict[str, Any]:
    """诊断辅助：返回当前上行配置的快照（不含 token 明文）。"""
    url = _get_uplink_url()
    token = _get_uplink_token()
    return {
        "enabled": bool(url),
        "url": url,
        "user_id": _resolve_user_id(),
        "host": _resolve_host_id(),
        "token_configured": bool(token),
        "token_preview": (token[:4] + "..." + token[-2:]) if (token and len(token) >= 6) else None,
        "client_version": UPLINK_CLIENT_VERSION,
    }
