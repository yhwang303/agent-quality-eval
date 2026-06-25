"""
FastAPI 后端主入口
"""
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# v0.18.1: 强制覆盖 MIME type 表，必须在 import StaticFiles 之前。
#
# 背景：FastAPI 的 StaticFiles 通过 Python 标准库 mimetypes.guess_type() 判断
# Content-Type；后者在 Windows 上会读注册表 HKEY_CLASSES_ROOT\.js\Content Type。
# 部分用户电脑上这个值被装过的软件（旧 IIS / 某些杀软 / VS / 公司 IT 策略 …）
# 改成了 ``text/plain``，导致 StaticFiles 把 ES module 文件标记为 text/plain
# 发给浏览器；浏览器在 strict module MIME checking（HTML 规范强制）下拒绝执行：
#
#   Failed to load module script: Expected a JavaScript-or-Wasm module script
#   but the server responded with a MIME type of "text/plain".
#
# 结果：React 一行 JS 没跑，整个 SPA 在浏览器里渲染成全黑。
#
# 这里在进程启动早期 add_type，覆盖系统 / 注册表的"错误真相"。无论用户系统
# 怎么配置，浏览器都能拿到正确的 application/javascript / text/css，从而
# 保证 agent-cot start 后的本地 dashboard 在所有 Windows 机器上都能正常加载。
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from services.session_scanner import (
    scan_sessions,
    get_session_cot,
    get_session_response_report,
    get_session_transcript,
    get_session_langfuse_cache,
    delete_session,
)
# v0.16.0: Claude Code 原生 OTel 接收器（OTLP/HTTP/JSON）
from services.claude_otel_receiver import (
    receive_otlp_logs,
    receive_otlp_metrics,
    receive_otlp_traces,
    load_session_otel,
    list_otel_sessions,
)
# v0.18.0: CoT Uplink 接收器（小组中央 dashboard 模式）
from services.uplink_receiver import (
    receive_uplink_cot,
    list_uplink_users,
    uplink_server_status,
)

# Agent Quality Eval: local eval API, mounted into the copied dashboard backend.
try:
    from agent_quality_eval.evaluation.api import router as eval_router
except Exception:  # pragma: no cover - dashboard still serves observation if eval imports fail.
    eval_router = None

# v0.12.0：把 cot-extractor/src 加到 path，方便 import OTLP exporter
#
# v0.13 起 (P3) 优先用 AGENT_COT_EXTRACTOR_SRC env，让 agent-cot start
# 在打包后的 wheel 里也能定位到外部的 cot-extractor 安装位置。
#
# v0.17 起：当 backend 是从 agent-cot wheel 里 vendored 出来跑
# 的时候（用户 pip install 后用 `agent-cot start`），cot-extractor 就在
# wheel 内部 ``agent_cot/assets/cot-extractor-src/``。这里加第三种 fallback
# 让 wheel 安装的用户也能直接用 OTLP / OTel enricher / RAG 等需要 import
# extractor 模块的功能，而不是被 ImportError 卡住。
def _resolve_cot_extractor_src() -> Optional[Path]:
    # 1) env override（CI / 容器环境强制指定）
    env = os.environ.get("AGENT_COT_EXTRACTOR_SRC", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    # 2) sibling repo（开发态 / clone 后直接跑）
    here = Path(__file__).resolve().parent
    sibling = here.parent.parent / "cot-extractor" / "src"
    if sibling.is_dir():
        return sibling
    # 3) vendored 进 wheel（用户 pip install agent-cot 走的路径）
    #    backend 被 _build_assets sync 拷到 agent_cot/assets/backend/，
    #    cot-extractor 拷到 agent_cot/assets/cot-extractor-src/，平级。
    vendored = here.parent / "cot-extractor-src"
    if vendored.is_dir():
        return vendored
    return None


_COT_EXTRACTOR_SRC = _resolve_cot_extractor_src()
if _COT_EXTRACTOR_SRC and str(_COT_EXTRACTOR_SRC) not in sys.path:
    sys.path.insert(0, str(_COT_EXTRACTOR_SRC))

app = FastAPI(title="Agent Dashboard API", version="1.0.0")

# CORS 配置（允许前端开发服务器访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5180", "http://127.0.0.1:5180",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if eval_router is not None:
    app.include_router(eval_router)


@app.get("/eval")
def eval_workbench():
    """Serve the Agent Quality Eval workbench."""
    try:
        from importlib import resources

        page = resources.files("agent_quality_eval").joinpath("assets/eval_workbench.html")
        return FileResponse(str(page))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eval workbench unavailable: {exc}") from exc

# ─── API 路由 ─────────────────────────────────────────────

@app.get("/api/sessions", response_model=List[Dict[str, Any]])
def list_sessions():
    """获取所有 session 的概览列表"""
    return scan_sessions()


@app.get("/api/sessions/{session_id}/cot")
def get_cot(session_id: str):
    """获取指定 session 的完整 CoT 数据"""
    data = get_session_cot(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"CoT data not found for session {session_id}")
    return data


@app.get("/api/sessions/{session_id}/report")
def get_report(session_id: str):
    """获取指定 session 的 Response 准确度报告"""
    data = get_session_response_report(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Report not found for session {session_id}")
    return data


@app.get("/api/sessions/{session_id}/transcript")
def get_transcript(session_id: str):
    """获取指定 session 的 transcript 数据"""
    data = get_session_transcript(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Transcript not found for session {session_id}")
    return data


@app.get("/api/sessions/{session_id}/langfuse")
def get_langfuse(session_id: str):
    """获取指定 session 的 Langfuse 缓存数据"""
    data = get_session_langfuse_cache(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Langfuse cache not found for session {session_id}")
    return data


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ─── v0.16.0: Claude Code 原生 OTel 接收器 ────────────────────
#
# Claude Code 设置 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:8765 后会主动
# 把 metrics / logs / traces 三种信号 POST 到下面三个 OTLP 标准端点。
# 不需要部署 OTel Collector——本地直接落盘到 ~/.claude/state/otel/<sid>/ 后由
# cot_extractor 二次组装并展示在前端。
#
# 三个端点必须是绝对路径（OTLP/HTTP 标准约定，不能挂在 /api/ 下面），所以
# 路由前缀跟 /api/* 平级。

@app.post("/v1/logs")
async def otlp_logs(request: Request):
    """Claude Code logs/events 通道入口（user_prompt / api_request / tool_result / ...）。"""
    return await receive_otlp_logs(request)


@app.post("/v1/metrics")
async def otlp_metrics(request: Request):
    """Claude Code metrics 通道入口（token / cost / decision counters）。"""
    return await receive_otlp_metrics(request)


@app.post("/v1/traces")
async def otlp_traces(request: Request):
    """Claude Code traces beta 通道入口（claude_code.interaction → llm_request / tool 树）。"""
    return await receive_otlp_traces(request)


@app.get("/api/otel/sessions")
def list_otel():
    """列出所有已经收到 OTel 数据的 session（前端 SessionList 用来打 OTel 徽章）。"""
    return {"sessions": list_otel_sessions()}


@app.get("/api/sessions/{session_id}/otel")
def get_session_otel_api(session_id: str):
    """获取指定 session 的完整 OTel 数据（events + metrics + spans + summary）。"""
    data = load_session_otel(session_id)
    if not data.get("session_id"):
        raise HTTPException(status_code=400, detail="session_id is required")
    return data


@app.delete("/api/sessions/{session_id}")
def delete_session_api(session_id: str):
    """删除指定 session 的所有相关文件"""
    ok = delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"deleted": session_id}


# ─── v0.18.0: CoT Uplink ──────────────────────────────────────
#
# 小组中央 dashboard 模式：同事的 cot-extractor 在本地写完 cot.json 后，
# 多发一份到中央服务（你这里）。完全 additive，不改任何现有功能。
#
# 同事侧只需配置 2 个环境变量：
#   AGENT_COT_UPLINK_URL=http://<this_host>:8765/api/uplink/cot
#   AGENT_COT_UPLINK_TOKEN=<shared_secret>     # 可选
#   AGENT_COT_USER_ID=zhangsan                 # 可选
#
# 服务端可选地配 ``AGENT_COT_UPLINK_TOKEN`` env 来要求 token 校验。
# 落盘根目录 ``AGENT_COT_CENTRAL_ROOT`` env，默认 ~/.agent-cot-central/。

@app.post("/api/uplink/cot")
async def uplink_cot_endpoint(request: Request):
    """接收同事推上来的 cot.json，按 user_id 分桶落盘。"""
    return await receive_uplink_cot(request)


@app.get("/api/uplink/users")
def uplink_users_endpoint():
    """列出当前有上行数据的用户（前端用户筛选下拉框数据源）。"""
    return {"users": list_uplink_users()}


@app.get("/api/uplink/status")
def uplink_status_endpoint():
    """中央服务自检：token 配置 / 落盘根 / 已收到的用户与 session 总数。"""
    return uplink_server_status()


# ─── v0.12.0: OTLP 导出 API ──────────────────────────────
# 把 cot.json 重放为标准 OTLP/HTTP traces，推到任意 OTel 兼容后端
# （Phoenix / Langfuse / SigNoz / Jaeger / Datadog / Honeycomb / Tempo / ...）
# 本地前端不会丢——这是一个『追加』的便利通道，方便社区复用。

class OtlpExportRequest(BaseModel):
    endpoint: Optional[str] = Field(
        default=None,
        description="OTLP/HTTP traces endpoint，如 http://localhost:4318/v1/traces；"
                    "缺省读 OTEL_EXPORTER_OTLP_ENDPOINT 或默认值",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="OTLP HTTP header（如 Authorization / x-honeycomb-team）",
    )
    service_name: Optional[str] = Field(default="cot-extractor")
    service_version: Optional[str] = Field(default="v0.12.0")
    deployment_environment: Optional[str] = Field(default=None)
    dry_run: Optional[bool] = Field(
        default=False,
        description="True = 不发送到远程，只在内存里序列化预览（调试用）",
    )
    timeout: Optional[float] = Field(default=10.0)


@app.get("/api/otlp/presets")
def list_otlp_presets():
    """列出内置后端预设（前端 OTLP 弹窗用）。"""
    try:
        from cot_otlp_exporter import BACKEND_PRESETS  # type: ignore
        return {"presets": BACKEND_PRESETS}
    except ImportError:
        return {
            "presets": [],
            "error": (
                "OTLP exporter 模块缺失。请确认 cot-extractor/src/cot_otlp_exporter.py 存在。"
            ),
        }


@app.post("/api/sessions/{session_id}/export/otlp")
def export_session_otlp(session_id: str, body: OtlpExportRequest):
    """把指定 session 的 cot.json 重放为 OTLP traces 推到 ``body.endpoint``。

    ``body.dry_run=True`` 时不连后端，仅返回 span 树预览。
    """
    cot_data = get_session_cot(session_id)
    if cot_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"CoT data not found for session {session_id}",
        )

    try:
        from cot_otlp_exporter import export_session_to_otlp  # type: ignore
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "opentelemetry 依赖未安装。请：\n"
                "  pip install -r cot-extractor/requirements.txt\n"
                f"原始错误：{e}"
            ),
        )

    try:
        result = export_session_to_otlp(
            cot_data,
            endpoint=body.endpoint,
            headers=body.headers,
            service_name=body.service_name or "cot-extractor",
            service_version=body.service_version or "v0.12.0",
            deployment_environment=body.deployment_environment,
            dry_run=bool(body.dry_run),
            timeout=float(body.timeout or 10.0),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OTLP 导出失败：{type(e).__name__}: {e}",
        )

    return result


# ─── v0.14.8: 离线 OTLP/JSON 协议下载 ────────────────────────
# 与上面的 POST /export/otlp 不同——那个是把 trace 实时推到一个真实 OTel 后端
# （Phoenix / SigNoz / ...）。这里是"导出到本地一份完整 OTLP/JSON 协议文件"，
# 用户在 UI 上点导出按钮直接下载，可以离线交给 jaeger-cli / otel-cli /
# 任意二次处理脚本。Why 必要：很多场景没法部署 collector，但拿到协议文件就能
# 用 SDK 重放、做对比测试、塞进 CI artifact。

@app.get("/api/sessions/{session_id}/export/otlp.json")
def download_session_otlp_json(
    session_id: str,
    service_name: str = "cot-extractor",
    service_version: str = "v0.12.0",
    deployment_environment: Optional[str] = None,
):
    """返回当前 session 的完整 OTLP/JSON 协议 payload，浏览器会触发下载。

    Query params 都是可选的 service.* resource attrs，UI 一般用默认即可。
    """
    cot_data = get_session_cot(session_id)
    if cot_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"CoT data not found for session {session_id}",
        )

    try:
        from cot_otlp_exporter import build_session_otlp_json  # type: ignore
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "opentelemetry SDK 缺失。请：\n"
                "  pip install -r cot-extractor/requirements.txt\n"
                f"原始错误：{e}"
            ),
        )

    try:
        payload = build_session_otlp_json(
            cot_data,
            service_name=service_name or "cot-extractor",
            service_version=service_version or "v0.12.0",
            deployment_environment=deployment_environment,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OTLP/JSON 构建失败：{type(e).__name__}: {e}",
        )

    # 触发浏览器下载，文件名带 session id 的前 8 位 + trace_id 前 8 位，方便归档
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    trace_id = (payload.get("_meta") or {}).get("trace_id") or ""
    fname_session = (session_id or "session")[:8]
    fname_trace = (trace_id or "")[:8]
    suffix = f"-{fname_trace}" if fname_trace else ""
    filename = f"otlp-{fname_session}{suffix}.json"

    return Response(
        content=body,
        media_type="application/json",
        headers={
            # RFC 5987 兼容写法 —— filename 给 ASCII fallback；filename* 给 UTF-8
            "Content-Disposition": (
                f'attachment; filename="{filename}"; '
                f"filename*=UTF-8''{filename}"
            ),
            # 让前端 fetch 能拿到 trace_id 不用解 body
            "X-Cot-Trace-Id": trace_id,
            "X-Cot-Span-Count": str((payload.get("_meta") or {}).get("span_count") or 0),
        },
    )


# ─── 静态文件服务（生产模式） ──────────────────────────────
#
# Resolution order:
#   1. AGENT_COT_FRONTEND_DIST env     ← set by `agent-cot start` (P3)
#   2. ../frontend/dist                 ← dev: source-tree relative
#   3. ../frontend-dist                 ← v0.18.15: wheel-bundled SPA
#                                         (assets/backend & assets/frontend-dist
#                                         are siblings inside the wheel).
#                                         Without this, a user who launches
#                                         the backend directly (without going
#                                         through `agent-cot start` so without
#                                         AGENT_COT_FRONTEND_DIST set) would
#                                         get API only — no UI — even though
#                                         the SPA is sitting right there.

def _resolve_frontend_dist() -> Path | None:
    env_dist = os.environ.get("AGENT_COT_FRONTEND_DIST", "").strip()
    candidates = []
    if env_dist:
        candidates.append(Path(env_dist))
    here = Path(__file__).parent
    candidates.append(here.parent / "frontend" / "dist")
    candidates.append(here.parent / "frontend-dist")
    for cand in candidates:
        if cand and cand.is_dir() and (cand / "index.html").is_file():
            return cand.resolve()
    return None


_FRONTEND_DIST = _resolve_frontend_dist()


def _no_cache_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }


class NoCacheStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers.update(_no_cache_headers())
        return response

if _FRONTEND_DIST is not None:
    _ASSETS_DIR = _FRONTEND_DIST / "assets"
    if _ASSETS_DIR.is_dir():
        app.mount("/assets", NoCacheStaticFiles(directory=str(_ASSETS_DIR)), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        # 不能让 catch-all 劫持 /api/*（导致 OTLP 等 API 返回 HTML）
        if full_path.startswith("api/") or full_path.startswith("api"):
            raise HTTPException(status_code=404, detail=f"API route not found: /{full_path}")
        if full_path:
            candidate = _FRONTEND_DIST / full_path
            try:
                candidate.resolve().relative_to(_FRONTEND_DIST.resolve())
            except ValueError:
                raise HTTPException(status_code=404, detail="not found")
            if candidate.is_file():
                return FileResponse(str(candidate), headers=_no_cache_headers())
        index = _FRONTEND_DIST / "index.html"
        return FileResponse(str(index), headers=_no_cache_headers())


# ─── 启动入口 ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config import HOST, PORT
    print(f"🚀 Agent Dashboard API 启动: http://{HOST}:{PORT}")
    print(f"📖 API 文档: http://{HOST}:{PORT}/docs")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
