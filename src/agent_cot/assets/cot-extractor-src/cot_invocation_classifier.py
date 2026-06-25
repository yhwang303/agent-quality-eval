#!/usr/bin/env python3
"""LLM / RAG / Web Search 调用分类器（v0.8.0）

把 tool_decision / tool_execution 步骤按调用语义打标签：

- ``llm_call``   显式向 LLM 发起对话补全（claude / openai CLI、Anthropic / OpenAI HTTP、
                 任何一类 LLM-MCP）
- ``rag_query``  查 RAG / 向量库 / 知识库（aiSearchDocument、Chroma、WebFetch、
                 任何一类 vector / knowledge MCP）
- ``web_search`` 在线搜索（WebSearch / 任何"在 web 上检索"类工具）

完全不依赖网络/LLM，纯启发式 + 白名单：

1. 白名单优先（高置信度，明确知道是哪一类的工具/服务/CLI 直接命中）；
2. 白名单全 miss 时再用关键词启发式（含 ``search`` / ``query`` / ``retrieve`` /
   ``rag`` / ``embed`` / ``vector`` / ``knowledge`` 时归 ``rag_query``）；
3. 任何一处分类失败都返回 ``None``，调用方应**整段 try/except 兜底**，分类
   失败绝不能影响主提取流程。

白名单都集中在本模块顶部，用户加新工具直接改这里就行，不需要改 cot_extractor。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


# ─── 高置信度白名单（小写匹配） ───────────────────────────

LLM_CALL_WHITELIST: Dict[str, Any] = {
    # Cursor MCP server 名（顶层 server）
    "mcp_servers": {
        "openai", "anthropic", "claude-mcp", "gemini-mcp", "ollama-mcp",
        "deepseek", "qwen", "moonshot", "litellm",
    },
    # Cursor MCP 子工具名（CallMcpTool 的 toolName 字段）
    "mcp_tools": {
        "chat_completion", "create_message", "complete", "generate",
        "create_completion", "messages.create", "chat",
    },
    # Shell command 中出现的 LLM HTTP endpoint（出现即命中）
    "shell_endpoints": (
        "api.anthropic.com",
        "api.openai.com",
        "generativelanguage.googleapis.com",
        "api.deepseek.com",
        "open.bigmodel.cn",
        "dashscope.aliyuncs.com",
        "api.together.xyz",
        "openrouter.ai/api",
        "huggingface.co/api/inference",
    ),
    # Shell command 首 token 是已知 LLM CLI
    "shell_clis": {
        "claude", "openai", "gemini", "ollama", "litellm", "aichat", "chatgpt",
    },
}

RAG_QUERY_WHITELIST: Dict[str, Any] = {
    # Cursor 顶层工具名（保留大小写匹配）
    "tools": {"WebFetch"},
    # MCP server 名（小写匹配）—— 整个 server 都视为知识库查询入口
    # （前提是 sub_tool 不在 RAG_WRITE_BLOCKLIST 里）
    "mcp_servers": {
        "user-iwiki", "iwiki",
        "confluence", "notion-mcp", "notion",
        "wiki-mcp", "wiki",
    },
    # MCP 子工具名（小写匹配）
    "mcp_tools": {
        "aisearchdocument", "searchdocument",
        "glossarytermsearch", "glossarybatchexactsearch", "glossarytermexactsearch",
        "vector_search", "knowledge_search", "rag_search",
        "documentsearch", "semanticsearch", "hybridsearch",
        "getdocument", "fetchdocument", "readdocument",  # 取知识库单篇文档也算 RAG 召回
        "listdocuments", "listpages",
    },
    # 向量库 / 知识库 CLI
    "shell_clis": {"chroma", "qdrant", "pinecone", "weaviate", "milvus"},
}

# RAG MCP server 内部的"写"类工具：即便 server 在白名单也不应归 RAG（这是写入而非查询）
# v0.14.4：补 saveDocumentParts / createDocumentPart 等"分块写"变体，并用前缀兜底
# （之前 saveDocumentParts 漏过了 BLOCKLIST，被 user-iWiki server 通配命中
# rag_query，导致前端渲染出"RAG Query → 局部更新成功"这种文不对题的卡片）
RAG_WRITE_BLOCKLIST: frozenset = frozenset({
    "savedocument", "createdocument", "updatedocument", "deletedocument",
    "patchdocument", "putdocument", "writedocument",
    "savepage", "createpage", "updatepage",
    # 分块版（user-iWiki saveDocumentParts 系列）
    "savedocumentparts", "savedocumentpart",
    "createdocumentpart", "createdocumentparts",
    "updatedocumentpart", "updatedocumentparts",
    "deletedocumentpart", "deletedocumentparts",
})

# v0.14.4：写操作前缀兜底——只要子工具名以这些动词前缀开头就视为写。
# 不放 "get" / "search" / "list" / "fetch" / "read" / "find" 等读语义前缀。
RAG_WRITE_PREFIXES: Tuple[str, ...] = (
    "save", "create", "update", "delete", "patch", "put",
    "write", "remove", "modify", "edit", "insert", "append",
    "publish", "submit", "upload",
)


def _is_mcp_write_op(sub: Optional[str]) -> bool:
    """子工具名是否为'写'类操作（白名单 + 前缀兜底）。"""
    if not sub:
        return False
    if sub in RAG_WRITE_BLOCKLIST:
        return True
    return any(sub.startswith(p) for p in RAG_WRITE_PREFIXES)

WEB_SEARCH_TOOLS = {"WebSearch"}

# 启发式回退关键词（小写匹配 tool_name 或子工具名子串）
RAG_HEURISTIC_KW: Tuple[str, ...] = (
    "search", "query", "retrieve", "rag", "embed", "vector", "knowledge",
)


# ─── 内部小工具 ───────────────────────────────────────────

def _safe_lower(x: Any) -> str:
    return str(x).strip().lower() if x is not None else ""


def _shell_first_token(cmd: str) -> str:
    """从 shell command 提取真正要执行的第一个程序名（跳过 sudo / env=val 等 wrapper）。"""
    if not cmd:
        return ""
    skip = {"sudo", "exec", "env", "time", "nice", "stdbuf"}
    for raw in cmd.strip().split():
        token = raw.strip("\"'`")
        if not token:
            continue
        low = token.lower()
        if low in skip:
            continue
        # `KEY=value foo` 这种环境变量前缀
        if "=" in token and not token.startswith("-"):
            continue
        # 路径切到 basename：`/usr/bin/curl` → `curl`
        if "/" in token or "\\" in token:
            token = token.replace("\\", "/").rsplit("/", 1)[-1]
        return token.lower()
    return ""


def _looks_like_curl_to_llm(cmd: str) -> bool:
    """``curl ... api.openai.com ...`` / ``wget ... api.anthropic.com``。"""
    if not cmd:
        return False
    low = cmd.lower()
    if "curl" not in low and "wget" not in low and "http" not in low:
        return False
    return any(ep in low for ep in LLM_CALL_WHITELIST["shell_endpoints"])


def _extract_mcp_server_tool(tool_input: Optional[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """对 ``CallMcpTool`` 这种 MCP 入口，从 tool_input 抽出真正的 server / toolName。"""
    if not isinstance(tool_input, dict):
        return None, None
    server = (
        tool_input.get("server")
        or tool_input.get("server_name")
        or tool_input.get("mcp_server")
    )
    sub_tool = (
        tool_input.get("toolName")
        or tool_input.get("tool_name")
        or tool_input.get("tool")
        or tool_input.get("name")
    )
    return _safe_lower(server) or None, _safe_lower(sub_tool) or None


# ─── 主分类入口 ───────────────────────────────────────────

def classify(
    tool_name: Optional[str],
    mcp_server: Optional[str] = None,
    tool_input: Optional[Dict] = None,
    command: Optional[str] = None,
) -> Optional[str]:
    """返回 ``'llm_call'`` / ``'rag_query'`` / ``'web_search'`` 之一，否则 ``None``。

    Parameters
    ----------
    tool_name
        Cursor / Claude transcript 中 ``tool_use.name`` 字段。
    mcp_server
        若 transcript 直接带了 ``server_name`` / ``mcp_server`` 字段就传进来，
        没有就 None，代码会从 tool_input 兜底解析。
    tool_input
        ``tool_use.input`` 整段 dict；既用来抽 MCP 子工具名，也用来抽 shell command。
    command
        若调用方已经从 tool_input 里解析出 shell command，就直接传字符串；
        否则函数会尝试 ``tool_input.get('command')``。
    """
    name = (tool_name or "").strip()
    name_lower = name.lower()
    inp = tool_input if isinstance(tool_input, dict) else {}

    # 1) 顶层工具白名单（精确大小写）
    if name in WEB_SEARCH_TOOLS:
        return "web_search"
    if name in RAG_QUERY_WHITELIST["tools"]:
        return "rag_query"

    # 2) MCP 入口：CallMcpTool / call_mcp_tool / mcp 都可能
    if name_lower in ("callmcptool", "call_mcp_tool", "mcp", "callmcptools"):
        srv, sub = _extract_mcp_server_tool(inp)
        # 写入类工具直接跳过 RAG 判定（saveDocument / saveDocumentParts 这类不是查询）
        is_write_op = _is_mcp_write_op(sub)
        if srv and srv in LLM_CALL_WHITELIST["mcp_servers"]:
            return "llm_call"
        if sub and sub in LLM_CALL_WHITELIST["mcp_tools"]:
            return "llm_call"
        if not is_write_op:
            if sub and sub in RAG_QUERY_WHITELIST["mcp_tools"]:
                return "rag_query"
            if srv and srv in RAG_QUERY_WHITELIST["mcp_servers"]:
                # 整个 server 是知识库类，且不是写操作 → RAG 查询
                return "rag_query"
            if sub and any(kw in sub for kw in RAG_HEURISTIC_KW):
                return "rag_query"

    # 3) 显式传进来的 mcp_server（Claude transcript 有时直接带）
    srv_low = _safe_lower(mcp_server)
    if srv_low and srv_low in LLM_CALL_WHITELIST["mcp_servers"]:
        return "llm_call"

    # 4) Shell command 启发
    cmd = command or (inp.get("command") if isinstance(inp, dict) else None) or ""
    cmd = str(cmd)
    if cmd:
        if _looks_like_curl_to_llm(cmd):
            return "llm_call"
        first = _shell_first_token(cmd)
        if first in LLM_CALL_WHITELIST["shell_clis"]:
            return "llm_call"
        if first in RAG_QUERY_WHITELIST["shell_clis"]:
            return "rag_query"

    # 5) 启发式回退：tool_name 子串命中 RAG 关键词
    if name_lower and any(kw in name_lower for kw in RAG_HEURISTIC_KW):
        return "rag_query"

    return None


# ─── prompt 抽取 ───────────────────────────────────────────

# 优先级从高到低：messages 是 OpenAI/Anthropic chat 协议的主载体；prompt/system
# 是 legacy completions；query/question/search* 是 RAG / Web Search 工具常用字段。
# objective / search_queries 是 Cursor WebSearch 的真实字段名；docid 是 user-iWiki
# getDocument 这类按 ID 取文档的入参（也用于评估 RAG"取了什么"）。
PROMPT_KEY_PRIORITY: Tuple[str, ...] = (
    "messages", "prompt", "system", "instructions",
    "objective", "search_queries",
    "query", "question", "search", "search_query", "q",
    "docid", "doc_id", "url",
    "text", "input", "content",
)
PREVIEW_LIMIT = 1024


def _stringify_prompt_value(val: Any) -> str:
    """把 messages / prompt 字段拍成单串字符串，专给前端预览用。"""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        chunks: List[str] = []
        for m in val:
            if isinstance(m, dict):
                role = m.get("role") or m.get("type") or ""
                content = m.get("content")
                if content is None:
                    content = m.get("text") or ""
                # OpenAI 多模态 content blocks: [{type:'text', text:'...'}, ...]
                if isinstance(content, list):
                    parts: List[str] = []
                    for c in content:
                        if isinstance(c, dict):
                            parts.append(str(c.get("text") or c.get("content") or ""))
                        else:
                            parts.append(str(c))
                    content = " ".join(p for p in parts if p)
                chunks.append(f"[{role}] {content}".strip())
            else:
                chunks.append(str(m))
        return "\n".join(chunks)
    if isinstance(val, dict):
        try:
            return json.dumps(val, ensure_ascii=False, indent=2)
        except Exception:
            return str(val)
    return str(val)


def _walk_prompt_keys(d: Dict, depth: int = 0) -> Tuple[str, int]:
    """按优先级在 d 里找最像 prompt 的字段；找到就返回 (preview, full_chars)。"""
    if depth > 2 or not isinstance(d, dict):
        return "", 0
    for key in PROMPT_KEY_PRIORITY:
        if key in d and d[key]:
            text = _stringify_prompt_value(d[key]).strip()
            if text:
                full = len(text)
                preview = (
                    text if full <= PREVIEW_LIMIT
                    else text[:PREVIEW_LIMIT] + f"…[+{full - PREVIEW_LIMIT}ch]"
                )
                return preview, full
    # 嵌套兜底：常见包装字段
    for outer in ("arguments", "params", "body", "data", "kwargs", "payload"):
        sub = d.get(outer)
        if isinstance(sub, dict):
            preview, full = _walk_prompt_keys(sub, depth=depth + 1)
            if preview:
                return preview, full
    return "", 0


def extract_prompt(tool_input: Optional[Dict]) -> Tuple[str, int]:
    """从 tool_input 里抽完整 prompt 内容。

    Returns
    -------
    (preview, full_chars)
        ``preview`` 是用于前端展示的字符串（超过 ``PREVIEW_LIMIT`` 会带尾巴
        ``…[+Nch]``），``full_chars`` 是原始字符数。两个都为空表示没找到。
    """
    if not isinstance(tool_input, dict):
        return "", 0
    return _walk_prompt_keys(tool_input)


# ─── 召回片段抽取（仅 rag_query / web_search 用） ─────────

RECALL_LIMIT = 2048


# synthetic 占位文本前缀 —— 这些是 cot_extractor 在 transcript 没有 tool_result
# 时填的"伪内容"，绝对不能当真实 RAG 召回输出
_SYNTHETIC_PLACEHOLDER_MARKERS = (
    "(工具执行结果未记录",
    "（工具执行结果未记录",
    "tool_result not captured",
    "no tool_result",
)


def _is_synthetic_placeholder(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if not t:
        return True
    return any(t.startswith(m) for m in _SYNTHETIC_PLACEHOLDER_MARKERS)


def extract_recall(
    tool_input: Optional[Dict] = None,
    observed_output: Optional[Dict] = None,
    content: Optional[str] = None,
) -> str:
    """对 RAG / Web Search 步骤，挑回的结果文本作为前端预览。

    优先级：
    1. ``observed_output.result_text``（cot-stream.js 解析 MCP result_json 后的纯文本）
    2. ``observed_output.stdout`` / ``output`` / ``result``（shell / 通用回灌）
    3. ``content``（transcript 落到 step 上的 tool_result 文本，且不是 synthetic 占位）
    4. ``tool_input.{result, results, documents, chunks}``（极少数 transcript inline 召回）

    返回截断到 ``RECALL_LIMIT`` 的预览串；找不到则返回空串。
    会**显式排除** cot_extractor 写下的 synthetic 占位文本（"(工具执行结果未记录…"），
    避免前端把伪结果当成真召回展示。
    """
    candidates: List[str] = []

    if isinstance(observed_output, dict):
        # 把 MCP result_text 放最前 —— 这是 cot-stream.js 从 result_json 拆出来的纯文本
        for k in ("result_text", "stdout", "result", "output", "data", "response"):
            v = observed_output.get(k)
            if isinstance(v, str) and v.strip():
                candidates.append(v)
                break

    if (
        content
        and isinstance(content, str)
        and not _is_synthetic_placeholder(content)
    ):
        candidates.append(content)

    if isinstance(tool_input, dict):
        for k in ("result", "results", "documents", "chunks", "tool_use_result"):
            v = tool_input.get(k)
            if v:
                candidates.append(_stringify_prompt_value(v))
                break

    for c in candidates:
        text = c.strip()
        if not text or _is_synthetic_placeholder(text):
            continue
        # v0.20.11：识别"工具明确返回了'空结果'但本身有效"的情况——例如
        # MCP `mcp__shadow-folk__search_memory` 在 query 没命中时返回字面量
        # `[]`、有些工具返回 `{}` 或 `null`。这些字符串虽然有效但对用户来说
        # 没有可读内容，前端如实渲染会让人误以为"程序拿不到召回"。这里把
        # 它们识别成"empty_recall"，返回 ``__EMPTY_RECALL__`` 哨兵，由
        # ``_propagate_invocation_to_executions`` 走专门的 unavailable_reason
        # 路径，前端渲染成"⚠️ RAG 命中 0 条结果"。
        if _is_empty_payload(text):
            return "__EMPTY_RECALL__"
        full = len(text)
        return (
            text if full <= RECALL_LIMIT
            else text[:RECALL_LIMIT] + f"…[+{full - RECALL_LIMIT}ch]"
        )
    return ""


def _is_empty_payload(text: str) -> bool:
    """判断文本是不是"工具有效返回但内容为空"的标记。

    覆盖三种典型 MCP / RAG / Web 工具空结果约定：
      * JSON 空数组：``[]``
      * JSON 空对象：``{}``
      * JSON null：``null``
    去除前后空白后做严格相等匹配，避免误判（例如 ``[a]`` 不算空）。
    """
    t = text.strip()
    return t in ("[]", "{}", "null")


def diagnose_recall_unavailable(
    observed_output: Optional[Dict],
    synthetic: bool,
    recall_preview: Optional[str] = None,
) -> str:
    """无法抽到真实召回时，给前端一句话解释**为什么没有**。

    返回值会被 ``_propagate_invocation_to_executions`` 写到
    ``metadata.recall_unavailable_reason``，前端在缺 recall_preview 时
    展示"暂未捕获，原因：…"代替"什么都没有"。

    v0.20.11：``recall_preview="__EMPTY_RECALL__"`` 哨兵代表"工具明确
    返回 [] / {} / null 等空结果"，给一条专门的友好提示，跟"实时事件流
    缺失"区分开。
    """
    if recall_preview == "__EMPTY_RECALL__":
        return "RAG 命中 0 条结果（工具返回空数组/对象，说明该 query 在知识库中无匹配）。"
    if observed_output and isinstance(observed_output, dict) and (
        observed_output.get("result_text")
        or observed_output.get("stdout")
        or observed_output.get("result")
    ):
        return ""  # 有数据，根本不需要解释
    if synthetic:
        return (
            "Cursor transcript 不返回 MCP / WebSearch 的 tool_result，且 "
            "cot-stream.js 在此调用发生时未挂载（events.jsonl 缺对应 "
            "afterMCPExecution / afterShellExecution 事件）。新会话已修复，"
            "此条属于历史调用。"
        )
    return "实时事件流缺失对应 after* 事件，未能回灌真实返回内容。"


__all__ = [
    "LLM_CALL_WHITELIST",
    "RAG_QUERY_WHITELIST",
    "WEB_SEARCH_TOOLS",
    "RAG_HEURISTIC_KW",
    "PREVIEW_LIMIT",
    "RECALL_LIMIT",
    "classify",
    "extract_prompt",
    "extract_recall",
    "diagnose_recall_unavailable",
]
