"""Agent 临时脚本 / 文件产物追踪器（v0.9.0 — L5 Execution Trace 落地）

背景
----
Agent 在执行任务时常会自己写一些 ``.py``、``.sh``、``.cjs`` 等"工作台脚本"
（典型如 ``_verify_v080.py``、``_audit_files.cjs``）—— 创建 → 运行 → 验证
→ 有时直接删除。这是 CoT 完整性中 L5（Execution Trace）的核心场景，但
``cot_extractor.py`` 之前只把 ``Write`` / ``StrReplace`` / ``Delete`` 记录成
普通的 ``tool_decision`` + ``tool_execution`` 步骤，前端无法快速看到：

1. 这一轮 agent 究竟创建了多少个临时文件？哪些是验证脚本？
2. 创建之后是否被 ``Shell python xxx.py`` 真正执行过？
3. 是否在 session 末尾被删除（"用完即焚"模式）？

本模块负责：

- 扫描所有 ``Write`` / ``StrReplace`` / ``MultiEdit`` / ``Delete`` 的
  ``tool_decision`` 步骤，按 ``path`` 聚合成 ``ScriptArtifact``。
- 对每个 ``Shell`` 步骤的 ``command`` 字段做反向查找——把"哪一步运行了
  哪个脚本"写回 ``ScriptArtifact.executed_at_steps`` + 对应 Shell 步的
  ``metadata.executed_artifact``。
- 启发式打 ``is_temp`` 标记（见 ``_classify_temp_script``）。
- 在每一个相关 step 的 ``metadata`` 上写 ``file_op`` 子对象，方便前端在
  StepDetail / SpanTree 直接拿到"文件操作摘要"而无需重复解析 tool_input。
- 聚合 session 级 ``ScriptStats``，挂到 ``SessionCoT.script_stats``。

设计选择
--------
- **保守启发式**：``is_temp`` 只看文件名（前缀 ``_``、含 ``verify``/``test_``/
  ``audit``/``tmp``/``debug``/``scratch``）+ 路径关键字（``output/reports``、
  ``.tmp``、``scratch``）。误杀好过漏判；如果一个文件最终被 ``Delete``
  也强制 ``is_temp=True``。
- **执行匹配只用绝对路径或 basename**：避免 ``python -m foo`` 这种没出现
  完整路径的命令被误配。
- **只读，不修改 step.content**：所有信息写到 ``metadata.file_op``，与
  现有 ``tool_input`` 并存，向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional


# ─── 配置 ──────────────────────────────────────────────────

# 视为"文件写入"的工具名（产生 / 修改文件）
_WRITE_TOOLS = {"Write", "StrReplace", "MultiEdit", "Edit"}
# 视为"文件删除"
_DELETE_TOOLS = {"Delete"}
# 临时脚本启发式：basename 前缀
_TEMP_PREFIXES = ("_",)
# 临时脚本启发式：basename 关键词（小写比对）
_TEMP_KEYWORDS = (
    "verify", "audit", "scratch", "debug", "scrap",
    "tmp", "temp", "check_", "diag_", "diagnose",
    "_test_", "test_", "probe",
)
# 临时脚本启发式：路径段关键词
_TEMP_PATH_FRAGMENTS = (
    "output/reports", "output\\reports",
    ".tmp", "/tmp/", "\\tmp\\",
    "scratch", "/scrap/", "\\scrap\\",
    "playground",
)
# 文件扩展名 → 语言标签（用于前端 chip 颜色）
_EXT_TO_LANG = {
    ".py": "python", ".pyw": "python",
    ".sh": "shell", ".bash": "shell",
    ".ps1": "powershell",
    ".js": "javascript", ".cjs": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".jsx": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".html": "html", ".css": "css",
    ".sql": "sql",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".rb": "ruby",
    ".txt": "text", ".log": "text",
}
# 视为"可执行脚本语言"——只有这些扩展名我们才会去 Shell 命令里反查执行
_EXECUTABLE_EXTS = {".py", ".pyw", ".sh", ".bash", ".ps1", ".js", ".cjs", ".mjs", ".ts"}


# ─── 数据结构 ──────────────────────────────────────────────

@dataclass
class FileEditEvent:
    """单次 file 操作事件——Write / StrReplace / Delete 各算一条。"""
    step_index: int
    turn_index: int
    timestamp: Optional[str]
    kind: str                  # 'create' | 'modify' | 'delete'
    tool_name: str
    edits_count: int = 1
    added_lines: int = 0
    removed_lines: int = 0
    content_chars: int = 0     # Write 内容长度 / StrReplace new_string 长度

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ScriptArtifact:
    """会话期间被 agent 触碰过的某个文件路径的完整生命周期。"""
    path: str
    basename: str
    extension: str
    language: str
    is_temp: bool
    purpose_hint: str = ""
    lifecycle: str = "modified"        # see _compute_lifecycle()
    first_seen_step: int = 0
    last_seen_step: int = 0
    created_at_step: Optional[int] = None
    deleted_at_step: Optional[int] = None
    edit_count: int = 0                # Write + StrReplace + MultiEdit
    delete_count: int = 0
    executed_at_steps: List[int] = field(default_factory=list)
    total_added_lines: int = 0
    total_removed_lines: int = 0
    last_content_chars: int = 0        # 最后一次 Write 的字符长度
    timeline: List[FileEditEvent] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["timeline"] = [t.to_dict() if isinstance(t, FileEditEvent) else t for t in self.timeline]
        return d


@dataclass
class ScriptStats:
    """session 级聚合——前端"概览"区一眼能看到的数字。"""
    total_artifacts: int = 0
    total_writes: int = 0
    total_strreplaces: int = 0
    total_deletes: int = 0
    total_executions: int = 0          # Σ executed_at_steps 长度
    temp_scripts: int = 0
    executed_temp_scripts: int = 0
    deleted_temp_scripts: int = 0
    extensions: Dict[str, int] = field(default_factory=dict)
    languages: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ─── 工具函数 ──────────────────────────────────────────────

def _normalize_path(p: str) -> str:
    """统一斜杠 + 小写盘符方便比对（仅用于 key）。"""
    if not p:
        return ""
    s = str(p).replace("\\", "/").strip()
    if len(s) >= 2 and s[1] == ":":
        s = s[0].lower() + s[1:]
    return s


def _basename(p: str) -> str:
    if not p:
        return ""
    s = str(p).replace("\\", "/")
    return s.rsplit("/", 1)[-1] if "/" in s else s


def _extension(p: str) -> str:
    bn = _basename(p)
    if "." not in bn:
        return ""
    ext = "." + bn.rsplit(".", 1)[-1].lower()
    return ext


def _count_lines(s: Any) -> int:
    if not isinstance(s, str):
        return 0
    if not s:
        return 0
    n = s.count("\n")
    if not s.endswith("\n"):
        n += 1
    return n


def _classify_temp_script(path: str, was_deleted_in_session: bool = False) -> bool:
    """启发式：是否为"临时脚本/草稿产物"。

    用于前端在 SessionDetail 高亮"📜 临时验证脚本 ×N"。误判好过漏判。
    """
    if not path:
        return False
    bn = _basename(path).lower()
    np = _normalize_path(path).lower()
    if any(bn.startswith(p) for p in _TEMP_PREFIXES):
        return True
    if any(kw in bn for kw in _TEMP_KEYWORDS):
        return True
    if any(frag in np for frag in _TEMP_PATH_FRAGMENTS):
        return True
    # 在同一 session 中创建后又删除：经典"用完即焚"
    if was_deleted_in_session:
        return True
    return False


def _purpose_hint_from_content(content: Any, fallback: str = "") -> str:
    """尝试从 Write 内容首行 docstring / 注释里提取一句话用途说明。"""
    if not isinstance(content, str) or not content.strip():
        return fallback
    lines = content.lstrip().splitlines()
    for ln in lines[:6]:
        s = ln.strip()
        if not s:
            continue
        # python docstring / 注释
        if s.startswith('"""') or s.startswith("'''"):
            stripped = s.strip('"').strip("'").strip()
            if stripped:
                return stripped[:140]
            continue
        if s.startswith("#"):
            t = s.lstrip("# ").strip()
            if t:
                return t[:140]
        # JS / TS 注释
        if s.startswith("//"):
            t = s.lstrip("/ ").strip()
            if t:
                return t[:140]
        if s.startswith("/*"):
            t = s.lstrip("/*").rstrip("*/").strip()
            if t:
                return t[:140]
    return fallback


def _purpose_hint_from_basename(basename: str) -> str:
    """没有内容时，从文件名 stem 反推一个粗略用途。"""
    if not basename:
        return ""
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    # _verify_v080 → "verify v080"
    return stem.lstrip("_").replace("_", " ").replace("-", " ").strip()


def _compute_lifecycle(art: "ScriptArtifact") -> str:
    """根据 timeline 给 artifact 画一个"生命周期标签"。"""
    has_create = art.created_at_step is not None
    has_delete = art.deleted_at_step is not None
    has_exec = bool(art.executed_at_steps)
    has_modify = art.edit_count > (1 if has_create else 0)

    if has_create and has_exec and has_delete:
        return "created_executed_deleted"
    if has_create and has_delete:
        return "created_deleted"
    if has_create and has_exec and has_modify:
        return "created_modified_executed"
    if has_create and has_exec:
        return "created_executed"
    if has_create and has_modify:
        return "created_modified"
    if has_create:
        return "created"
    if has_delete and not has_create:
        return "deleted_only"
    return "modified_only"


# ─── 主提取入口 ──────────────────────────────────────────────

def build_script_artifacts(turns_cot: Iterable[Any]) -> Dict[str, Any]:
    """扫描所有 turn → 产出 ScriptArtifact 列表 + ScriptStats + 给 step 写 file_op 元数据。

    Args:
        turns_cot: 已 attach_cursor_events 之后的 TurnCoT 列表。

    Returns:
        ``{"artifacts": [ScriptArtifact, ...], "stats": ScriptStats}``。
        同时**会就地修改**输入的 step.metadata：

        - 对 Write/StrReplace/Delete 的 tool_decision 步骤：
          ``metadata.file_op = {kind, path, basename, extension, language,
          is_temp, edits_count, added_lines, removed_lines, purpose_hint,
          temp_artifact_id}``
        - 对 Shell tool_decision 步骤如果 command 里识别到某个 artifact：
          ``metadata.executed_artifact = {path, basename, language,
          is_temp, artifact_id}``
    """
    # Pass 1：收集所有 file ops，先得到每个 path 的总览（特别是是否被删过）
    all_file_ops: List[Dict] = []
    for turn in turns_cot:
        for s in getattr(turn, "steps", []) or []:
            tn = (getattr(s, "tool_name", "") or "").strip()
            if getattr(s, "step_type", "") != "tool_decision":
                continue
            if tn not in _WRITE_TOOLS and tn not in _DELETE_TOOLS:
                continue
            tool_input = (getattr(s, "metadata", {}) or {}).get("tool_input") or {}
            path = tool_input.get("path") or tool_input.get("file_path")
            if not path:
                continue
            kind = "delete" if tn in _DELETE_TOOLS else (
                "create" if tn == "Write" else "modify"
            )
            ev = {
                "step": s,
                "turn": turn,
                "tool_name": tn,
                "kind": kind,
                "path": str(path),
                "tool_input": tool_input,
            }
            all_file_ops.append(ev)

    # Pass 2：哪些 path 在 session 中被删过——用来给同 path 的 temp 判定打补丁
    deleted_paths_norm: set = {
        _normalize_path(op["path"]) for op in all_file_ops if op["kind"] == "delete"
    }

    # Pass 3：聚合成 ScriptArtifact
    artifacts: Dict[str, ScriptArtifact] = {}

    def _ensure_artifact(path: str) -> ScriptArtifact:
        key = _normalize_path(path)
        art = artifacts.get(key)
        if art is None:
            ext = _extension(path)
            lang = _EXT_TO_LANG.get(ext, "unknown" if ext else "unknown")
            bn = _basename(path)
            was_deleted = key in deleted_paths_norm
            art = ScriptArtifact(
                path=str(path),
                basename=bn,
                extension=ext,
                language=lang,
                is_temp=_classify_temp_script(path, was_deleted_in_session=was_deleted),
                purpose_hint=_purpose_hint_from_basename(bn),
            )
            artifacts[key] = art
        return art

    for op in all_file_ops:
        s = op["step"]
        turn = op["turn"]
        tn = op["tool_name"]
        kind = op["kind"]
        path = op["path"]
        tool_input = op["tool_input"] or {}
        art = _ensure_artifact(path)

        step_index = int(getattr(s, "step_index", 0) or 0)
        turn_index = int(getattr(turn, "turn_index", 0) or 0)
        ts = getattr(s, "timestamp", None)

        added = removed = 0
        content_chars = 0
        edits_count = 1

        if tn == "Write":
            content = tool_input.get("contents") or tool_input.get("content") or ""
            content_chars = len(content) if isinstance(content, str) else 0
            added = _count_lines(content)
            removed = 0
            # 第一次 Write 视作 create；后续重写也算 modify
            if art.created_at_step is None:
                art.created_at_step = step_index
                kind_actual = "create"
                art.edit_count += 1
            else:
                kind_actual = "modify"
                art.edit_count += 1
            # 记录最后一次写入的 size，方便前端展示"当前 X 行"
            art.last_content_chars = content_chars
            # 用首行注释精修 purpose_hint
            hint = _purpose_hint_from_content(content, fallback=art.purpose_hint)
            if hint:
                art.purpose_hint = hint
            kind = kind_actual
        elif tn in ("StrReplace", "MultiEdit", "Edit"):
            old_s = tool_input.get("old_string") or ""
            new_s = tool_input.get("new_string") or ""
            removed = _count_lines(old_s)
            added = _count_lines(new_s)
            content_chars = len(new_s) if isinstance(new_s, str) else 0
            art.edit_count += 1
            kind = "modify"
        elif tn == "Delete":
            art.delete_count += 1
            if art.deleted_at_step is None:
                art.deleted_at_step = step_index
            kind = "delete"

        art.total_added_lines += added
        art.total_removed_lines += removed
        if art.first_seen_step == 0 or step_index < art.first_seen_step:
            art.first_seen_step = step_index
        if step_index > art.last_seen_step:
            art.last_seen_step = step_index

        ev = FileEditEvent(
            step_index=step_index,
            turn_index=turn_index,
            timestamp=ts,
            kind=kind,
            tool_name=tn,
            edits_count=edits_count,
            added_lines=added,
            removed_lines=removed,
            content_chars=content_chars,
        )
        art.timeline.append(ev)

        # 把 file_op 元数据写回 tool_decision step（前端用）
        md = dict(getattr(s, "metadata", {}) or {})
        md["file_op"] = {
            "kind": kind,
            "tool_name": tn,
            "path": art.path,
            "basename": art.basename,
            "extension": art.extension,
            "language": art.language,
            "is_temp": art.is_temp,
            "edits_count": edits_count,
            "added_lines": added,
            "removed_lines": removed,
            "content_chars": content_chars,
            "artifact_key": _normalize_path(art.path),
        }
        s.metadata = md

    # Pass 4：扫 Shell 命令，把"哪一步运行了哪个脚本"落到 step + artifact
    if artifacts:
        # 准备一个查询表：normalized_path → artifact，basename → [artifact]
        path_to_art: Dict[str, ScriptArtifact] = dict(artifacts)
        basename_to_arts: Dict[str, List[ScriptArtifact]] = {}
        for art in artifacts.values():
            if art.extension and art.extension in _EXECUTABLE_EXTS:
                basename_to_arts.setdefault(art.basename, []).append(art)

        for turn in turns_cot:
            for s in getattr(turn, "steps", []) or []:
                if getattr(s, "step_type", "") != "tool_decision":
                    continue
                tn = (getattr(s, "tool_name", "") or "").strip()
                if tn != "Shell":
                    continue
                tool_input = (getattr(s, "metadata", {}) or {}).get("tool_input") or {}
                cmd = tool_input.get("command") or ""
                if not isinstance(cmd, str) or not cmd:
                    continue
                cmd_norm = cmd.replace("\\", "/")
                cmd_lower = cmd_norm.lower()

                matched: Optional[ScriptArtifact] = None
                # 精确：完整路径子串命中
                for key, art in path_to_art.items():
                    if not art.extension or art.extension not in _EXECUTABLE_EXTS:
                        continue
                    if key and key in cmd_lower:
                        matched = art
                        break
                # 退化：basename 作为独立 token
                if matched is None:
                    import re as _re
                    for bn, arts in basename_to_arts.items():
                        # \b 防 'foo.py' 匹配到 'foobar.py'
                        if _re.search(r"(?<![A-Za-z0-9_/])" + _re.escape(bn) + r"(?![A-Za-z0-9_])", cmd_norm):
                            matched = arts[0]
                            break
                if matched is None:
                    continue

                step_index = int(getattr(s, "step_index", 0) or 0)
                if step_index and step_index not in matched.executed_at_steps:
                    matched.executed_at_steps.append(step_index)

                md = dict(getattr(s, "metadata", {}) or {})
                md["executed_artifact"] = {
                    "path": matched.path,
                    "basename": matched.basename,
                    "language": matched.language,
                    "extension": matched.extension,
                    "is_temp": matched.is_temp,
                    "artifact_key": _normalize_path(matched.path),
                }
                s.metadata = md
                # 同步到对应 tool_execution（按 tool_use_id）
                tu_id = getattr(s, "tool_use_id", "") or ""
                if tu_id:
                    for s2 in getattr(turn, "steps", []) or []:
                        if (getattr(s2, "step_type", "") == "tool_execution"
                                and (getattr(s2, "tool_use_id", "") or "") == tu_id):
                            md2 = dict(getattr(s2, "metadata", {}) or {})
                            md2["executed_artifact"] = md["executed_artifact"]
                            s2.metadata = md2
                            break

    # Pass 5：每个 artifact 收尾——重算 lifecycle（依赖 executed_at_steps）
    for art in artifacts.values():
        art.lifecycle = _compute_lifecycle(art)

    # Pass 6：聚合 ScriptStats
    stats = ScriptStats()
    for art in artifacts.values():
        stats.total_artifacts += 1
        for ev in art.timeline:
            if ev.tool_name == "Write":
                stats.total_writes += 1
            elif ev.tool_name in ("StrReplace", "MultiEdit", "Edit"):
                stats.total_strreplaces += 1
            elif ev.tool_name == "Delete":
                stats.total_deletes += 1
        stats.total_executions += len(art.executed_at_steps)
        if art.is_temp:
            stats.temp_scripts += 1
            if art.executed_at_steps:
                stats.executed_temp_scripts += 1
            if art.deleted_at_step is not None:
                stats.deleted_temp_scripts += 1
        if art.extension:
            stats.extensions[art.extension] = stats.extensions.get(art.extension, 0) + 1
        if art.language:
            stats.languages[art.language] = stats.languages.get(art.language, 0) + 1

    # 排序：临时脚本优先 + 首次出现晚的靠前（最近的更可能是用户当前关心的）
    artifacts_sorted = sorted(
        artifacts.values(),
        key=lambda a: (not a.is_temp, -a.first_seen_step),
    )

    return {"artifacts": artifacts_sorted, "stats": stats}


__all__ = [
    "FileEditEvent",
    "ScriptArtifact",
    "ScriptStats",
    "build_script_artifacts",
]
