#!/usr/bin/env python3
"""
子会话标题生成器 — 纯规则，不依赖 LLM。

目标效果：
  输入：用户原始消息（可能是一大段文字）
  输出：「用户XX：YYY」一句话意图描述（30~40 字）

实现思路：
  1) 抽取真正的提问文本（跳过 <image_files>/<system_reminder>/<attached_files>，
     若外层被 <user_query>...</user_query> 包裹，只取包裹内容）。
  2) 前缀识别：扫描关键词决定「用户想了解/用户反馈问题/用户希望/用户反馈」。
  3) 正文挑选：把文本按中文/英文标点切成若干片段，对每段打分（疑问词、问题词、
     靠前、长度合适都加分），取得分最高的片段作为正文。
  4) 清洗与限长：去起句修饰/行内填充词、不切断英文/数字单词。

任何异常都返回 None 或回退到"清洗首行"，确保不破坏主流程。
"""

from __future__ import annotations

import re
from typing import List, Optional

_STRIPPABLE = "「」『』\"'“”《》【】()（）[]{}。！？!?.:：;；、， \t"

# ── 起句修饰（多轮剥离） ──────────────────────────────────
_LEADING = [
    "有一个问题，就是", "有一个问题", "有个问题",
    "现在我想", "现在", "目前", "我想",
    "对于", "关于", "针对",
    "请你", "请帮我", "请", "帮我", "帮忙", "麻烦你", "麻烦",
    "另外", "然后", "接下来", "接着",
    "那么",
    "就是",
]
_FILLERS = [
    "这样", "这个", "这些", "那么", "那个", "这里", "那里",
    "然后呢", "的时候",
]

# ── 意图识别关键词 ────────────────────────────────────────
_QUESTION_MARKERS = [
    "为什么", "怎么", "如何", "是否", "能不能", "可不可以",
    "什么原因", "什么意思", "什么", "？", "?",
]
_QUESTION_TAIL = ["吗", "呢", "么"]   # 句末语气词
_REQUEST_MARKERS = [
    "帮我", "请你", "请帮我", "麻烦你", "你来", "让你",
    "你需要", "你要", "你可以", "直接", "把这个", "把它",
    "改成", "改为", "修改", "实现", "生成", "写一个",
    "做一个", "集成", "安装", "部署",
]
_PROBLEM_MARKERS = [
    "报错", "错误", "失败", "bug", "BUG", "问题", "不工作",
    "无法", "不能", "卡死", "崩溃", "异常", "没反应", "出错",
]
_CONFIRM_MARKERS = [
    "确认", "是不是", "是否", "有没有", "要不要", "能否",
]
_FEEDBACK_MARKERS = [
    "通了", "成功", "完成", "搞定", "好的", "已经", "解决了",
    "搞好了", "OK", "ok",
]


# ── 文本抽取 ──────────────────────────────────────────────
def _extract_user_query_core(user_query: str, limit: int = 1200) -> str:
    """
    剥去 <user_query>/<image_files>/<attached_files>/<system_reminder> 等包装，
    仅保留纯文本。
    """
    if not user_query:
        return ""
    m = re.search(r"<user_query>\s*(.+?)\s*</user_query>", user_query, re.DOTALL)
    text = m.group(1).strip() if m else user_query
    text = re.sub(r"<image_files>.*?</image_files>", "", text, flags=re.DOTALL)
    text = re.sub(r"<attached_files>.*?</attached_files>", "", text, flags=re.DOTALL)
    text = re.sub(r"<system_reminder>.*?</system_reminder>", "", text, flags=re.DOTALL)
    text = re.sub(r"<open_and_recently_viewed_files>.*?</open_and_recently_viewed_files>",
                  "", text, flags=re.DOTALL)
    # 常见样板行
    text = re.sub(r"^\s*\[Image\]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*The following images? (?:was|were) provided.*$", "",
                  text, flags=re.MULTILINE | re.IGNORECASE)
    return text.strip()[:limit]


def _strip_leading(t: str) -> str:
    changed = True
    while changed and t:
        changed = False
        for p in sorted(_LEADING, key=lambda x: -len(x)):
            if t.startswith(p):
                t = t[len(p):].lstrip("：:，。, \t")
                changed = True
                break
    return t


# ── Smart truncate（不切断英文/数字单词） ─────────────────
_WORD_CHAR = re.compile(r"[A-Za-z0-9_\-]")


def _smart_truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    cut = max_chars
    if (
        cut < len(s)
        and _WORD_CHAR.match(s[cut - 1])
        and _WORD_CHAR.match(s[cut])
    ):
        i = cut - 1
        while i > 0 and _WORD_CHAR.match(s[i]):
            i -= 1
        if i > 0:
            candidate = s[: i + 1].rstrip(" -_")
            if len(candidate) >= 5:
                return candidate + "…"
        j = cut
        hard_cap = max_chars + 14
        while j < len(s) and j < hard_cap and _WORD_CHAR.match(s[j]):
            j += 1
        suffix = "…" if j < len(s) else ""
        return s[:j] + suffix
    return s[:cut] + "…"


# ── 意图识别 ──────────────────────────────────────────────
def _detect_intent_prefix(text: str) -> str:
    """
    根据关键词扫描返回前缀（含冒号分隔符交由外部拼接）。
    优先级：报错/问题 > 确认 > 疑问 > 请求 > 反馈
    """
    head = text[:100]  # 前 100 字权重高

    if any(m in text for m in _PROBLEM_MARKERS):
        return "用户反馈问题"
    if any(m in head for m in _CONFIRM_MARKERS):
        return "用户想确认"
    if any(m in text for m in _QUESTION_MARKERS) or \
       any(text.rstrip(_STRIPPABLE).endswith(t) for t in _QUESTION_TAIL):
        return "用户想了解"
    if any(m in text for m in _REQUEST_MARKERS):
        return "用户希望"
    if any(m in text for m in _FEEDBACK_MARKERS):
        return "用户反馈"
    return "用户反馈"


# ── 正文挑选（打分制） ────────────────────────────────────
_KEY_WORDS = _QUESTION_MARKERS + _PROBLEM_MARKERS + _REQUEST_MARKERS + _CONFIRM_MARKERS


def _score_segment(seg: str, idx: int, total_segs: int) -> float:
    """
    为句片段打分。
    - 长度在 [8, 50] 为最佳，越偏离扣分
    - 含关键词加分
    - 越靠前加分（前 30% 位置加分）
    """
    L = len(seg)
    if L < 4:
        return -1.0
    score = 0.0
    # 长度
    if 8 <= L <= 60:
        score += 2.0
    elif L < 8:
        score += L / 4.0
    else:
        score += max(0.0, 3.0 - (L - 60) / 30.0)
    # 关键词
    for k in _KEY_WORDS:
        if k in seg:
            score += 1.2
            break
    # 位置（越靠前越好）
    if total_segs > 0:
        pos = idx / total_segs
        if pos < 0.3:
            score += 1.5
        elif pos < 0.6:
            score += 0.5
    return score


def _pick_main_segment(body: str) -> str:
    """把 body 按标点切片段，打分，取最高分片段。"""
    # 按硬分句标点切
    parts = re.split(r"[。！？；;\n]+", body)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return body.strip()

    # 如果首段非常长（>70 字）且含多个"，"，再按"，"进一步切开
    candidates: List[str] = []
    for p in parts:
        if len(p) > 70 and p.count("，") >= 2:
            sub = [s.strip() for s in p.split("，") if s.strip()]
            # 把相邻短小片段粘合在一起（避免切得太碎）
            merged: List[str] = []
            buf = ""
            for s in sub:
                if len(buf) + len(s) <= 45:
                    buf = (buf + "，" + s) if buf else s
                else:
                    if buf:
                        merged.append(buf)
                    buf = s
            if buf:
                merged.append(buf)
            candidates.extend(merged)
        else:
            candidates.append(p)

    if not candidates:
        return body.strip()

    scored = [(s, _score_segment(s, i, len(candidates))) for i, s in enumerate(candidates)]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]
    return best


# ── 对外入口 ──────────────────────────────────────────────
def generate_intent_summary(
    user_query: str,
    final_response_preview: str = "",
    tool_names: Optional[List[str]] = None,
    max_body_chars: int = 34,
) -> Optional[str]:
    """
    生成「用户XX：正文」格式的意图概要。
    失败时返回 None。
    """
    _ = (final_response_preview, tool_names)  # 保留参数兼容未来扩展（可接入工具倾向）
    core = _extract_user_query_core(user_query)
    if not core:
        return None

    prefix = _detect_intent_prefix(core)

    # 正文预处理
    body = _strip_leading(core)
    for f in sorted(_FILLERS, key=lambda x: -len(x)):
        body = body.replace(f, "")
    body = re.sub(r"\s+", " ", body).strip(_STRIPPABLE)
    if not body:
        return None

    main = _pick_main_segment(body)
    main = _strip_leading(main).strip(_STRIPPABLE)
    if not main:
        main = body

    main = _smart_truncate(main, max_body_chars)
    if not main:
        return None
    return f"{prefix}：{main}"


# 兼容老入口（若其他代码仍在 import generate_short_title）
def generate_short_title(*args, **kwargs) -> Optional[str]:
    return generate_intent_summary(*args, **kwargs)


__all__ = [
    "generate_intent_summary",
    "generate_short_title",
    "_extract_user_query_core",
    "_smart_truncate",
]
