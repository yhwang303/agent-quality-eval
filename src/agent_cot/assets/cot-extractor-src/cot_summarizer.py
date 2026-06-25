#!/usr/bin/env python3
"""
CoT Summarizer — 用智谱 GLM-4.7-Flash API 为每个 Turn 生成标准思维链摘要

方案选择说明：
  - 使用智谱 GLM-4.7-Flash（完全免费，无需付费）
  - 智谱 API 兼容 OpenAI 格式，通过 requests 直接调用
  - 在 Stop hook 触发后异步调用，不阻塞主流程
  - 生成标准 CoT 格式：目标理解 → 信息收集 → 决策推理 → 执行验证 → 结论

配置方式：
  在项目根目录（cot-extractor/）创建 .env 文件，写入：
    ZHIPU_API_KEY=your_zhipu_api_key_here
  获取免费 API Key：https://open.bigmodel.cn/usercenter/apikeys
"""

import os
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .cot_extractor import TurnCoT

# ─── 加载 .env 文件 ───────────────────────────────────────

def _load_dotenv() -> None:
    """从 cot-extractor/.env 或项目根 .env 加载环境变量"""
    # 优先尝试 python-dotenv
    try:
        from dotenv import load_dotenv
        # 尝试多个可能的 .env 路径
        candidates = [
            Path(__file__).parent.parent / ".env",          # cot-extractor/.env
            Path(__file__).parent.parent.parent / ".env",   # 项目根 .env
        ]
        for p in candidates:
            if p.exists():
                load_dotenv(p, override=False)
                break
        return
    except ImportError:
        pass

    # 回退：手动解析 .env 文件
    candidates = [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass
        break


# 模块加载时自动读取 .env
_load_dotenv()


# ─── 智谱 API 配置 ────────────────────────────────────────

ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.7-flash"

# 429 速率限制重试配置
MAX_RETRIES = 5
RETRY_BASE_DELAY = 10  # 秒（智谱免费账户RPM限制较严格）


# ─── CoT 摘要 Prompt ──────────────────────────────────────

COT_SUMMARY_SYSTEM = """你是一个专业的 AI Agent 行为分析师。
你的任务是：根据 Claude Code Agent 在一个 Turn 中的完整行为序列，
生成一份符合标准思维链（Chain-of-Thought）格式的推理过程摘要。

输出格式（严格遵守，使用 Markdown）：

## 🎯 目标理解
[Claude 在这个 Turn 中要完成什么任务？从用户输入和上下文推断]

## 🔍 信息收集
[Claude 收集了哪些信息？读取了哪些文件/执行了哪些探索性命令？]
（如无信息收集步骤，写"无"）

## 💡 决策推理
[Claude 如何决定采取行动？遇到了什么问题？做了哪些判断？]
（重点：如有错误恢复或策略转换，详细描述推理过程）

## ⚙️ 执行过程
[Claude 具体执行了哪些操作？按顺序列出关键步骤]

## ✅ 结论
[这个 Turn 的最终结果是什么？是否成功完成目标？]

注意：
- 保持简洁，每个部分不超过 3-4 句话
- 重点突出推理过程，而非重复描述操作细节
- 如有错误恢复，在"决策推理"中重点分析
- 使用中文输出"""

COT_SUMMARY_USER_TMPL = """以下是 Claude Code Agent 在 Turn {turn_index} 中的完整行为序列：

{steps_text}

请根据以上行为序列，生成标准思维链摘要。"""


def _build_steps_text(turn: "TurnCoT") -> str:
    """将 Turn 的步骤序列转换为文本描述"""
    lines = []
    if turn.user_query:
        lines.append(f"[用户输入] {turn.user_query[:500]}")

    for step in turn.steps:
        st = step.step_type
        dur = f" ({step.duration_ms:.0f}ms)" if step.duration_ms else ""
        tok = f" [{step.tokens}t]" if step.tokens > 0 else ""

        if st == "user_input":
            continue  # 已在上面处理
        elif st == "tool_decision":
            tool = step.tool_name or step.metadata.get("tool_name", "?")
            inp = step.metadata.get("input_summary", "")[:200]
            lines.append(f"[决定调用工具]{dur} {tool}({inp})")
        elif st == "tool_execution":
            is_err = step.metadata.get("is_error", False)
            prefix = "[工具执行-错误]" if is_err else "[工具执行-成功]"
            lines.append(f"{prefix}{dur} {step.content[:300]}")
        elif st == "error_recovery":
            lines.append(f"[错误恢复]{dur} {step.content[:200]}")
        elif st == "strategy_shift":
            lines.append(f"[策略转换]{dur} {step.content}")
        elif st == "pre_tool_reasoning":
            lines.append(f"[决策说明]{dur}{tok} {step.content[:300]}")
        elif st in ("thinking_inter", "thinking_explicit"):
            lines.append(f"[推理过程]{dur}{tok} {step.content[:300]}")
        elif st == "final_response":
            lines.append(f"[最终回复]{dur}{tok} {step.content[:500]}")
        else:
            lines.append(f"[{st}]{dur} {step.content[:200]}")

    return "\n".join(lines)


def _call_zhipu_api(
    api_key: str,
    model: str,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> Optional[str]:
    """
    调用智谱 GLM API（OpenAI 兼容格式），内置 429 重试机制。

    智谱 API 文档：https://open.bigmodel.cn/dev/api
    接口地址：https://open.bigmodel.cn/api/paas/v4/chat/completions
    """
    import time
    import requests

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                ZHIPU_API_BASE,
                headers=headers,
                json=payload,
            timeout=90,
            )

            # 处理 429 速率限制：等待后重试
            if resp.status_code == 429:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                print(f"[CoT Summarizer] 429 速率限制，{delay}秒后重试 (第{attempt+1}/{MAX_RETRIES}次)...")
                time.sleep(delay)
                continue

            resp.raise_for_status()
            data = resp.json()

            # 解析 OpenAI 兼容格式的响应
            # GLM-4.7-Flash 是推理模型，会先在 reasoning_content 中思考，
            # 然后在 content 中输出最终结果
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "")

                # 优先返回 content（最终输出）
                if content and content.strip():
                    return content.strip()

                # 如果 content 为空但有 reasoning_content（推理模型特性），
                # 说明 max_tokens 不够，推理过程占满了 token
                # 此时使用 reasoning_content 作为回退
                if reasoning and reasoning.strip():
                    print(f"[CoT Summarizer] content 为空，使用 reasoning_content 作为回退")
                    return reasoning.strip()

            print(f"[CoT Summarizer] 智谱 API 返回无内容: {json.dumps(data, ensure_ascii=False)[:300]}")
            return None

        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            print(f"[CoT Summarizer] 智谱 API HTTP 错误: {last_error}")
            return None
        except requests.exceptions.Timeout:
            last_error = "请求超时"
            delay = RETRY_BASE_DELAY * (attempt + 1)
            print(f"[CoT Summarizer] 请求超时，{delay}秒后重试 (第{attempt+1}/{MAX_RETRIES}次)...")
            time.sleep(delay)
            continue
        except Exception as e:
            print(f"[CoT Summarizer] 智谱 API 调用失败: {type(e).__name__}: {e}")
            return None

    print(f"[CoT Summarizer] 达到最大重试次数({MAX_RETRIES})，最后错误: {last_error}")
    return None


def generate_turn_cot_summary(
    turn: "TurnCoT",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> Optional[str]:
    """
    为单个 Turn 生成 LLM CoT 摘要（使用智谱 GLM API）。

    Args:
        turn: TurnCoT 对象
        api_key: 智谱 API Key（默认从环境变量 ZHIPU_API_KEY 读取）
        model: 使用的模型（默认 glm-4.7-flash，完全免费）
        max_tokens: 最大输出 token 数（推理模型需要更多 token）

    Returns:
        CoT 摘要文本（Markdown 格式），失败时返回 None
    """
    key = api_key or os.environ.get("ZHIPU_API_KEY", "")
    if not key:
        return None

    steps_text = _build_steps_text(turn)
    if not steps_text.strip():
        return None

    user_prompt = COT_SUMMARY_USER_TMPL.format(
        turn_index=turn.turn_index,
        steps_text=steps_text,
    )

    try:
        result = _call_zhipu_api(
            api_key=key,
            model=model,
            max_tokens=max_tokens,
            system_prompt=COT_SUMMARY_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.3,
        )
        return result
    except Exception as e:
        print(f"[CoT Summarizer] Turn {turn.turn_index} 摘要生成失败: {type(e).__name__}: {e}")
        return None


def enrich_session_with_cot_summaries(
    session_cot,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    skip_simple_turns: bool = True,
) -> int:
    """
    为 session 中所有 Turn 生成 CoT 摘要（原地修改）。

    Args:
        session_cot: SessionCoT 对象
        api_key: 智谱 API Key（默认从环境变量读取）
        model: 使用的模型（默认 glm-4.7-flash）
        skip_simple_turns: 是否跳过简单 Turn（步骤数 <= 2 且无工具调用）

    Returns:
        成功生成摘要的 Turn 数量
    """
    count = 0
    for i, turn in enumerate(session_cot.turns):
        # 跳过过于简单的 Turn（节省 API 调用）
        if skip_simple_turns and turn.total_steps <= 2 and not turn.tool_calls:
            continue

        # 智谱免费账户 RPM 限制严格，每次请求间隔等待
        if i > 0 and count > 0:
            import time
            time.sleep(RETRY_BASE_DELAY)

        summary = generate_turn_cot_summary(turn, api_key=api_key, model=model)
        if summary:
            turn.cot_summary = summary
            count += 1
            print(f"[CoT Summarizer] Turn {turn.turn_index} 摘要生成完成 ({len(summary)} chars)")

    return count
