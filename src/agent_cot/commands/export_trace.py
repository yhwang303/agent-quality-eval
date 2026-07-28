"""``agent-cot export-trace`` — 把一个会话的完整 trace 导出到文件或 stdout。

给闭环 harness 用：脚本不需要起 dashboard 就能拿到上一轮执行的完整 trace，
喂给下一轮的 agent 去判断优化方向。
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from agent_cot.commands.otlp_bridge import OtlpBridgeError, resolve_cot_json
from agent_cot.trace import SUPPORTED_FORMATS, export_session_trace
from agent_cot.trace.otel_source import load_otel_events


def run_export(
    *,
    session_id: str | None,
    cot_path: str | None,
    fmt: str,
    output: str | None,
    quiet: bool = False,
) -> int:
    """解析一个会话的 cot.json，拍平并写出。返回进程退出码。"""
    if fmt not in SUPPORTED_FORMATS:
        click.secho(
            f"error: 不支持的导出格式 {fmt!r}，可选：{', '.join(SUPPORTED_FORMATS)}",
            fg="red",
            err=True,
        )
        return 2

    try:
        resolved = resolve_cot_json(session_id=session_id, cot_path=cot_path)
    except OtlpBridgeError as exc:
        click.secho(f"error: {exc}", fg="red", err=True)
        return 1

    # OTel 只是补充数据源（Claude subagent 内部的工具调用），缺了不该阻断导出
    otel = load_otel_events(resolved.session_id)

    try:
        result = export_session_trace(resolved.raw, fmt=fmt, otel=otel)
    except Exception as exc:
        click.secho(f"error: trace 导出失败：{type(exc).__name__}: {exc}", fg="red", err=True)
        return 1

    if output in (None, "-"):
        # stdout 走 buffer 写，避免 Windows 控制台编码把 UTF-8 内容打坏
        sys.stdout.buffer.write(result["content"].encode("utf-8"))
        return 0

    target = Path(output).expanduser()
    if target.is_dir():
        target = target / result["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # newline="" 关掉 Windows 上的 CRLF 转换：jsonl 是「一行一个事件」的机器
        # 格式，CLI 和 HTTP 两条路径导出同一个会话必须字节一致，否则下游做
        # diff / 哈希比对时会凭空多出一堆假差异。
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(result["content"])
    except OSError as exc:
        click.secho(f"error: 写入 {target} 失败：{exc}", fg="red", err=True)
        return 1

    if not quiet:
        click.secho("agent-cot export-trace — done", bold=True, fg="green")
        click.echo(f"  session : {result['session_id']}")
        click.echo(f"  source  : {resolved.path}")
        click.echo(f"  events  : {result['event_count']}")
        click.echo(f"  schema  : {result['schema']}")
        click.echo(f"  written : {target}")
        if otel:
            click.echo(
                click.style(
                    f"  otel    : merged {len(otel['events'])} native OTel events",
                    dim=True,
                )
            )
    return 0
