"""``agent-cot dedupe-thinking`` — 清掉已存盘会话里重复的 thinking 步骤。

**为什么需要这个命令**

重复的根因在提取环节：Cursor 会把同一段 reasoning 推两次 ``afterAgentThought``
hook（一次带裸 ``generation_id``，一次带加了后缀的），transcript 的 text block
又会再产出一份副本。修复已经落在 extractor 里，但它只对**之后**提取的数据生效
——已经写完的 cot.json 还带着旧的重复，前端照样把每条思考画两遍。

所以这条命令是给存量数据用的一次性清理：它调用的就是 extractor 里那两个去重
函数，不是另写一套判定逻辑，所以清理结果与重新提取一致。函数本身幂等，重复跑
不会越删越多。

**不做什么**：不碰 transcript，不重新提取，不改任何非 thinking 的步骤。只删
「同一轮里正文相同、时间挨在一起」的多余思考，占位符（``[REDACTED]``）和过短
文本一律保留——那些正文没有区分度，按文本判重会删掉真实步骤。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import click


class _Step:
    """存盘 cot.json 里的一个步骤。

    去重函数只读 ``step_type`` / ``content`` / ``metadata``，所以这里用一个
    直接包住原始 dict 的薄壳：改完之后原 dict 还在，写回时不必反序列化整个
    SessionCoT（那要求 extractor 的全部字段都对得上，对存量老数据太脆）。
    """

    __slots__ = ("raw", "step_type", "content", "metadata")

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.step_type = raw.get("step_type")
        self.content = raw.get("content") or ""
        # metadata 直接引用原 dict：去重函数往保留项里写
        # duplicate_generation_ids，这个副作用要落到写回的数据上
        md = raw.get("metadata")
        if not isinstance(md, dict):
            md = {}
            raw["metadata"] = md
        self.metadata = md


class _Turn:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.turn_index = raw.get("turn_index")
        self.steps = [_Step(s) for s in (raw.get("steps") or [])]
        self.total_steps = raw.get("total_steps")
        self.thinking_depth = raw.get("thinking_depth")

    def write_back(self) -> None:
        self.raw["steps"] = [s.raw for s in self.steps]
        self.raw["total_steps"] = self.total_steps
        self.raw["thinking_depth"] = self.thinking_depth


def _load_dedupers():
    """从打包进来的 extractor 里取去重函数。

    刻意不在这里复制一份判定逻辑：两处实现一旦漂移，「清理过的存量数据」和
    「新提取的数据」就会给出不同的步骤数。
    """
    from agent_cot import _assets

    root = _assets.bundled_extractor_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cot_extractor import (  # type: ignore
        _dedupe_double_written_thoughts,
        _dedupe_redundant_thinking_inter,
    )

    return _dedupe_double_written_thoughts, _dedupe_redundant_thinking_inter


def dedupe_cot_dict(cot: dict[str, Any]) -> dict[str, int]:
    """就地清理一个 cot.json 的重复 thinking，返回删除统计。"""
    dedupe_double, dedupe_inter = _load_dedupers()

    turns = [_Turn(t) for t in (cot.get("turns") or [])]
    if not turns:
        return {"double_written": 0, "inter_copies": 0, "steps_before": 0, "steps_after": 0}

    before = sum(len(t.steps) for t in turns)
    double_written = dedupe_double(turns)
    inter_copies = dedupe_inter(turns)
    after = sum(len(t.steps) for t in turns)

    if after != before:
        for turn in turns:
            turn.write_back()
        # step_index 重排成稠密序列：extractor 在去重之后也做同一件事，
        # 存量数据不跟上就会留下一串空号，看起来像丢了步骤。
        idx = 1
        for turn in turns:
            for step in turn.raw["steps"]:
                step["step_index"] = idx
                idx += 1
        # 会话级聚合是提取时冻结下来的，不重算的话前端 KPI 会跟树上的数量对不上
        cot["total_thinking_steps"] = sum(
            int(t.thinking_depth or 0) for t in turns
        )
        turn_count = len(turns)
        if turn_count:
            cot["avg_steps_per_turn"] = round(
                sum(int(t.total_steps or 0) for t in turns) / turn_count, 2
            )

    return {
        "double_written": double_written,
        "inter_copies": inter_copies,
        "steps_before": before,
        "steps_after": after,
    }


def _iter_cot_files(dirs: Iterable[Path], session_id: str | None) -> list[Path]:
    """收集待处理的 cot.json。

    ``session_id`` 允许只给前缀——会话 id 是 uuid，手打全长不现实。
    """
    pattern = f"{session_id}*_cot.json" if session_id else "*_cot.json"
    seen: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob(pattern)):
            seen.setdefault(str(path.resolve()), path)
        # sessions/<sid>/*_cot.json 这种按会话分目录的布局
        for sub in sorted(d.glob("*/*_cot.json")):
            if session_id and not sub.parent.name.startswith(session_id):
                continue
            seen.setdefault(str(sub.resolve()), sub)
    return list(seen.values())


def run_dedupe(
    *,
    session_id: str | None,
    cot_dir: str | None,
    apply: bool,
    quiet: bool = False,
) -> int:
    """扫描存盘的 cot.json 并清理重复 thinking。返回进程退出码。"""
    from agent_cot.commands.otlp_bridge import _candidate_cot_dirs

    # 目录解析复用 otlp_bridge 那套（env → runtime.json → 用户默认 → 源码树），
    # 免得这里再长出第三套「cot.json 在哪」的猜测逻辑。
    dirs = [Path(cot_dir).expanduser()] if cot_dir else _candidate_cot_dirs()
    files = _iter_cot_files(dirs, session_id)
    if not files:
        which = f"会话 {session_id!r} 的 " if session_id else ""
        looked = ", ".join(str(d) for d in dirs[:3]) or "(无候选目录)"
        click.secho(
            f"error: 没找到{which}cot.json。已查找：{looked}", fg="red", err=True,
        )
        return 1

    touched = 0
    removed_total = 0
    for path in files:
        try:
            cot = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            if not quiet:
                click.secho(f"  skip {path.name}: 读取失败 {exc}", fg="yellow")
            continue

        stats = dedupe_cot_dict(cot)
        removed = stats["steps_before"] - stats["steps_after"]
        if removed <= 0:
            continue
        touched += 1
        removed_total += removed
        if apply:
            # newline="" 保持与提取时一致的换行，免得整个文件在 diff 里全变红
            with open(path, "w", encoding="utf-8", newline="") as fh:
                json.dump(cot, fh, ensure_ascii=False, indent=2)
        if not quiet:
            click.echo(
                f"  {path.name}: -{removed} 步 "
                f"（hook 双写 {stats['double_written']}，"
                f"transcript 副本 {stats['inter_copies']}）"
            )

    if not quiet:
        mode = "已写回" if apply else "预演（未写回，加 --apply 才落盘）"
        click.secho(
            f"agent-cot dedupe-thinking — {mode}", bold=True,
            fg="green" if apply else "yellow",
        )
        click.echo(f"  扫描 : {len(files)} 个会话")
        click.echo(f"  命中 : {touched} 个会话，共 {removed_total} 个重复步骤")
        if touched and not apply:
            click.echo("  提示 : 重跑一次并加 --apply 即可清理")
    return 0
