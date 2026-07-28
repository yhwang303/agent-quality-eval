"""会话级 trace 导出。

把观测到的一个 agent 会话拍平成一条按执行顺序排列的事件流，再渲染成
jsonl / json / md。给闭环 harness 用：agent 读上一轮执行的完整 trace
（thinking / tool use / plan / permission / subagent / …）来判断下一步
往哪个方向优化 harness。

与 ``cot_otlp_exporter`` 的分工：那个走 OTel 的 span 模型，跨系统兼容性好，
但 span 装不下 thinking 正文、plan 快照、权限请求这些 agent 特有的东西。
本模块导出的是本项目自己捕捉到的**全部**事件，不做取舍。

这两个模块刻意放在 ``agent_cot`` 包里而不是 ``assets/cot-extractor/`` 下：
assets 里那两份是上游 cursor-cot-observer 的 vendor 镜像，会被资产同步流程
整体覆盖；trace 导出是本项目自己的功能，放在正规包里才能被正常 import，
也不会在下次同步时被抹掉。
"""

from agent_cot.trace.exporter import (
    SUPPORTED_FORMATS,
    export_session_trace,
    export_turn_trace,
    render_json,
    render_jsonl,
    render_markdown,
    sanitize_session_id,
)
from agent_cot.trace.flattener import TRACE_SCHEMA, flatten_session

__all__ = [
    "SUPPORTED_FORMATS",
    "TRACE_SCHEMA",
    "export_session_trace",
    "export_turn_trace",
    "flatten_session",
    "render_json",
    "render_jsonl",
    "render_markdown",
    "sanitize_session_id",
]
