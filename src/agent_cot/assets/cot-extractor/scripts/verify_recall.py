"""一次性验证脚本：v0.8.1 修复后，重新提取最新 session 的 CoT，
检查 RAG step 的 recall_preview 是否：
  1. 不再是 synthetic 占位 "(工具执行结果未记录..."
  2. 当真有 cot-stream.js 的 result_text 时被正确提取
  3. 缺失时给出 recall_unavailable_reason

跑法：
  cd d:\\ai-ide-langfuse\\cot-extractor
  python scripts\\verify_recall.py
"""

from pathlib import Path
import os
import sys
import json

SESSION_ID = "a61b75a0-ca00-492f-a590-21f6d487280f"

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# 默认 cot-extractor root（events.jsonl 路径）
os.environ.setdefault("COT_EXTRACTOR_ROOT", str(ROOT))

from cot_extractor import extract_session_cot  # noqa: E402

TRANSCRIPT = (
    Path(os.environ.get("USERPROFILE", os.path.expanduser("~")))
    / ".cursor" / "projects" / "d-ai-ide-langfuse"
    / "agent-transcripts" / SESSION_ID
    / f"{SESSION_ID}.jsonl"
)

print(f"transcript: {TRANSCRIPT}  exists={TRANSCRIPT.exists()}  size={TRANSCRIPT.stat().st_size if TRANSCRIPT.exists() else 0}")

cot, _ = extract_session_cot(TRANSCRIPT, SESSION_ID, offset=0)
if cot is None:
    print("[!] extract_session_cot returned None")
    sys.exit(1)

print(f"turns: {len(cot.turns)}  invocation_stats: {cot.invocation_stats}")
print(f"observed_events: {cot.observed_events}")

placeholder_hits = 0
real_recall_hits = 0
unavailable_with_reason = 0
no_recall_no_reason = 0

def _kind(s):
    k = getattr(s, "step_type", None)
    return getattr(k, "value", k)

for t in cot.turns:
    for s in t.steps:
        if _kind(s) != "tool_execution":
            continue
        cat = (s.metadata or {}).get("invocation_category")
        if cat not in ("rag_query", "web_search"):
            continue
        recall = (s.metadata or {}).get("recall_preview") or ""
        reason = (s.metadata or {}).get("recall_unavailable_reason") or ""
        if recall.startswith("(工具执行结果未记录") or recall.startswith("（工具执行结果未记录"):
            placeholder_hits += 1
        elif recall:
            real_recall_hits += 1
        elif reason:
            unavailable_with_reason += 1
        else:
            no_recall_no_reason += 1

print("─── verification result ───")
print(f"  placeholder_hits     = {placeholder_hits}  (应为 0，>0 说明 fix 失败)")
print(f"  real_recall_hits     = {real_recall_hits}  (大于 0 说明拿到了真实 RAG/Web 召回)")
print(f"  unavailable_with_reason = {unavailable_with_reason}  (没拿到但有解释)")
print(f"  no_recall_no_reason  = {no_recall_no_reason}  (沉默缺失，应为 0)")

# 把新生成的 CoT 持久化到 cot/ 目录，让 dashboard 立即看到效果
out_dir = ROOT / "output" / "cot"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{SESSION_ID}_cot.json"
out_path.write_text(json.dumps(cot.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[saved] {out_path} ({out_path.stat().st_size:,} bytes)")

# 输出前 3 条 RAG/Web 步骤示例，验证 prompt_preview / decision_tool_input / recall 三件套
shown = 0
print()
print("─── sample RAG/Web execution steps ───")
for t in cot.turns:
    for s in t.steps:
        if shown >= 3:
            break
        if _kind(s) != "tool_execution":
            continue
        cat = (s.metadata or {}).get("invocation_category")
        if cat not in ("rag_query", "web_search"):
            continue
        md = s.metadata or {}
        print(f"--- turn{t.turn_index} step#{s.step_index} ({cat}) ---")
        print(f"  tool_name             = {md.get('tool_name')}")
        print(f"  prompt_preview        = {(md.get('prompt_preview') or '')[:80]!r}")
        print(f"  decision_tool_input   = {json.dumps(md.get('decision_tool_input'))[:120] if md.get('decision_tool_input') else None}")
        print(f"  recall_preview (chars)= {len(md.get('recall_preview') or '')}")
        print(f"  recall_unavailable_reason = {(md.get('recall_unavailable_reason') or '')[:120]!r}")
        print(f"  observed_output keys  = {list((md.get('observed_output') or {}).keys())}")
        print(f"  synthetic / upgraded  = {md.get('synthetic')} / {md.get('synthetic_upgraded')}")
        shown += 1
    if shown >= 3:
        break
