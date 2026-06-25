#!/usr/bin/env python3
"""
端到端测试：把已有的 CoT JSON 上传到 Langfuse，验证 span 树是否正确
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, 'd:/ai-ide-langfuse/cot-extractor/src')
from cot_extractor import extract_session_cot
from cot_uploader import create_langfuse_client, upload_session_cot

# 用已有的 transcript 提取 CoT
session_id = '19ac5755-b272-49e3-ab65-1723ddc932fc'
transcript_path = Path.home() / '.claude-internal' / 'projects' / 'D--SST' / f'{session_id}.jsonl'

if not transcript_path.exists():
    print(f'ERROR: transcript not found at {transcript_path}')
    sys.exit(1)

session_cot, offset = extract_session_cot(transcript_path, session_id, offset=0)
if session_cot is None:
    print('ERROR: extract_session_cot returned None')
    sys.exit(1)

print(f'Loaded CoT: {len(session_cot.turns)} turns, {session_cot.total_tool_calls} tool calls')
for t in session_cot.turns:
    print(f'  Turn {t.turn_index}: {len(t.steps)} steps, query={t.user_query[:40] if t.user_query else "(tool result)"}')

client = create_langfuse_client()
if not client:
    print('ERROR: No Langfuse client')
    sys.exit(1)

ok = upload_session_cot(session_cot, session_id, client)
print(f'Upload result: {"SUCCESS" if ok else "FAILED"}')
