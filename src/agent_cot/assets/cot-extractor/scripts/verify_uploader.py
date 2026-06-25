#!/usr/bin/env python3
import sys
import json
from pathlib import Path

sys.path.insert(0, 'd:/ai-ide-langfuse/cot-extractor/src')
from cot_uploader import get_trace_id, get_turn_span_id, get_turn_count, create_langfuse_client

state_file = Path.home() / '.claude' / 'state' / 'langfuse_state.json'
state = json.loads(state_file.read_text(encoding='utf-8'))
print('State keys:', list(state.keys()))

session_id = '19ac5755-b272-49e3-ab65-1723ddc932fc'
trace_id = get_trace_id(session_id)
turn_span = get_turn_span_id(session_id)
turn_count = get_turn_count(session_id)
print(f'session: {session_id[:16]}')
print(f'trace_id: {trace_id}')
print(f'current_turn_span_id: {turn_span}')
print(f'turn_count: {turn_count}')

client = create_langfuse_client()
if client:
    print('Langfuse client: OK - keys found, upload will work')
else:
    print('Langfuse client: FAILED - no keys configured')
