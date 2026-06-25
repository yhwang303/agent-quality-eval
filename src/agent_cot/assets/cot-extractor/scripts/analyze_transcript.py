import json

path = r'C:\Users\milkwang\.claude-internal\projects\D--SST\c08e4c42-bc38-4fc9-b7e7-1a5cc4a1f10c.jsonl'
lines = open(path, encoding='utf-8').readlines()
msgs = [json.loads(l) for l in lines]

assistant_msgs = [m for m in msgs if m.get('message', {}).get('role') == 'assistant']
print(f"Total lines: {len(lines)}, assistant msgs: {len(assistant_msgs)}")

for i, m in enumerate(assistant_msgs):
    msg = m['message']
    stop = msg.get('stop_reason')
    content = msg.get('content', [])
    print(f"\n--- assistant[{i}] stop_reason={stop} ---")
    for b in content:
        if not isinstance(b, dict):
            continue
        btype = b.get('type')
        if btype == 'text':
            text = b.get('text', '')
            print(f"  TEXT ({len(text)} chars): {repr(text[:200])}")
        elif btype == 'tool_use':
            print(f"  TOOL_USE name={b.get('name')} id={b.get('id')}")
        elif btype == 'thinking':
            print(f"  THINKING ({len(b.get('thinking',''))} chars)")
        else:
            print(f"  OTHER type={btype}")
