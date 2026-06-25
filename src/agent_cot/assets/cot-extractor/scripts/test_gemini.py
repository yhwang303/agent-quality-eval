#!/usr/bin/env python3
"""测试智谱 GLM-4.7-Flash CoT 摘要生成"""
import os
import sys
import time

sys.path.insert(0, 'd:/ai-ide-langfuse/cot-extractor/src')

# 手动加载 .env
env_path = 'd:/ai-ide-langfuse/cot-extractor/.env'
with open(env_path, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, _, v = line.partition('=')
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

api_key = os.environ.get('ZHIPU_API_KEY', '')
print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")

MODEL = "glm-4.7-flash"

# 只测试 cot_summarizer 模块调用
print(f"\n=== 测试: cot_summarizer.generate_turn_cot_summary（{MODEL}）===")
try:
    from cot_summarizer import generate_turn_cot_summary

    class FakeStep:
        step_type = 'tool_decision'
        tool_name = 'Bash'
        duration_ms = 1200.0
        tokens = 0
        content = '调用工具 Bash: ls -la /project'
        metadata = {'tool_name': 'Bash', 'input_summary': 'ls -la /project'}

    class FakeStep2:
        step_type = 'tool_execution'
        tool_name = 'Bash'
        duration_ms = 350.0
        tokens = 0
        content = 'total 48\ndrwxr-xr-x  5 user user 4096 main.py\n-rw-r--r--  1 user user 1234 config.json'
        metadata = {'is_error': False}

    class FakeStep3:
        step_type = 'final_response'
        tool_name = None
        duration_ms = 500.0
        tokens = 120
        content = '当前项目目录包含 main.py 和 config.json 两个文件。'
        metadata = {}

    class FakeTurn:
        turn_index = 1
        user_query = '帮我查看当前项目目录下有哪些文件'
        steps = [FakeStep(), FakeStep2(), FakeStep3()]
        tool_calls = ['Bash']
        total_steps = 3

    result = generate_turn_cot_summary(FakeTurn(), api_key=api_key, model=MODEL)
    if result:
        print(f"✅ 摘要生成成功！长度: {len(result)} chars")
        print("─" * 50)
        print(result)
        print("─" * 50)
    else:
        print("❌ 返回 None，摘要生成失败")
except Exception as e:
    import traceback
    print(f"❌ 异常: {e}")
    traceback.print_exc()

print("\n=== 测试完成 ===")
