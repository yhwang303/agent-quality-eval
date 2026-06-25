"""mcp.json swap/restore — 把 Cursor 的 MCP server url 切到本地代理 / 还原回原 url。

为什么独立成一个工具：
  ~/.cursor/mcp.json 是 Cursor 重启时读的关键配置，直接手改容易忘备份。
  这里做"备份 → 替换 → 还原"三件套，并且校验代理是否正在监听，避免切过去
  之后 Cursor MCP 全炸。

用法：
  python mcp_swap_config.py status
  python mcp_swap_config.py enable               # 切到代理（自动备份）
  python mcp_swap_config.py enable --servers iWiki shadow-folk
  python mcp_swap_config.py disable              # 还原到 .bak

约定：
  ~/.cursor/mcp.json                  当前生效的配置
  ~/.cursor/mcp.json.original         首次 enable 时的原始备份（永远不动）
  ~/.cursor/mcp.json.last-disabled    上次 disable 时的状态（再次 disable 用）

设计原则：
  - 只改我们认识的 server（UPSTREAM 字典里的），其它 server 一字不动
  - enable 前必须 health-check 代理可达，否则拒绝（防止 Cursor MCP 全断）
  - 备份用原子重命名 + JSON pretty print，方便人工 diff
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

# Windows 控制台默认 GBK，print(✓) 会炸；统一强制 UTF-8 输出。
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 必须和 mcp_traffic_proxy.js 里的 UPSTREAM keys 保持同步
# value 是只用于回滚检测的"原 url 子串"，万一用户的 mcp.json 之前改过，我们也能识别
KNOWN_SERVERS = {
    "iWiki": ["mcp.it.woa.com", "iwiki", "app_iwiki_mcp"],
    "gongfengStreamable": ["git.woa.com/api/mcp"],
    "shadow-folk": ["9.134.128.138", "/api/mcp"],
}

DEFAULT_PROXY = "http://127.0.0.1:8766"

MCP_PATH = Path(os.path.expanduser("~/.cursor/mcp.json"))
BAK_ORIGINAL = MCP_PATH.with_suffix(MCP_PATH.suffix + ".original")
BAK_LAST = MCP_PATH.with_suffix(MCP_PATH.suffix + ".last-disabled")


def _read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(p: Path, data: Dict[str, Any]) -> None:
    """Atomic write: dump → temp → rename。
    避免半写中状态被 Cursor 读到导致 JSON parse 错误。"""
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, p)


def _is_proxy_url(url: str, proxy_base: str) -> bool:
    return url.strip().lower().startswith(proxy_base.lower())


def _health_check(proxy_base: str) -> Optional[Dict[str, Any]]:
    try:
        req = urllib.request.Request(f"{proxy_base}/_health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"  ✗ health check failed: {e}", file=sys.stderr)
        return None


def cmd_status(args: argparse.Namespace) -> int:
    if not MCP_PATH.exists():
        print(f"mcp.json not found: {MCP_PATH}")
        return 2
    data = _read_json(MCP_PATH)
    servers = data.get("mcpServers", {})
    proxy_base = args.proxy

    print(f"mcp.json: {MCP_PATH}")
    print(f"original backup: {'YES' if BAK_ORIGINAL.exists() else 'NO'} ({BAK_ORIGINAL})")
    print(f"proxy target: {proxy_base}")
    print()
    print(f"{'server':<25} {'state':<10} {'url'}")
    print("-" * 90)
    for name, conf in servers.items():
        url = conf.get("url", "")
        if _is_proxy_url(url, proxy_base):
            state = "PROXIED"
        elif name in KNOWN_SERVERS:
            state = "DIRECT"
        else:
            state = "UNKNOWN"
        print(f"{name:<25} {state:<10} {url}")

    print()
    print("--- proxy health ---")
    h = _health_check(proxy_base)
    if h:
        print(f"  ✓ proxy alive pid={h.get('pid')} exposes {h.get('servers')}")
    return 0


def cmd_enable(args: argparse.Namespace) -> int:
    if not MCP_PATH.exists():
        print(f"mcp.json not found: {MCP_PATH}", file=sys.stderr)
        return 2
    proxy_base = args.proxy

    # 1. 健康检查 —— 代理必须先起来，否则切过去 Cursor 立刻全炸
    h = _health_check(proxy_base)
    if not h:
        print(f"refuse to swap: proxy {proxy_base} not responding", file=sys.stderr)
        print("  start it first with: cot-extractor\\scripts\\mcp_proxy.cmd /b")
        return 3
    proxy_servers = set(h.get("servers", []))
    print(f"✓ proxy alive: pid={h.get('pid')} exposes {sorted(proxy_servers)}")

    # 2. 备份原始（只在第一次）
    data = _read_json(MCP_PATH)
    if not BAK_ORIGINAL.exists():
        shutil.copy2(MCP_PATH, BAK_ORIGINAL)
        print(f"✓ saved original backup: {BAK_ORIGINAL}")
    else:
        print(f"… original backup already exists: {BAK_ORIGINAL}")

    # 3. 决定要换哪些 server
    target_servers = args.servers or list(KNOWN_SERVERS.keys())
    swapped = []
    skipped = []
    for name in target_servers:
        if name not in data.get("mcpServers", {}):
            skipped.append((name, "missing in mcp.json"))
            continue
        if name not in proxy_servers:
            skipped.append((name, "not exposed by proxy"))
            continue
        conf = data["mcpServers"][name]
        new_url = f"{proxy_base}/{name}"
        if conf.get("url") == new_url:
            skipped.append((name, "already proxied"))
            continue
        # 注意：把 headers 删掉，因为代理会自动注入 Authorization
        # 留 timeout 这种和凭证无关的字段
        if "headers" in conf:
            del conf["headers"]
        conf["url"] = new_url
        swapped.append(name)

    if not swapped:
        print("nothing to swap.")
        for n, why in skipped:
            print(f"  - {n}: {why}")
        return 0

    # 4. 原子写入
    _write_json(MCP_PATH, data)
    print(f"✓ swapped {len(swapped)} server(s) → proxy:")
    for n in swapped:
        print(f"    {n} → {proxy_base}/{n}")
    if skipped:
        print("  skipped:")
        for n, why in skipped:
            print(f"    {n}: {why}")

    print()
    print("⚠️  Cursor 需要重新连接 MCP server 才会生效。")
    print("   两种触发方式：")
    print("     A. 重启 Cursor")
    print("     B. 在 Cursor Settings → MCP 里点'Refresh' / 关再开对应 server")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    if not BAK_ORIGINAL.exists():
        print(f"no original backup at {BAK_ORIGINAL}; refuse to restore", file=sys.stderr)
        return 4
    # 把当前文件存到 last-disabled 方便排查
    if MCP_PATH.exists():
        shutil.copy2(MCP_PATH, BAK_LAST)
        print(f"saved current state → {BAK_LAST}")
    shutil.copy2(BAK_ORIGINAL, MCP_PATH)
    print(f"✓ restored {MCP_PATH} from {BAK_ORIGINAL}")
    print()
    print("⚠️  Cursor 需要重新连接 MCP server 才会生效（重启或 Settings → MCP Refresh）")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="show current mcp.json state")
    s.add_argument("--proxy", default=DEFAULT_PROXY)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("enable", help="swap mcp.json to point at the local proxy")
    s.add_argument("--proxy", default=DEFAULT_PROXY)
    s.add_argument("--servers", nargs="+", help="which servers to swap (default: all known)")
    s.set_defaults(func=cmd_enable)

    s = sub.add_parser("disable", help="restore mcp.json from the original backup")
    s.set_defaults(func=cmd_disable)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
