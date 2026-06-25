"""
Agent Dashboard 配置文件
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# 项目根目录（ai-ide-langfuse / 源码态）
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _looks_bundled() -> bool:
    """启发式判断本文件是不是被 wheel 当作 ``agent_cot`` 资产打包了。

    在 wheel 安装态下 ``__file__`` 形如
    ``<site-packages>/agent_cot/assets/backend/config.py``；同样地，
    sync 完 assets 后开发态的源文件也会落到
    ``agent-cot/src/agent_cot/assets/backend/config.py`` —— 两种都
    意味着 ``_PROJECT_ROOT`` 退化成了 wheel 内部，把数据目录指过去
    既写不进去（pip 安装目录），后端自己扫该路径也只会是空。

    判定就看一件事：``__file__`` 路径里是否同时含 ``agent_cot`` 和
    ``assets/backend``。两条都中即认为我们在 vendor 后的副本里跑，
    应当走用户级 ``~/.agent-cot/data/`` 而不是源仓库路径。
    """
    s = str(Path(__file__).resolve()).replace("\\", "/").lower()
    return "/agent_cot/assets/backend/" in s


def _user_data_root() -> Path:
    """v0.18.2: 跨进程的"单一真相"数据根。

    - 客户端 hook (cot-bridge.js → extract_cot.py) 写这里
    - backend (本文件 → session_scanner) 扫这里
    - agent-cot CLI 起 backend 时也通过 ``AGENT_COT_DATA_ROOT`` 注入这里

    不依赖代码安装位置 —— 装 wheel / git clone / editable install
    全都对齐到 ``~/.agent-cot/data/``，这样 dashboard 永远能看到
    最新的 cot.json，无论用户走哪种安装路径。
    """
    env = os.environ.get("AGENT_COT_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".agent-cot" / "data"


def _default_data_dir(subdir: str, legacy_subdir: str) -> Path:
    """优先 wheel 安全的 ~/.agent-cot/data/<subdir>；源码态保留老路径。"""
    if _looks_bundled():
        return _user_data_root() / subdir
    return _PROJECT_ROOT / legacy_subdir


# 数据目录配置（支持环境变量覆盖；wheel 安装态自动跳到用户级目录）
COT_DIR = Path(os.environ.get("COT_DIR") or _default_data_dir("cot", "cot-extractor/output/cot"))
COT_REPORTS_DIR = Path(os.environ.get("COT_REPORTS_DIR") or _default_data_dir("reports", "cot-extractor/output/reports"))
RESPONSE_REPORTS_DIR = Path(os.environ.get("RESPONSE_REPORTS_DIR") or _default_data_dir("response-reports", "response-verifier/output/reports"))
TRANSCRIPTS_DIR = Path(os.environ.get("TRANSCRIPTS_DIR") or _default_data_dir("transcripts", "response-verifier/output/transcripts"))
LANGFUSE_CACHE_DIR = Path(os.environ.get("LANGFUSE_CACHE_DIR") or _default_data_dir("langfuse-cache", "response-verifier/output/langfuse_cache"))


def _cot_scan_dirs() -> list[Path]:
    dirs = [COT_DIR]
    raw = os.environ.get("COT_SCAN_DIRS") or os.environ.get("AGENT_COT_COT_SCAN_DIRS")
    if raw:
        for item in raw.split(os.pathsep):
            item = item.strip()
            if item:
                dirs.append(Path(item).expanduser())
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


COT_SCAN_DIRS = _cot_scan_dirs()

# AGENT_COT_PIPELINE_LOG_INJECT_v1
def _read_runtime_state():
    try:
        p = Path.home() / '.agent-cot' / 'runtime.json'
        if p.is_file():
            data = json.loads(p.read_text(encoding='utf-8'))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _normalize_data_root(p: Path) -> Path:
    name = p.name
    if name == 'data':
        return p
    if name in ('.agent-cot', '.agent-cot'):
        return p / 'data'
    return p

def _legacy_scan_roots():
    primary = _user_data_root().resolve()
    home = Path.home()
    cands = [home / '.agent-cot', home / '.agent-cot' / 'data',
             home / '.agent-cot', home / '.agent-cot' / 'data']
    extra = os.environ.get('AGENT_COT_EXTRA_SCAN_ROOTS', '').strip()
    if extra:
        for chunk in extra.replace(';', os.pathsep).split(os.pathsep):
            c = chunk.strip()
            if c:
                cands.append(Path(c).expanduser())
    out, seen = [], set()
    for c in cands:
        try:
            r = c.resolve()
        except Exception:
            continue
        if r == primary or str(r) in seen:
            continue
        seen.add(str(r))
        if r.is_dir():
            out.append(r)
    return out

COT_SCAN_DIRS = [COT_DIR]
for _legacy_root in _legacy_scan_roots():
    _legacy_cot = _legacy_root / 'cot'
    if _legacy_cot.is_dir() and _legacy_cot.resolve() != COT_DIR.resolve():
        COT_SCAN_DIRS.append(_legacy_cot)

def _config_pipeline_breadcrumb():
    try:
        lp = Path(os.environ.get('AGENT_COT_PIPELINE_LOG')
                  or str(Path.home() / '.agent-cot' / 'logs' / 'pipeline.log')
                  ).expanduser()
        lp.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        sd = ';'.join(str(p) for p in COT_SCAN_DIRS)
        with open(lp, 'a', encoding='utf-8', errors='replace') as f:
            f.write(f'[{ts}] [backend.config] [-] [sid=-] event=boot status=ok '
                    f'cot_dir={COT_DIR} scan_dirs="{sd}"\n')
    except Exception:
        pass

_config_pipeline_breadcrumb()

# v0.18.2: 启动时把数据目录 mkdir 出来，避免后续 ``COT_DIR.glob(...)`` 因为
# 路径不存在而走静默"零结果"路径（之前最常见的体验是 backend 跑得好好的、
# 接口 200 OK，但 SESSIONS 0，让人误以为是前端 / hook 出了 bug）。
for _d in (COT_DIR, *COT_SCAN_DIRS, COT_REPORTS_DIR, TRANSCRIPTS_DIR, LANGFUSE_CACHE_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# 服务配置
HOST = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))
