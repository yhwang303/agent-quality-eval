# -*- mode: python ; coding: utf-8 -*-
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

version_text = Path('src/agent_quality_eval/__init__.py').read_text(encoding='utf-8')
version = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', version_text).group(1)

hiddenimports = ['psutil', 'requests', 'dotenv', 'click', 'colorama', 'PIL', 'PIL.Image', 'PIL.ImageDraw']
hiddenimports += collect_submodules('pystray')
hiddenimports += collect_submodules('agent_cot')
hiddenimports += collect_submodules('agent_quality_eval')
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('starlette')
hiddenimports += collect_submodules('pydantic')
hiddenimports += collect_submodules('pydantic_core')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('watchfiles')
hiddenimports += collect_submodules('opentelemetry')


a = Analysis(
    ['scripts\\observation_agent_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src\\agent_cot\\assets', 'agent_cot\\assets'),
        ('src\\agent_quality_eval\\assets', 'agent_quality_eval\\assets'),
        ('src\\agent_quality_eval\\templates', 'agent_quality_eval\\templates'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=f'agent-quality-eval-{version}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets\\agent-quality-eval.ico',
)
