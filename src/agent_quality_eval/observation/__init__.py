"""Compatibility layer for the copied observation-agent runtime.

The original observation package is preserved as top-level ``agent_cot`` so
existing hooks and subprocess imports keep working. New product surfaces should
import through this module when they need to delegate to observation commands.
"""

from __future__ import annotations

from agent_cot import __version__ as observation_version

__all__ = ["observation_version"]
