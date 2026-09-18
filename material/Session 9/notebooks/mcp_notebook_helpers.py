"""
Helpers for running MCP stdio examples inside Jupyter notebooks.

On Windows, notebook stderr is often an ipykernel OutStream object that does
not expose a real fileno(). The MCP client's Windows fallback subprocess path
tries to pass stderr through to subprocess.Popen(), which then crashes before
the MCP server starts.
"""

from __future__ import annotations

import subprocess
from typing import Any

from agents.mcp import MCPServerStdio
from mcp.client.stdio import stdio_client


class NotebookSafeMCPServerStdio(MCPServerStdio):
    """MCPServerStdio variant that is safe to launch from Windows notebooks."""

    def __init__(
        self,
        *args: Any,
        errlog: Any = subprocess.DEVNULL,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.errlog = errlog

    def create_streams(self):
        return stdio_client(self.params, errlog=self.errlog)
