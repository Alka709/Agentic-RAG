import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("mcp_client")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = PROJECT_ROOT / "mcp_server" / "server.py"


def _direct_mcp_fallback(tool_name: str, arguments: dict):
    """Fallback to direct execution of MCP tool functions if stdio transport fails."""
    try:
        from mcp_server.server import vector_search, web_search

        if tool_name == "vector_search":
            return vector_search(**arguments)
        elif tool_name == "web_search":
            return web_search(**arguments)
        else:
            raise ValueError(f"Unknown MCP tool: {tool_name}")
    except Exception as e:
        logger.error(f"Direct MCP tool fallback also failed: {e}", exc_info=True)
        raise


async def call_mcp_tool(tool_name: str, arguments: dict):
    env_vars = dict(os.environ)
    if "PYTHONPATH" in env_vars:
        env_vars["PYTHONPATH"] = f"{str(PROJECT_ROOT)}{os.pathsep}{env_vars['PYTHONPATH']}"
    else:
        env_vars["PYTHONPATH"] = str(PROJECT_ROOT)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env=env_vars
    )

    err_capture = io.StringIO()

    try:
        async with stdio_client(server_params, errlog=err_capture) as (
            read_stream,
            write_stream
        ):
            async with ClientSession(
                read_stream,
                write_stream
            ) as session:

                await session.initialize()

                result = await session.call_tool(
                    tool_name,
                    arguments
                )

                if result.isError:
                    raise RuntimeError(
                        f"MCP tool '{tool_name}' failed: {result.content}"
                    )

                parsed_results = []

                for content_block in result.content:
                    text = content_block.text
                    parsed = json.loads(text)

                    # Handle double-encoded JSON
                    if isinstance(parsed, str):
                        parsed = json.loads(parsed)

                    if isinstance(parsed, list):
                        parsed_results.extend(parsed)
                    else:
                        parsed_results.append(parsed)

                return parsed_results

    except Exception as e:
        stderr_output = err_capture.getvalue().strip()
        logger.warning(
            f"MCP stdio transport encountered an issue ({e}). Stderr: {stderr_output}. "
            "Falling back to direct MCP tool execution."
        )
        return _direct_mcp_fallback(tool_name, arguments)


def invoke_mcp_tool(tool_name: str, arguments: dict):
    """
    Synchronous wrapper so it can be used inside LangGraph nodes
    whether or not a running event loop is present (e.g. FastAPI/uvicorn).
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, call_mcp_tool(tool_name, arguments))
        try:
            return future.result()
        except Exception as e:
            if hasattr(e, "exceptions") and e.exceptions:
                sub_msgs = [str(sub) for sub in e.exceptions]
                raise RuntimeError("; ".join(sub_msgs)) from e
            raise