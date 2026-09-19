import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SERVER_PATH = PROJECT_ROOT / "mcp_server" / "server.py"


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

    async with stdio_client(server_params) as (
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

def invoke_mcp_tool(tool_name: str, arguments: dict):
    """
    Synchronous wrapper so it can be used inside LangGraph nodes
    whether or not a running event loop is present (e.g. FastAPI/uvicorn).

    asyncio.run() cannot be called from inside a running event loop, so
    we spin up a fresh loop on a dedicated background thread instead.
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, call_mcp_tool(tool_name, arguments))
        try:
            return future.result()
        except Exception as e:
            # Unwrap ExceptionGroup if present
            if hasattr(e, "exceptions") and e.exceptions:
                sub_msgs = [str(sub) for sub in e.exceptions]
                raise RuntimeError("; ".join(sub_msgs)) from e
            raise