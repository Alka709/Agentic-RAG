from mcp_client.client import invoke_mcp_tool


result = invoke_mcp_tool(
    "vector_search",
    {
        "query": "What is the document about?",
        "top_k": 4
    }
)

print(result)