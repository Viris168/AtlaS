"""agent/tools/web.py — Server-side web search tool schema (Milestone 2).

Web search runs on the provider's servers, not on our machine. We only send this
schema with the request; the provider does the search and puts the results straight
into the model's reply. So there is no Python function and no ToolRegistry entry.
"""


def web_search_schema(max_uses: int = 5) -> dict:
    return {
        "type": "web_search_20250305",  # tells the provider which built-in tool this is
        "name": "web_search",
        "max_uses": max_uses,  # cap on searches per request, to limit cost
    }
