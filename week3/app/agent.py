import json
from typing import Callable, Dict

from openai import OpenAI
from tavily import TavilyClient

from .rag_chain import search_local_knowledge


def search_web(query: str) -> str:
    tavily_client = TavilyClient()
    response = tavily_client.search(
        query=query,
        topic="general",
        search_depth="advanced",
        max_results=5,
        include_answer=False,
        include_raw_content=False,
    )

    results = response.get("results", [])
    if not results:
        return "No relevant web results found."

    blocks = []
    for idx, item in enumerate(results, 1):
        blocks.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"title: {item.get('title', '')}",
                    f"url: {item.get('url', '')}",
                    f"content: {item.get('content', '')}",
                    f"score: {item.get('score', '')}",
                ]
            )
        )

    return "\n\n".join(blocks)


def get_tools():
    return [
        {
            "type": "function",
            "name": "search_local_knowledge",
            "description": (
                "Search the local machine learning knowledge base and return "
                "relevant evidence snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A machine learning question to search in the local "
                            "knowledge base."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "search_web",
            "description": "Search the web for recent or external information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A query to search on the web.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def run_agent(
    user_query: str,
    *,
    model: str = "gpt-5.5",
    tool_response_model: str | None = None,
    instructions: str | None = None,
) -> str:
    client = OpenAI()
    tools = get_tools()
    tool_handlers: Dict[str, Callable[[str], str]] = {
        "search_local_knowledge": search_local_knowledge,
        "search_web": search_web,
    }

    input_items = [{"role": "user", "content": user_query}]
    response = client.responses.create(
        model=model,
        tools=tools,
        input=input_items,
        instructions=instructions,
    )

    while True:
        input_items += response.output
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            return response.output_text

        for item in function_calls:
            query = json.loads(item.arguments)["query"]
            output = tool_handlers[item.name](query)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output,
                }
            )

        response = client.responses.create(
            model=tool_response_model or model,
            tools=tools,
            input=input_items,
            instructions=instructions,
        )
