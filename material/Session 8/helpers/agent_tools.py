"""Tools and a bounded Responses API loop, following notebook 06."""
from __future__ import annotations

import json
from datetime import datetime
from urllib.parse import quote

import requests


def tool_schema(name, description, properties):
    return {
        "type": "function", "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False},
        "strict": True,
    }


TOOLS = [
    tool_schema(
        "search_documents",
        "Search the company handbook and benefits PDFs for evidence. Use when company "
        "policy facts are needed. Returns passages with source filenames and chunk IDs.",
        {"query": {"type": "string", "description": "A focused search question."}},
    ),
    tool_schema(
        "get_weather", "Get current weather for a city. This does not provide forecasts.",
        {"location": {"type": "string", "description": "City and country, e.g. Sydney, Australia."},
         "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    ),
    tool_schema(
        "get_current_datetime",
        "Get the current date, time, weekday and UTC offset on the computer running the app. "
        "This is the server's local timezone, not necessarily the user's timezone.",
        {},
    ),
]


def get_weather(location: str, units: str = "celsius") -> dict:
    """Use the same no-key wttr.in service as notebook 06."""
    if not isinstance(location, str) or not location.strip() or len(location) > 200:
        raise ValueError("Provide a city and country (up to 200 characters).")
    if units not in ("celsius", "fahrenheit"):
        raise ValueError("Units must be celsius or fahrenheit.")
    try:
        response = requests.get(
            f"https://wttr.in/{quote(location.strip(), safe='')}",
            params={"format": "j1"}, timeout=10,
        )
        response.raise_for_status()
        current = response.json()["current_condition"][0]
        return {
            "location": location, "units": units,
            "temperature": int(current["temp_C" if units == "celsius" else "temp_F"]),
            "description": current["weatherDesc"][0]["value"],
            "observed_at": current.get("localObsDateTime", "Not supplied"),
            "source": "wttr.in",
        }
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return {"error": "Weather service unavailable or returned an unexpected response. "
                         "Do not invent weather values.", "location": location}


def get_current_datetime() -> dict:
    now = datetime.now().astimezone()
    return {"datetime": now.isoformat(timespec="seconds"), "date": now.date().isoformat(),
            "day_of_week": now.strftime("%A"), "timezone": now.tzname(),
            "clock": "App server's local clock"}


def execute_tool(call, registry: dict) -> dict:
    """Dispatch only registered tools; return recoverable failures to the model."""
    if call.name not in registry:
        return {"error": f"Unknown tool: {call.name}"}
    try:
        arguments = json.loads(call.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        schema = next(tool["parameters"] for tool in TOOLS if tool["name"] == call.name)
        if set(arguments) != set(schema["required"]):
            raise ValueError("Tool arguments do not match the required fields.")
        if any(not isinstance(value, str) for value in arguments.values()):
            raise ValueError("Tool arguments must be strings.")
        return registry[call.name](**arguments)
    except (ValueError, TypeError):
        return {"error": "Invalid tool arguments. Use the tool's schema and try again."}
    except Exception:
        # Avoid sending credentials or backend exception details to the model/UI.
        return {"error": "Tool execution failed. Do not invent a successful result."}


def stream_tool_loop(client, model: str, messages: list, instructions: str,
                     registry: dict, max_rounds: int = 4):
    """Yield text/tool/done events; never retrieve before the model requests it.

    Preserve every response output item (including reasoning) and match each tool
    result to its call_id. After the budget, request a final answer with tools off.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    conversation = list(messages)
    for round_number in range(max_rounds + 1):
        final_round = round_number == max_rounds
        prompt = instructions
        if final_round:
            prompt += "\nTool budget reached. Answer from available results; disclose any missing evidence."
            yield {"type": "limit", "text": "Tool-round limit reached; requesting a final answer."}
        response = None
        with client.responses.create(
            model=model, instructions=prompt, input=conversation, tools=TOOLS,
            tool_choice="none" if final_round else "auto",
            parallel_tool_calls=False, stream=True,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield {"type": "text", "text": event.delta}
                elif event.type == "response.refusal.delta":
                    yield {"type": "text", "text": event.delta}
                elif event.type == "response.completed":
                    response = event.response
                elif event.type in ("response.failed", "response.incomplete", "error"):
                    raise RuntimeError("The model response did not complete. Please retry.")
        if response is None:
            raise RuntimeError("The model stream ended before completion. Please retry.")
        conversation.extend(response.output)
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            yield {"type": "done", "messages": conversation}
            return
        if final_round:
            raise RuntimeError("The model requested a tool after the tool budget was exhausted.")
        for call in calls:
            yield {"type": "tool_start", "name": call.name}
            result = execute_tool(call, registry)
            conversation.append({"type": "function_call_output", "call_id": call.call_id,
                                 "output": json.dumps(result, ensure_ascii=False)})
            yield {"type": "tool_result", "name": call.name,
                   "arguments": call.arguments, "result": result}
        yield {"type": "text", "text": "\n\n"}
