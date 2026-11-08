"""Anthropic tool-loop instrumentation; requires credentials and the anthropic extra.

Run manually with ANTHROPIC_API_KEY and ANTHROPIC_MODEL set. This is not a test.
"""

import os

from anthropic import Anthropic

import promptkeep


def main() -> None:
    """Track a real tool-use round trip, with explicit tool/model parenting."""
    client = promptkeep.wrap(Anthropic())
    messages = [{"role": "user", "content": "What is the status of order 123?"}]
    tools = [
        {
            "name": "order_status",
            "description": "Get an order's current status",
            "input_schema": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    ]
    with client, promptkeep.agent("order-assistant") as task:
        response = client.messages.create(
            model=os.environ["ANTHROPIC_MODEL"], max_tokens=500, messages=messages, tools=tools
        )
        messages.append(
            {"role": "assistant", "content": [b.model_dump() for b in response.content]}
        )
        results = []
        for block in response.content:
            if block.type == "tool_use":
                with promptkeep.tool(block.name, input=block.input, call_id=block.id) as tool:
                    result = "shipped"  # Replace with the application's actual lookup.
                    tool.output(result)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        if results:
            messages.append({"role": "user", "content": results})
            response = client.messages.create(
                model=os.environ["ANTHROPIC_MODEL"], max_tokens=500, messages=messages, tools=tools
            )
        task.output("".join(b.text for b in response.content if b.type == "text"))
    promptkeep.flush()
    print(task.trace_key)


if __name__ == "__main__":
    main()
