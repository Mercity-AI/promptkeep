"""An offline walkthrough of agent traces: one task with a successful model
call, one with caught tool and provider failures, and one escaped application
failure.

No network and no API key: an ``httpx.MockTransport`` answers the real
``openai`` SDK, so the wrapper and adapters run exactly as they do live.

    uv run python examples/agent_trace_demo.py

Requires the dev dependencies (``openai`` and ``httpx``). Writes to a
throwaway database.
"""

import tempfile
from pathlib import Path

import httpx
import openai

import promptkeep
from promptkeep import Prompt, history, wrap

PROMPT = Prompt("Answer in at most {n} words.", {"n": 4}, name="TRACE_DEMO")
MESSAGES = [
    {"role": "system", "content": PROMPT},
    {"role": "user", "content": "Capital of Peru?"},
]
CALLS = 0


def answer(request: httpx.Request) -> httpx.Response:
    """A tiny mock endpoint: one success, then one rate-limit error."""
    global CALLS
    CALLS += 1
    if CALLS > 1:
        return httpx.Response(
            429,
            json={"error": {"message": "mock rate limit", "type": "rate_limit_error"}},
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 1,
            "model": "demo-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Lima"},
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 2,
                "total_tokens": 13,
                "cost": 0.00002,
            },
        },
    )


def show(title: str, trace) -> None:
    """Print the parts this example exists to make visible."""
    print(f"\n{title}")
    print(f"  root:        {trace.root.name!r} -> {trace.status}")
    print(f"  spans:       {[(span.kind, span.name, span.status) for span in trace.spans]}")
    print(f"  child errors:{trace.error_count}")
    print(f"  model calls: {trace.model_calls}")
    print(f"  tokens/cost: {trace.total_tokens} / {history.format_cost(trace.total_cost)}")
    print(f"  cost known:  {trace.cost_complete}")
    print(f"  versions:    {trace.versions_used}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        promptkeep.configure(db_path=Path(tmp) / "traces.db", write_mode="sync")
        transport = httpx.MockTransport(answer)
        client = wrap(
            openai.OpenAI(
                api_key="sk-test",
                base_url="http://mock/v1",
                http_client=httpx.Client(transport=transport),
            )
        )

        # Success path: an agent task, a local tool, and a wrapped model call.
        with promptkeep.agent("research", input={"question": "Capital of Peru?"}) as task:
            with promptkeep.tool("local_search", input={"query": "Peru capital"}) as search:
                search.output({"hits": ["Lima is the capital"]})

            response = client.chat.completions.create(model="demo-model", messages=MESSAGES)
            task.output(response.choices[0].message.content)

        show("1. successful task (root ok, model call ok)", history.trace(task.trace_key))

        # Caught failure path: a tool and the provider fail, application code
        # recovers. Both child failures stay visible; the root still exits ok.
        with promptkeep.agent("recovery-demo", input={"question": "Capital of Chile?"}) as retry:
            try:
                with promptkeep.tool("flaky_search", input={"query": "Chile capital"}) as search:
                    search.output({"partial": ["Santiago"]})
                    raise TimeoutError("search timed out")
            except TimeoutError as exc:
                print(f"\n  caught tool error: {type(exc).__name__}")
            try:
                client.chat.completions.create(model="demo-model", messages=MESSAGES)
            except Exception as exc:  # a real service would retry or degrade
                print(f"  caught provider error: {type(exc).__name__}")
                retry.output("degraded answer")

        show(
            "2. caught failures (root ok, child tool + llm errors)", history.trace(retry.trace_key)
        )

        # Escaped failure path: the root itself exits with an exception.
        try:
            with promptkeep.agent("failing-task") as failed:
                raise RuntimeError("application failed")
        except RuntimeError:
            pass
        show("3. escaped failure (root error)", history.trace(failed.trace_key))

        print(f"\nroots in history.list_traces(): {len(history.list_traces())}")


if __name__ == "__main__":
    main()
