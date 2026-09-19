"""A live smoke test against a real endpoint — the one thing the fakes in
tests/ can't prove: that a real provider's responses are shaped the way the
adapters read them.

It makes three tiny calls and prints what promptkeep recorded for each:

1. a chat completion            -> tokens, and ``cost_usd`` if the endpoint reports one
2. a streamed chat completion   -> the same, read off the stream's last chunk
3. two chained Responses calls  -> grouped into one conversation by previous_response_id
   (skipped with a note if the endpoint has no Responses API, or no chaining)

Run from the repo root with a key in the environment (never on the command line):

    OPENROUTER_API_KEY=... uv run --with openai python examples/live_smoke.py
    OPENAI_API_KEY=...     uv run --with openai python examples/live_smoke.py

Costs a fraction of a cent. Writes to a throwaway DB next to this file.
"""

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB = Path(__file__).parent / "live_smoke.promptkeep.db"
for leftover in (DB, Path(str(DB) + "-wal"), Path(str(DB) + "-shm")):
    leftover.unlink(missing_ok=True)

from openai import OpenAI  # noqa: E402

import promptkeep  # noqa: E402
from promptkeep import Prompt, history, wrap  # noqa: E402

# --- which endpoint: OpenRouter if its key is set, else OpenAI --------------------
if os.environ.get("OPENROUTER_API_KEY"):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]
    )
    model = os.environ.get("SMOKE_MODEL", "openai/gpt-4o-mini")
    endpoint = "OpenRouter"
elif os.environ.get("OPENAI_API_KEY"):
    client = OpenAI()
    model = os.environ.get("SMOKE_MODEL", "gpt-4o-mini")
    endpoint = "OpenAI"
else:
    raise SystemExit("Set OPENROUTER_API_KEY or OPENAI_API_KEY in the environment first.")

promptkeep.configure(db_path=str(DB), write_mode="sync")
client = wrap(client)
SYSTEM = Prompt("You answer in at most {n} words.", {"n": 5}, name="SMOKE_SYSTEM")
print(f"endpoint: {endpoint} · model: {model}\n")


def show(title: str, run) -> None:
    """Print the recorded fields this script exists to eyeball."""
    print(f"{title}")
    print(f"  output:  {run.output_text!r}")
    print(f"  tokens:  {run.prompt_tokens} in / {run.completion_tokens} out")
    print(f"  cost:    {history.format_cost(run.cost_usd)}   (— means the endpoint reported none)")
    print(f"  run_key: {run.run_key}\n")


# --- 1. a chat completion ------------------------------------------------------------
client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Capital of Peru?"},
    ],
)
show("1. chat completion", history.runs("SMOKE_SYSTEM")[0])

# --- 2. the same, streamed --------------------------------------------------------------
stream = client.chat.completions.create(
    model=model,
    stream=True,
    stream_options={"include_usage": True},  # OpenAI needs this; OpenRouter ignores it
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Capital of Chile?"},
    ],
)
for _chunk in stream:
    pass
show("2. streamed chat completion", history.runs("SMOKE_SYSTEM")[0])

# --- 3. two chained Responses calls ------------------------------------------------------
try:
    first = client.responses.create(model=model, instructions=SYSTEM, input="Capital of Peru?")
    show("3a. responses.create", history.runs("SMOKE_SYSTEM")[0])
    client.responses.create(
        model=model, instructions=SYSTEM, input="And of Chile?", previous_response_id=first.id
    )
    convo = history.conversation(f"response:{first.id}")
    print(f"3b. chained call -> conversation {convo.external_id!r}, {len(convo.turns)} turns:")
    for turn in convo.turns:
        print(f"  #{turn.turn_index}  {turn.input_text!r} -> {turn.output_text!r}")
    print(f"  total cost: {history.format_cost(convo.total_cost)}")
except Exception as exc:  # an endpoint without the Responses API, or without chaining
    print(f"3. Responses API not exercised on this endpoint: {type(exc).__name__}: {exc}")

print(f"\nEverything above is in {DB.name}:  uv run promptkeep serve --db {DB}")
