"""A realistic multi-turn support chat with a PII gate — seed it, then view
it in the dashboard.

The story: a user is debugging a 401 error and, mid-conversation, pastes
their raw API key. A global pre-check catches it and blocks the call before
it ever reaches the model; the app notices the block, asks the user to retype
without the secret, and the clean message goes through. Exactly how a real
guarded assistant behaves.

Run from the repo root:
    uv run --no-sync python examples/pii_conversation_demo.py
    uv run promptkeep serve --db pii_demo.promptkeep.db
"""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.stdout.reconfigure(encoding="utf-8")

DB = Path(__file__).parent.parent / "pii_demo.promptkeep.db"
for leftover in (DB, Path(str(DB) + "-wal"), Path(str(DB) + "-shm")):
    leftover.unlink(missing_ok=True)

import promptkeep  # noqa: E402
from promptkeep import Prompt, PromptBlocked, Verdict, check, wrap  # noqa: E402

# --- the PII gate: global, so it guards *every* turn, not just the first -------
# (Follow-up turns carry no prompt of their own, so a prompt-scoped check would
#  miss them. Global scope is the right home for a guard like this.)
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9\-]{8,})|(\b\d{13,16}\b)")  # api keys, card numbers


@check.pre(name="no_secrets")
def block_secrets(ctx):
    """Stop any turn that contains a raw API key or card number."""
    if ctx.last_text and SECRET_RE.search(ctx.last_text):
        return Verdict.block("a secret (API key or card number) was in the message")
    return Verdict.ok()


promptkeep.configure(db_path=str(DB), write_mode="sync", pre=[block_secrets])


# --- a fake OpenAI-shaped client so this needs no network/key -----------------
class FakeCompletions:
    reply = "..."

    def create(self, **kwargs):
        return SimpleNamespace(
            id="resp",
            model=kwargs.get("model"),
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=12, total_tokens=42),
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


client = wrap(FakeClient())
completions = client.chat.completions  # the wrapper replaced .create; .reply still lives here
system = Prompt(
    "You are a support engineer for Acme Cloud. Help the user resolve their issue.",
    name="SUPPORT_AGENT",
)
SESSION = "support-8842"

history_msgs = [{"role": "system", "content": system}]


def user_turn(text, assistant_reply):
    """A normal turn that goes through: append user text, call, append reply."""
    history_msgs.append({"role": "user", "content": text})
    completions.reply = assistant_reply
    client.chat.completions.create(model="gpt-4o-mini", messages=history_msgs)
    history_msgs.append({"role": "assistant", "content": assistant_reply})
    print(f"  user: {text}\n  bot:  {assistant_reply}\n")


print("=" * 68)
print("A guarded support conversation")
print("=" * 68)

with promptkeep.conversation(SESSION, title="401 on deploy", user_id=42):
    user_turn(
        "My deployment started failing this morning with a 401 error.",
        "Thanks for reaching out. A 401 means authentication failed — often an "
        "expired or wrong API key. When did it last work?",
    )
    user_turn(
        "It worked yesterday. Nothing changed on my end that I know of.",
        "Keys on Acme rotate every 90 days. Let's check yours — but please don't "
        "paste the full key here.",
    )

    # The user ignores the warning and pastes their real key. The gate blocks it.
    leaked = "here it is: sk-live-9fA2Bq7Xk send that and check?"
    history_msgs.append({"role": "user", "content": leaked})
    print(f"  user: {leaked}")
    try:
        client.chat.completions.create(model="gpt-4o-mini", messages=history_msgs)
    except PromptBlocked as e:
        history_msgs.pop()  # drop the leaked message; it never went to the model
        print(f"  [BLOCKED by promptkeep] {e.message}")
        print(
            "  bot:  ⚠️ For your security I can't accept a full key. Please paste "
            "only the last 4 characters.\n"
        )

    # The user retypes safely; this one passes the gate.
    user_turn(
        "ok sorry — my key ends in 7Xk and it stopped working today.",
        "No problem. A key ending 7Xk that stopped today has almost certainly "
        "expired. Regenerate it under Settings > API keys, update your env, and "
        "redeploy — that should clear the 401.",
    )
    user_turn(
        "that fixed it, thank you!",
        "Glad to hear it! I'd also set a calendar reminder for the next rotation "
        "so it doesn't surprise you. Anything else?",
    )

print(f"Seeded {DB.name}.")
print(f"View it:  uv run promptkeep serve --db {DB}")
print(f"Then open:  http://127.0.0.1:8420/conversations/{SESSION}")
