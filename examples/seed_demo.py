"""Seed a demo database for the dashboard: `promptkeep serve --db demo.promptkeep.db`.

Everything flows through the real pipeline — a wrapped OpenAI-shaped fake
client and `promptkeep.conversation(...)` blocks — so versions, turn
indexes, and input extraction are produced by the actual wrapper, exactly
as they would be in production. No network, no API key.

A cosmetic pass at the end spreads timestamps over the past week and fills
in realistic latencies, so list views look lived-in rather than seeded.

Run from the repo root:  uv run python examples/seed_demo.py
"""

import random
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

DB_PATH = Path(__file__).parent.parent / "demo.promptkeep.db"
for suffix in ("", "-wal", "-shm"):
    p = Path(str(DB_PATH) + suffix)
    if p.exists():
        p.unlink()

import promptkeep  # noqa: E402
from promptkeep import Prompt, PromptBlocked, Verdict, check  # noqa: E402

promptkeep.configure(db_path=str(DB_PATH))
random.seed(42)


# --- a minimal OpenAI-shaped fake (same surface tests/fakes.py mirrors) --------


# Dollars per million tokens (input, output) — only so the fake can report a
# believable cost. promptkeep itself never prices anything: it stores the
# ``usage.cost`` an endpoint like OpenRouter sends back, which this imitates.
_PRICES = {"gpt-4o-mini": (0.15, 0.60), "o4-mini": (1.10, 4.40), "gpt-4.1": (2.00, 8.00)}


def _response(text, model, prompt_tokens, completion_tokens):
    """One chat-completion response shaped like the real SDK object, with the
    OpenRouter-style ``usage.cost`` on it."""
    price_in, price_out = _PRICES.get(model, (1.0, 3.0))
    return SimpleNamespace(
        id=f"resp_{random.randrange(10**8):08x}",
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=(prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


class FakeCompletions:
    """Replays whatever `next_result` holds; an Exception instance raises."""

    def __init__(self):
        self.next_result = None

    def create(self, **kwargs):
        result = self.next_result
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


client = promptkeep.wrap(FakeClient())


def call(messages, reply, model="gpt-4o-mini", conversation=None):
    """One tracked call: canned reply in, wrapper records everything."""
    tokens_in = sum(len(str(m.get("content", ""))) // 4 for m in messages) + 20
    client.chat.completions.next_result = _response(
        reply, model, tokens_in, max(len(reply) // 4, 5)
    )
    kwargs = {"model": model, "messages": messages}
    if conversation:
        kwargs["promptkeep_conversation"] = conversation
    response = client.chat.completions.create(**kwargs)
    _maybe_feedback(response)
    return response


def _maybe_feedback(response):
    """Some users rate what they got: mostly thumbs up, the odd complaint. The
    key comes off the response — every recorded call carries one."""
    roll = random.random()
    key = response.promptkeep.run_key
    if roll < 0.30:
        promptkeep.feedback(key, score=1.0, label="thumbs_up")
    elif roll < 0.40:
        promptkeep.feedback(
            key, score=0.0, label="thumbs_down", comment="didn't answer the question"
        )


def fail(messages, error, model="gpt-4o-mini", conversation=None):
    """One failing tracked call: the wrapper records the error and re-raises."""
    client.chat.completions.next_result = error
    kwargs = {"model": model, "messages": messages}
    if conversation:
        kwargs["promptkeep_conversation"] = conversation
    try:
        client.chat.completions.create(**kwargs)
    except Exception:
        pass


# --- prompts with real version lineage -----------------------------------------

SUPPORT_V1 = Prompt(
    "You are a support agent for Acme Cloud. Help the user with their question.",
    name="SUPPORT_AGENT",
)
SUPPORT_V2 = Prompt(
    "You are a support agent for Acme Cloud. Help the user with their {product} "
    "question. Be warm, be concise, and always end with a next step.",
    {"product": "billing"},
    name="SUPPORT_AGENT",
)
REVIEW_V1 = Prompt("You are a code reviewer.", name="REVIEW_SYSTEM")
REVIEW_V2 = Prompt(
    "You are a code reviewer. Focus on {focus}.", {"focus": "security"}, name="REVIEW_SYSTEM"
)
REVIEW_V3 = Prompt(
    "You are a senior code reviewer. Focus on {focus}. Be concise and cite line numbers.",
    {"focus": "security"},
    name="REVIEW_SYSTEM",
)
SUMMARIZE_V1 = Prompt("Summarize the following text in one sentence.", name="SUMMARIZE")
SUMMARIZE_V2 = Prompt(
    "Summarize the following text in {n} bullet points. Plain language only.",
    {"n": 3},
    name="SUMMARIZE",
)
EXTRACT = Prompt(
    "Extract every person, company, and date from the text as JSON.", name="EXTRACT_ENTITIES"
)

# Registration is lazy (first render wins the next version number), so touch
# each lineage in narrative order — otherwise whichever prompt happens to be
# called first would become v1 regardless of the story we're telling here.
for _p in (
    SUPPORT_V1,
    SUPPORT_V2,
    REVIEW_V1,
    REVIEW_V2,
    REVIEW_V3,
    SUMMARIZE_V1,
    SUMMARIZE_V2,
    EXTRACT,
):
    _p.version


# --- conversation 1: a six-turn support thread ----------------------------------

with promptkeep.conversation("user-42-session-9", title="Password reset", user_id=42):
    history = [{"role": "system", "content": SUPPORT_V2.format(product="account")}]

    def turn(user_text, reply):
        history.append({"role": "user", "content": user_text})
        response = call(history, reply)
        history.append({"role": "assistant", "content": reply})
        return response

    turn(
        "Hi, I can't log into my account anymore. It says my password is wrong "
        "but I'm sure it's right.",
        "Sorry about that! Let's get you back in. First, could you confirm the email "
        "address on the account? Then try the 'Forgot password' link on the sign-in page.",
    )
    turn(
        "It's meera@example.com. I clicked forgot password but no email arrived.",
        "Thanks, Meera. Reset emails can take up to five minutes — and do check spam. "
        "If nothing lands, I can trigger a manual reset from our side. Want me to?",
    )
    turn(
        "Checked spam, nothing. Yes please trigger it manually.",
        "Done — a reset link was just sent from noreply@acme.cloud. It expires in 30 "
        "minutes. Next step: open it and set a new password.",
    )
    turn(
        "Got it! I'm in now. One more thing — can I turn on two-factor auth?",
        "Great! Yes: Settings > Security > Two-factor authentication. You can use an "
        "authenticator app or SMS. The app option is the more secure next step.",
    )
    turn(
        "Done. Thanks for the quick help!",
        "Anytime, Meera. You're all set with a fresh password and 2FA enabled. Have a great day!",
    )

# --- conversation 2: billing thread with a failed call --------------------------

with promptkeep.conversation("user-7-session-1", title="Double charge", user_id=7):
    msgs = [{"role": "system", "content": SUPPORT_V2}]
    msgs.append(
        {"role": "user", "content": "I was charged twice for March. Invoice INV-2201 and INV-2214."}
    )
    call(
        msgs,
        "I can see both invoices. INV-2214 looks like a duplicate — let me check with billing.",
    )
    msgs.append(
        {
            "role": "assistant",
            "content": "I can see both invoices. INV-2214 looks like a duplicate — let me check with billing.",
        }
    )
    msgs.append({"role": "user", "content": "OK, how long will that take?"})
    fail(msgs, RuntimeError("RateLimitError: rate limited, retry in 20s"))
    call(
        msgs,
        "Sorry for the hiccup. Refunds for duplicate charges land in 3-5 business days. I've filed it as ticket #8841.",
    )

# --- conversation 3: an onboarding thread on the older prompt version ------------

with promptkeep.conversation("user-19-session-3", title="Onboarding questions", user_id=19):
    msgs = [{"role": "system", "content": SUPPORT_V1}]
    for question, answer in [
        (
            "How do I invite teammates to my workspace?",
            "Settings > Members > Invite. Enter their emails and pick a role — Admin, Editor, or Viewer.",
        ),
        (
            "What's the difference between Editor and Admin?",
            "Editors can create and modify projects. Admins can also manage members, billing, and workspace settings.",
        ),
        (
            "Can I limit a member to a single project?",
            "Yes — use a project-level invite instead: open the project, Share, and add them there. They won't see anything else.",
        ),
        (
            "Is there an audit log?",
            "On the Team plan and up, yes: Settings > Audit log, with export to CSV.",
        ),
    ]:
        msgs.append({"role": "user", "content": question})
        call(msgs, answer, model="gpt-4o")
        msgs.append({"role": "assistant", "content": answer})

# --- conversation 4: agent loop reusing REVIEW_SYSTEM v3 every turn --------------

with promptkeep.conversation("agent-review-loop-1", title="PR #312 review", repo="acme/api"):
    for file, verdict in [
        (
            "auth/session.py",
            "Line 44: token comparison uses ==; switch to hmac.compare_digest. Line 90: session TTL never enforced.",
        ),
        (
            "api/routes.py",
            "Line 12: user input interpolated into SQL string — parameterize. Otherwise clean.",
        ),
        (
            "db/models.py",
            "No security issues. Minor: password_hash column is nullable; make it NOT NULL.",
        ),
    ]:
        call(
            [
                {"role": "system", "content": REVIEW_V3},
                {"role": "user", "content": f"Review {file} from PR #312."},
            ],
            verdict,
            model="o4-mini",
        )

# --- standalone runs: no conversation, mixed prompts/models/outcomes -------------

call(
    [
        {"role": "system", "content": SUMMARIZE_V1},
        {
            "role": "user",
            "content": "Q3 revenue grew 18% year over year, driven mostly by the enterprise tier...",
        },
    ],
    "Enterprise-tier growth pushed Q3 revenue up 18% year over year.",
)
call(
    [{"role": "system", "content": SUMMARIZE_V2}],
    "- Revenue up 18% YoY\n- Enterprise tier drove growth\n- Churn flat at 2.1%",
    model="gpt-4o",
)
call(
    [{"role": "system", "content": SUMMARIZE_V2.format(n=5)}],
    "- Revenue up 18%\n- Enterprise-led\n- Churn 2.1%\n- NRR 117%\n- Guidance raised",
    model="gpt-4o",
)
call(
    [
        {"role": "system", "content": EXTRACT},
        {"role": "user", "content": "Priya Sharma of Northwind met Alex Chen on 2026-03-14."},
    ],
    '{"people": ["Priya Sharma", "Alex Chen"], "companies": ["Northwind"], "dates": ["2026-03-14"]}',
    model="gpt-4o-mini",
)
call(
    [
        {"role": "system", "content": REVIEW_V1},
        {"role": "user", "content": "def add(a, b): return a + b"},
    ],
    "Looks fine. Consider type hints.",
)
call(
    [
        {"role": "system", "content": REVIEW_V2},
        {"role": "user", "content": "eval(request.args['q'])"},
    ],
    "Critical: eval on user input is remote code execution. Remove immediately.",
)
fail(
    [{"role": "system", "content": EXTRACT}, {"role": "user", "content": "..."}],
    RuntimeError("APITimeoutError: request timed out after 30s"),
)
fail(
    [{"role": "system", "content": SUMMARIZE_V2}],
    RuntimeError("AuthenticationError: invalid API key"),
)


# --- checked runs: pre-gates and post-audits, so the dashboard's checks view
#     and the runs verdict badges have data across ok / warn / blocked ----------


@check.pre(name="no_secrets")
def no_secrets(ctx):
    """Block a prompt that leaks something shaped like an API key."""
    if re.search(r"sk-[A-Za-z0-9\-]{6,}", ctx.rendered):
        return Verdict.block("an API key was in the prompt")
    return Verdict.ok()


@check.post(name="grounded", mode="blocking")
def grounded(ctx):
    """Toy groundedness score: a one-liner reply reads as unsupported."""
    return Verdict.from_score(0.92 if len(ctx.output_text or "") > 40 else 0.3, threshold=0.5)


@check.post(name="offers_next_step", mode="blocking")
def offers_next_step(ctx):
    """Warn (don't block) when a support reply names no next step."""
    text = (ctx.output_text or "").lower()
    return Verdict.ok() if ("next" in text or "step" in text) else Verdict.warn("no next step")


def checked_call(messages, reply, pre=None, post=None, model="gpt-4o-mini"):
    """A tracked call carrying per-call checks; swallow a block like a real app."""
    tokens_in = sum(len(str(m.get("content", ""))) // 4 for m in messages) + 20
    client.chat.completions.next_result = _response(
        reply, model, tokens_in, max(len(reply) // 4, 5)
    )
    kwargs = {"model": model, "messages": messages}
    if pre:
        kwargs["promptkeep_pre"] = pre
    if post:
        kwargs["promptkeep_post"] = post
    try:
        client.chat.completions.create(**kwargs)
    except PromptBlocked:
        pass  # recorded as a blocked run; the app would ask the user to retry


# both audits pass -> verification "ok"
checked_call(
    [
        {"role": "system", "content": REVIEW_V3.format(focus="security")},
        {"role": "user", "content": "def login(pw): return pw == stored_pw"},
    ],
    "Timing-unsafe comparison — use hmac.compare_digest. Next step: patch it and add a test.",
    post=[grounded, offers_next_step],
)
# a thin reply -> grounded warns, no next step warns -> verification "warn"
checked_call(
    [
        {"role": "system", "content": SUPPORT_V2.format(product="billing")},
        {"role": "user", "content": "why was I charged twice?"},
    ],
    "Let me check.",
    post=[grounded, offers_next_step],
)
# a leaked key never reaches the model -> a blocked run
checked_call(
    [
        {"role": "system", "content": SUPPORT_V1},
        {"role": "user", "content": "my key is sk-live-9fA2Bq7Xk — is it still valid?"},
    ],
    "(never sent)",
    pre=[no_secrets],
)


# --- cosmetic pass: spread timestamps over the past week, add latencies ----------

# Runs are persisted by the background writer; make sure every row is on disk
# before rewriting them with plain sqlite3, or the writer lands them afterwards
# with their real (seconds-apart) timestamps.
promptkeep.flush(timeout=10)

conn = sqlite3.connect(DB_PATH)
run_ids = [r[0] for r in conn.execute("SELECT id FROM runs ORDER BY id")]
now = datetime.now(UTC)
start = now - timedelta(days=6, hours=3)
step = (now - start) / max(len(run_ids), 1)
for i, run_id in enumerate(run_ids):
    ts = (start + step * i + timedelta(seconds=random.randrange(-900, 900))).isoformat()
    latency = random.randrange(350, 2600)
    conn.execute(
        "UPDATE runs SET created_at = ?, latency_ms = ? WHERE id = ?", (ts, latency, run_id)
    )
conn.execute(
    """UPDATE conversations SET
         created_at = (SELECT MIN(created_at) FROM runs WHERE conversation_id = conversations.id),
         updated_at = (SELECT MAX(created_at) FROM runs WHERE conversation_id = conversations.id)"""
)
conn.commit()

runs_total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
convos = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
prompts = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
versions = conn.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()[0]
conn.close()

print(
    f"Seeded {DB_PATH.name}: {prompts} prompts, {versions} versions, {runs_total} runs, {convos} conversations"
)
print(f"View it:  uv run promptkeep serve --db {DB_PATH}")
