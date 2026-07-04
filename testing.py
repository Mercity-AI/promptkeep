"""Playground for promptkeep — run with `uv run python testing.py`.

Not part of the test suite; a narrated sandbox to watch versioning, lineage,
run tracking, and the revert-to-old-text behavior actually happen. Uses its
own throwaway DB (testing.db) and starts fresh every run.
"""

from pathlib import Path
from types import SimpleNamespace

import promptkeep
from promptkeep import Prompt, history, prompt, wrap

# Fresh throwaway DB every run so the output below is deterministic.
DB = Path("testing.db")
for leftover in (DB, Path("testing.db-wal"), Path("testing.db-shm")):
    leftover.unlink(missing_ok=True)
promptkeep.configure(db_path=DB, enabled=True)

print("=" * 70)
print("1) BASIC PROMPT + VERSION 1")
print("=" * 70)

original_text = "You are a code reviewer. Focus on {focus}."

p1 = Prompt(original_text, {"focus": "correctness"}, name="REVIEW")
print(f"raw:      {p1.raw!r}")
print(f"rendered: {p1.text!r}")
print(f"version:  {p1.version}")
assert p1.version == 1

print()
print("=" * 70)
print("2) CHANGE THE TEXT -> NEW VERSION")
print("=" * 70)

p2 = Prompt("You are a strict code reviewer. Focus on {focus}. Be terse.",
            {"focus": "correctness"}, name="REVIEW")
print(f"edited text version: {p2.version}")
assert p2.version == 2

print()
print("=" * 70)
print("3) THE BIG ONE: REVERT TO THE ORIGINAL TEXT")
print("   (should resolve back to version 1, NOT create version 3)")
print("=" * 70)

p3 = Prompt(original_text, {"focus": "security"}, name="REVIEW")  # old text, new vars
print(f"reverted prompt version: {p3.version}")
assert p3.version == 1, "revert should dedup back to v1!"

lineage = history.versions("REVIEW")
print(f"total versions in lineage: {len(lineage)}  (still just 2 — no v3 created)")
assert len(lineage) == 2
for v in lineage:
    print(f"  v{v.version}: {v.template!r}")

print()
print("=" * 70)
print("4) VARIABLES NEVER CREATE VERSIONS")
print("=" * 70)

for focus in ("speed", "readability", "naming"):
    Prompt(original_text, {"focus": focus}, name="REVIEW").version
print(f"after 3 more variable changes, versions: {len(history.versions('REVIEW'))} (unchanged)")
assert len(history.versions("REVIEW")) == 2

print()
print("=" * 70)
print("5) DIFF BETWEEN v1 AND v2")
print("=" * 70)
print(history.diff("REVIEW", 1, 2))

print()
print("=" * 70)
print("6) DECORATOR PROMPT — computed template, same rules")
print("=" * 70)


@prompt(name="SUMMARIZE")
def summarize_prompt(style="bullet points", max_words=50):
    """Build a summarization prompt with a computed constraint line."""
    constraint = f"Use at most {max_words} words."
    return f"Summarize the following text as {{style}}. {constraint}\n\nText: {{text}}"


d1 = summarize_prompt()                # default max_words=50 -> one template
d2 = summarize_prompt(max_words=50)    # same computed template -> same version
d3 = summarize_prompt(max_words=100)   # different computed template -> new version
print(f"default call:        v{d1.version}")
print(f"same args again:     v{d2.version}  (deduped)")
print(f"max_words=100:       v{d3.version}  (template genuinely changed)")
print(f"rendered: {d3.render(style='a haiku', text='Lorem ipsum...')!r}")
assert d1.version == d2.version == 1 and d3.version == 2

print()
print("=" * 70)
print("7) A WRAPPED 'OPENAI' CALL -> RUN RECORDED")
print("=" * 70)


class FakeCompletions:
    """Just enough OpenAI-shape to watch the wrapper do its thing."""

    def create(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        print(f"  [the API received]: {type(content).__name__} -> {content!r}")
        return SimpleNamespace(
            id="resp_demo", model=kwargs["model"],
            usage=SimpleNamespace(prompt_tokens=21, completion_tokens=8, total_tokens=29),
            choices=[SimpleNamespace(message=SimpleNamespace(content="LGTM with nits."))],
        )


class FakeOpenAI:
    """Stand-in for openai.OpenAI so this file needs no API key."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


client = wrap(FakeOpenAI())
client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "developer", "content": p3}],   # p3 is the reverted v1 prompt
)

(run,) = history.runs("REVIEW")
print(f"  run recorded -> version: v{run.version} (the run knows it ran the REVERTED v1)")
print(f"  variables:  {run.variables}")
print(f"  output:     {run.output_text!r}")
print(f"  usage:      {run.total_tokens} tokens | model: {run.model} | {run.latency_ms}ms")
assert run.version == 1 and run.variables == {"focus": "security"}

print()
print("=" * 70)
print("8) RENAMING A VARIABLE IS *NOT* A NEW VERSION")
print("   (matching ignores placeholder names — only static text counts)")
print("=" * 70)

r1 = Prompt("Grade this essay on {var1}. Reply in {var2}.", name="GRADER")
print(f"original names {{var1}}/{{var2}}:      v{r1.version}")

r2 = Prompt("Grade this essay on {x}. Reply in {y}.", name="GRADER")
print(f"renamed to {{x}}/{{y}}:                v{r2.version}  (same version!)")
assert r2.version == r1.version == 1

r3 = Prompt("Grade this essay on {topic}. Reply in {language}.", name="GRADER")
print(f"renamed to {{topic}}/{{language}}:     v{r3.version}  (still the same)")
assert r3.version == 1

# But structure is respected: same words, different repetition pattern.
s1 = Prompt("Compare {a} with {a}.", name="COMPARE")
s2 = Prompt("Compare {a} with {b}.", name="COMPARE")
print(f"'{{a}} with {{a}}' vs '{{a}} with {{b}}':  v{s1.version} vs v{s2.version}  (different — one value twice != two values)")
assert (s1.version, s2.version) == (1, 2)

# And actual wording changes still bump the version, of course.
r4 = Prompt("Grade this essay harshly on {topic}. Reply in {language}.", name="GRADER")
print(f"actually changed wording:          v{r4.version}")
assert r4.version == 2
print(f"GRADER lineage: {len(history.versions('GRADER'))} versions (3 renames collapsed into v1)")

print()
print("=" * 70)
print("ALL CHECKS PASSED — lineage, dedup-on-revert, rename-immunity,")
print("and run tracking all work.")
print("=" * 70)
