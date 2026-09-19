"""Tests for the local dashboard (`promptkeep serve`): FastAPI routes against
a seeded temp DB. fastapi/httpx are dev-only extras exercised here — the core
package works with neither installed."""

from fastapi.testclient import TestClient

from promptkeep import Prompt, storage, tracking
from promptkeep.dashboard.app import create_app


def _seed():
    """Two versions of one prompt with runs, plus one conversation (one turn
    of which errored) to exercise every row-rendering branch."""
    p1 = Prompt("hi {x}", {"x": 1}, name="REVIEW")
    p1.version
    p2 = Prompt("hi {x} there", {"x": 2}, name="REVIEW")
    p2.version
    tracking.record_prompt_run(
        p1, {"x": 1}, "hi 1", provider="openai", model="m", output_text="out1"
    )
    tracking.record_prompt_run(
        p2, {"x": 2}, "hi 2 there", provider="openai", model="m", output_text="out2"
    )

    cid = storage.get_or_create_conversation("sess-1", title="Demo session")
    storage.record_run(provider="openai", conversation_id=cid, input_text="q1", output_text="a1")
    storage.record_run(
        provider="openai",
        conversation_id=cid,
        input_text="q2",
        output_text="a2",
        status="error",
        error="boom",
    )


def _client():
    """A fresh TestClient wrapping a fresh app -- routes read live through
    history/storage, so no app-level state needs resetting between tests."""
    return TestClient(create_app())


class TestPromptsRoutes:
    def test_root_redirects_to_prompts(self):
        r = _client().get("/", follow_redirects=False)
        assert r.status_code in (302, 307)
        assert r.headers["location"] == "/prompts"

    def test_prompts_list_empty(self):
        r = _client().get("/prompts")
        assert r.status_code == 200
        assert "No prompts recorded yet" in r.text

    def test_prompts_list_shows_counts(self):
        _seed()
        r = _client().get("/prompts")
        assert r.status_code == 200
        assert "REVIEW" in r.text
        # version_count and run_count are both 2 for REVIEW.
        assert r.text.count('<td class="num">2</td>') == 2

    def test_prompt_detail_shows_versions(self):
        _seed()
        r = _client().get("/prompts/REVIEW")
        assert r.status_code == 200
        assert "v1" in r.text and "v2" in r.text

    def test_prompt_detail_unknown_is_404(self):
        r = _client().get("/prompts/NOPE")
        assert r.status_code == 404

    def test_diff_between_versions(self):
        _seed()
        r = _client().get("/prompts/REVIEW/diff", params={"old": 1, "new": 2})
        assert r.status_code == 200
        assert "there" in r.text

    def test_diff_unknown_version_is_404(self):
        _seed()
        r = _client().get("/prompts/REVIEW/diff", params={"old": 1, "new": 99})
        assert r.status_code == 404


class TestRunsRoute:
    def test_runs_page_lists_everything_by_default(self):
        _seed()
        r = _client().get("/runs")
        assert r.status_code == 200
        assert "REVIEW" in r.text
        assert "sess-1" in r.text  # conversation-only turns show up too
        assert "status-error" in r.text  # the errored turn is flagged

    def test_runs_page_filters_by_prompt(self):
        _seed()
        r = _client().get("/runs", params={"prompt": "REVIEW"})
        assert r.status_code == 200
        assert "sess-1" not in r.text  # conversation turns excluded once filtered

    def test_blank_version_param_is_not_an_error(self):
        """The form submits `version=` when the box is left empty; that must
        mean 'no version filter', never a 422 — regression for the filter
        appearing broken."""
        _seed()
        r = _client().get("/runs?prompt=REVIEW&version=")
        assert r.status_code == 200
        assert "REVIEW" in r.text

    def test_version_filter_narrows_results(self):
        _seed()
        r = _client().get("/runs", params={"prompt": "REVIEW", "version": "1"})
        assert r.status_code == 200
        assert "out1" not in r.text  # outputs aren't shown; check via badges
        assert 'class="badge">v1' in r.text
        assert 'class="badge">v2' not in r.text

    def test_garbage_version_param_is_ignored(self):
        _seed()
        r = _client().get("/runs", params={"prompt": "REVIEW", "version": "abc"})
        assert r.status_code == 200


class TestConversationsRoutes:
    def test_conversations_list(self):
        _seed()
        r = _client().get("/conversations")
        assert r.status_code == 200
        assert "sess-1" in r.text
        assert "Demo session" in r.text

    def test_conversation_detail_shows_turns(self):
        _seed()
        r = _client().get("/conversations/sess-1")
        assert r.status_code == 200
        assert "q1" in r.text and "a1" in r.text
        assert "q2" in r.text and "a2" in r.text

    def test_conversations_list_filters_by_prompt_and_version(self):
        """The filter form narrows to sessions a prompt (version) drove; a
        blank version box is not an error."""
        _seed()
        cid = storage.get_or_create_conversation("sess-review", title="Driven by REVIEW")
        p1 = Prompt("hi {x}", {"x": 1}, name="REVIEW")
        tracking.record_prompt_run(
            p1, {"x": 1}, "hi 1", provider="openai", conversation_id=cid, output_text="o"
        )
        client = _client()
        assert "sess-review" in client.get("/conversations").text
        filtered = client.get("/conversations", params={"prompt": "REVIEW", "version": ""})
        assert filtered.status_code == 200
        assert "sess-review" in filtered.text
        assert "sess-1" not in filtered.text
        narrowed = client.get("/conversations", params={"prompt": "REVIEW", "version": "2"})
        assert "sess-review" not in narrowed.text
        assert "No conversations recorded with REVIEW" in narrowed.text

    def test_conversation_detail_shows_stats(self):
        """Turn count, tokens, duration and the driving versions head the page."""
        _seed()
        r = _client().get("/conversations/sess-1")
        assert "2 turns" in r.text

    def test_reported_cost_shows_on_runs_and_conversation(self):
        cid = storage.get_or_create_conversation("sess-cost")
        storage.record_run(provider="openai", conversation_id=cid, input_text="q", cost_usd=0.0042)
        client = _client()
        assert "$0.0042" in client.get("/runs").text
        assert "$0.0042" in client.get("/conversations/sess-cost").text

    def test_conversation_detail_unknown_is_404(self):
        r = _client().get("/conversations/nope")
        assert r.status_code == 404
