"""The `promptkeep` CLI: each command is a presentation of a history read, so
these seed a DB, run ``main([...])`` and read what was printed."""

import json

import pytest

import promptkeep
from promptkeep import Prompt, cli, history, storage, tracking


@pytest.fixture
def seeded(isolated_db):
    """Two versions of REVIEW (v1 with a checked, labelled, priced run; v2
    with an error), and one two-turn conversation. Returns the DB path."""
    v1 = Prompt("Review {what}.", {"what": "code"}, name="REVIEW")
    v2 = Prompt("Review {what}.\nBe brief.", {"what": "code"}, name="REVIEW")
    key = tracking.record_prompt_run(
        v1,
        {"what": "code"},
        "Review code.",
        provider="openai",
        model="gpt-test",
        output_text="looks fine",
        total_tokens=30,
        cost_usd=0.0042,
        latency_ms=120,
        checks=[{"name": "grounded", "phase": "post", "status": "ok", "score": 0.9}],
    )
    promptkeep.feedback(key, score=1.0, label="thumbs_up")
    tracking.record_prompt_run(
        v2,
        {"what": "code"},
        "Review code.\nBe brief.",
        provider="openai",
        status="error",
        error="boom",
    )
    cid = storage.get_or_create_conversation("sess-1", title="Demo session")
    tracking.record_prompt_run(
        v2,
        {},
        "Review code.\nBe brief.",
        provider="openai",
        conversation_id=cid,
        input_text="first question",
        output_text="first answer",
        total_tokens=12,
    )
    storage.record_run(
        provider="openai",
        conversation_id=cid,
        input_text="card [x]",
        original_input_text="card 4111",
        output_text="second answer",
    )
    return isolated_db


def run(capsys, *argv):
    """Run the CLI and return what it printed to stdout."""
    cli.main(list(argv))
    return capsys.readouterr().out


class TestList:
    def test_lists_prompts_with_counts(self, seeded, capsys):
        out = run(capsys, "list")
        header, row = out.strip().splitlines()
        assert header.split() == ["PROMPT", "VERSIONS", "RUNS", "CREATED"]
        assert row.split()[:3] == ["REVIEW", "2", "3"]

    def test_empty_database(self, isolated_db, capsys):
        storage.get_db()  # the file exists, with nothing in it
        assert "no prompts recorded yet" in run(capsys, "list")

    def test_db_flag_points_at_another_file(self, seeded, tmp_path, capsys):
        other = tmp_path / "other.db"
        promptkeep.configure(db_path=other)
        Prompt("x", name="ELSEWHERE").version
        storage.reset_caches()
        promptkeep.configure(db_path=tmp_path / "unrelated.db")
        assert "ELSEWHERE" in run(capsys, "list", "--db", str(other))

    def test_a_missing_database_is_an_error_and_is_not_created(self, tmp_path):
        missing = tmp_path / "typo.db"
        with pytest.raises(SystemExit, match="no database at"):
            cli.main(["list", "--db", str(missing)])
        assert not missing.exists()


class TestVersionsAndDiff:
    def test_versions_table(self, seeded, capsys):
        out = run(capsys, "versions", "REVIEW")
        assert "v1" in out and "v2" in out
        assert "Review {what}." in out
        assert "Be brief." not in out  # only the first line in the table

    def test_versions_full_prints_whole_templates(self, seeded, capsys):
        assert "Be brief." in run(capsys, "versions", "REVIEW", "--full")

    def test_unknown_prompt(self, seeded):
        with pytest.raises(SystemExit, match="no prompt named 'NOPE'"):
            cli.main(["versions", "NOPE"])

    def test_diff_is_plain_when_not_a_terminal(self, seeded, capsys):
        out = run(capsys, "diff", "REVIEW", "1", "2")
        assert "+Be brief." in out
        assert "\033[" not in out

    def test_diff_is_coloured_on_a_terminal(self, seeded, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert "\033[32m+Be brief." in run(capsys, "diff", "REVIEW", "1", "2")
        assert "\033[" not in run(capsys, "diff", "REVIEW", "1", "2", "--no-color")

    def test_diff_of_a_version_with_itself(self, seeded, capsys):
        assert "identical" in run(capsys, "diff", "REVIEW", "1", "1")

    def test_diff_unknown_version(self, seeded):
        with pytest.raises(SystemExit, match="has no version 9"):
            cli.main(["diff", "REVIEW", "1", "9"])


class TestRuns:
    def test_runs_for_a_prompt(self, seeded, capsys):
        out = run(capsys, "runs", "REVIEW")
        assert len(out.strip().splitlines()) == 4  # header + 3
        assert "$0.0042" in out and "120ms" in out and "error" in out

    def test_version_filter_and_limit(self, seeded, capsys):
        assert len(run(capsys, "runs", "REVIEW", "--version", "1").strip().splitlines()) == 2
        assert len(run(capsys, "runs", "REVIEW", "--limit", "1").strip().splitlines()) == 2

    def test_all_runs_includes_promptless_turns(self, seeded, capsys):
        out = run(capsys, "runs")
        assert len(out.strip().splitlines()) == 5  # header + 4

    def test_version_without_a_name(self, seeded):
        with pytest.raises(SystemExit, match="needs a prompt name"):
            cli.main(["runs", "--version", "1"])

    def test_no_runs(self, seeded, capsys):
        Prompt("unused", name="UNUSED").version
        assert "no runs recorded" in run(capsys, "runs", "UNUSED")


class TestConvo:
    def test_transcript(self, seeded, capsys):
        out = run(capsys, "convo", "sess-1")
        assert out.splitlines()[0] == "Demo session"
        assert "2 turns" in out and "REVIEW v2" in out
        assert "user: first question" in out and "assistant: first answer" in out
        assert "user (as received): card 4111" in out and "user: card [x]" in out

    def test_labels_are_shown_under_their_turn(self, isolated_db, capsys):
        cid = storage.get_or_create_conversation("labelled")
        key = storage.record_run(
            provider="openai",
            conversation_id=cid,
            input_text="q",
            checks=[{"name": "no_pii", "phase": "pre", "status": "warn", "score": None}],
        )
        promptkeep.feedback(key, score=0.5, label="meh")
        out = run(capsys, "convo", "labelled")
        assert "[pre] no_pii: warn" in out
        assert "[feedback] meh (0.5)" in out

    def test_a_branch_says_where_it_forks_from(self, isolated_db, capsys):
        cid = storage.get_or_create_conversation("forked")
        root = storage.record_run(provider="openai", conversation_id=cid, input_text="q0")
        storage.record_run(provider="openai", conversation_id=cid, input_text="q1")
        storage.record_run(
            provider="openai", conversation_id=cid, input_text="q1 again", parent_run_key=root
        )
        outside = storage.record_run(provider="openai")
        storage.record_run(
            provider="openai", conversation_id=cid, input_text="sub", parent_run_key=outside
        )
        out = run(capsys, "convo", "forked")
        assert out.count("↳ branches from") == 2
        assert "#2 · ok · —\n  ↳ branches from #0" in out
        assert "↳ branches from another conversation" in out

    def test_unknown_conversation(self, seeded):
        with pytest.raises(SystemExit, match="no conversation found"):
            cli.main(["convo", "nope"])


class TestStats:
    def test_one_row_per_version(self, seeded, capsys):
        header, v1, v2 = run(capsys, "stats", "REVIEW").strip().splitlines()
        assert header.split()[:4] == ["VER", "RUNS", "ERRORS", "BLOCKED"]
        assert v1.split() == ["v1", "1", "0", "0", "100%", "0.90", "1.00", "120ms", "30", "$0.0042"]
        assert v2.split()[:4] == ["v2", "2", "1", "0"]
        assert v2.split()[4:7] == ["—", "—", "—"]  # nothing checked, scored or rated

    def test_unknown_prompt(self, seeded):
        with pytest.raises(SystemExit, match="no prompt named"):
            cli.main(["stats", "NOPE"])


class TestExport:
    def test_jsonl_to_stdout_carries_the_labels(self, seeded, capsys):
        lines = run(capsys, "export", "--prompt", "REVIEW", "--version", "1").strip().splitlines()
        (record,) = [json.loads(line) for line in lines]
        assert record["prompt_name"] == "REVIEW" and record["version"] == 1
        assert record["variables"] == {"what": "code"}
        assert record["cost_usd"] == 0.0042
        assert [(c["phase"], c["name"]) for c in record["checks"]] == [
            ("post", "grounded"),
            ("feedback", "thumbs_up"),
        ]

    def test_everything_by_default(self, seeded, capsys):
        assert len(run(capsys, "export").strip().splitlines()) == 4

    def test_to_a_file(self, seeded, tmp_path, capsys):
        target = tmp_path / "runs.jsonl"
        cli.main(["export", "--limit", "2", "-o", str(target)])
        captured = capsys.readouterr()
        assert captured.out == "" and "wrote 2 runs" in captured.err
        assert len(target.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_version_without_a_prompt(self, seeded):
        with pytest.raises(SystemExit, match="needs --prompt"):
            cli.main(["export", "--version", "1"])


class TestStatsRead:
    """history.stats(), the read behind `promptkeep stats`."""

    def test_a_run_with_several_verdicts_is_counted_once(self, isolated_db):
        p = Prompt("t {x}", name="FANOUT")
        tracking.record_prompt_run(
            p,
            {},
            "t",
            provider="o",
            total_tokens=10,
            checks=[
                {"name": "a", "phase": "post", "status": "ok", "score": 1.0},
                {"name": "b", "phase": "post", "status": "warn", "score": 0.0},
                {"name": "c", "phase": "pre", "status": "ok", "score": None},
            ],
        )
        (row,) = history.stats("FANOUT")
        assert row.runs == 1 and row.avg_tokens == 10
        assert row.checked_runs == 1 and row.check_pass_rate == 0.0  # one warn fails the run
        assert row.avg_score == 0.5

    def test_a_version_with_no_runs_still_has_a_row(self, isolated_db):
        Prompt("never run", name="IDLE").version
        (row,) = history.stats("IDLE")
        assert (row.runs, row.errors, row.total_cost, row.check_pass_rate) == (0, 0, None, None)

    def test_blocked_runs_are_counted(self, isolated_db):
        p = Prompt("t", name="GATED")
        tracking.record_prompt_run(p, {}, "t", provider="o", status="blocked")
        assert history.stats("GATED")[0].blocked == 1

    def test_disabled_tracking_reads_nothing(self, isolated_db):
        promptkeep.configure(enabled=False)
        assert history.stats("ANY") == []
