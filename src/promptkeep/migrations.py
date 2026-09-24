"""Forward-only schema upgrades, applied when a database file is first opened.

There are no migration files. A promptkeep database is a file the user owns,
possibly written by an older release, so every version of the library must
be able to bring any older file up to date on open. The DB carries its
schema number in ``PRAGMA user_version``; ``migrate()`` runs the steps for
everything below ``SCHEMA_VERSION``. To change the schema: bump the
constant, add a step here using ``playhouse.migrate`` operations, and never
edit an existing step — files in the wild already ran it.
"""

from __future__ import annotations

import peewee as pw
import playhouse.migrate as pm

from .models import MODELS, CheckRecord, ConversationRecord, PromptVersionRecord, new_run_key
from .rendering import template_hash

SCHEMA_VERSION = 8


def migrate(database: pw.SqliteDatabase) -> None:
    """Apply forward-only schema steps until the DB reaches SCHEMA_VERSION.

    Steps compose regardless of which version a real DB starts at (e.g. 1 -> 3
    runs both the v2 and v3 steps) — each guard tests the version the DB began
    at. Every step runs in its own transaction that also stamps the new
    `user_version`, so a step and its version bump commit together: an
    interrupted step rolls back whole and re-runs from the same version rather
    than replaying half of it into a "duplicate column" error. A brand-new DB
    is created straight from `MODELS` at the latest shape, so it skips the
    legacy fix-up steps meant for pre-existing rows.
    """
    (user_version,) = database.execute_sql("PRAGMA user_version").fetchone()

    # A fresh file: build every table at the current shape and stamp it.
    if user_version < 1:
        with database.atomic():
            database.create_tables(MODELS, safe=True)
            database.execute_sql(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return

    # v2: template_hash became a hash of the *normalized* template (variable
    # names canonicalized). Recompute stored hashes so old rows keep deduping
    # correctly against new registrations.
    if user_version < 2:
        with database.atomic():
            for row in PromptVersionRecord.select():
                new_hash = template_hash(row.template)
                if new_hash != row.template_hash:
                    try:
                        PromptVersionRecord.update(template_hash=new_hash).where(
                            PromptVersionRecord.id == row.id
                        ).execute()
                    except pw.IntegrityError:
                        # Two old versions differing only in variable names now
                        # collide; keep the older row normalized, leave this one.
                        pass
            database.execute_sql("PRAGMA user_version = 2")

    # v3: conversations. runs.version_id/rendered_text drop NOT NULL because a
    # conversation turn with no wrapped Prompt still gets a row (it has no
    # version to attach to), and the new columns link a run to its
    # conversation and turn position.
    if user_version < 3:
        with database.atomic():
            database.create_tables([ConversationRecord], safe=True)
            migrator = pm.SqliteMigrator(database)
            pm.migrate(
                migrator.add_column("runs", "conversation_id", pw.IntegerField(null=True)),
                migrator.add_column("runs", "turn_index", pw.IntegerField(null=True)),
                migrator.add_column("runs", "input_text", pw.TextField(null=True)),
                migrator.drop_not_null("runs", "version_id"),
                migrator.drop_not_null("runs", "rendered_text"),
            )
            database.execute_sql("PRAGMA user_version = 3")

    # v4: the checks table — one verdict per (run, check).
    if user_version < 4:
        with database.atomic():
            database.create_tables([CheckRecord], safe=True)
            database.execute_sql("PRAGMA user_version = 4")

    # v5: runs gain a caller-minted identity (run_key) so a run can be
    # referenced before its row lands — checked calls go through the
    # background writer like everything else, and late verdicts find their
    # run by key. Plus the rewrite audit trail: the turn as the caller passed
    # it (runs.original_input_text) and what the rewriting check produced
    # (checks.rewritten).
    if user_version < 5:
        with database.atomic():
            migrator = pm.SqliteMigrator(database)
            operations = [
                migrator.add_column("runs", "run_key", pw.TextField(null=True)),
                migrator.add_column("runs", "original_input_text", pw.TextField(null=True)),
            ]
            # A DB that started below v4 just had its checks table created
            # from the current model (rewritten included) in the step above;
            # one that started at v4 has the older shape and needs the column.
            if not _has_column(database, "checks", "rewritten"):
                operations.append(
                    migrator.add_column("checks", "rewritten", pw.TextField(null=True))
                )
            pm.migrate(*operations)

            # Backfill so every pre-existing run has a key: the column is the
            # run's identity from here on, and readers rely on it being set.
            for (row_id,) in database.execute_sql("SELECT id FROM runs").fetchall():
                database.execute_sql(
                    "UPDATE runs SET run_key = ? WHERE id = ?", (new_run_key(), row_id)
                )
            pm.migrate(migrator.add_index("runs", ("run_key",), True))
            database.execute_sql("PRAGMA user_version = 5")

    # v6: runs.cost_usd — the provider-reported cost of the call. Old rows stay
    # NULL: nothing recorded what they cost, and a guess would be worse.
    if user_version < 6:
        with database.atomic():
            migrator = pm.SqliteMigrator(database)
            pm.migrate(migrator.add_column("runs", "cost_usd", pw.FloatField(null=True)))
            database.execute_sql("PRAGMA user_version = 6")

    # v7: an index on runs.response_id. A call that names its predecessor
    # (the Responses API's previous_response_id) is filed in that run's
    # conversation, which is a lookup by response id on the request path.
    if user_version < 7:
        with database.atomic():
            migrator = pm.SqliteMigrator(database)
            pm.migrate(migrator.add_index("runs", ("response_id",), False))
            database.execute_sql("PRAGMA user_version = 7")

    # v8: runs.parent_run_key — the run a turn branched from, which makes a
    # conversation a tree. Old rows stay NULL: every one of them followed the
    # turn before it, which is exactly what NULL means.
    if user_version < 8:
        with database.atomic():
            migrator = pm.SqliteMigrator(database)
            pm.migrate(
                migrator.add_column("runs", "parent_run_key", pw.TextField(null=True)),
                migrator.add_index("runs", ("parent_run_key",), False),
            )
            database.execute_sql("PRAGMA user_version = 8")


def _has_column(database: pw.SqliteDatabase, table: str, column: str) -> bool:
    """Whether ``table`` already has ``column`` — for migration steps whose
    starting shape depends on which earlier steps ran in the same pass."""
    return any(col.name == column for col in database.get_columns(table))
