"""Import any supported saved agent session without credentials or API calls."""

import argparse
import json

from promptkeep import configure, history
from promptkeep.importers import import_trace


def main():
    """Detect a saved source and display its trace and separate usage summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--source", default="auto")
    parser.add_argument("--db", default="agent-history.promptkeep.db")
    args = parser.parse_args()
    configure(db_path=args.db)
    report = import_trace(args.path, source=args.source)
    trace = history.trace(report.trace_key)
    imported = history.imported_session(report.trace_key)
    print(f"{report.outcome}: {report.trace_key}; {trace.model_calls} observed model calls")
    print(json.dumps(imported.summary["total_usage"]))
    print(f"Source diagnostics: {len(report.diagnostics)}")


if __name__ == "__main__":
    main()
