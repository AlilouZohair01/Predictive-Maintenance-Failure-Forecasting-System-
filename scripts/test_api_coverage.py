"""Standard-library executable-line coverage, including API worker threads.

This is line coverage, not branch coverage. It avoids external coverage tooling.
The source-path filter avoids trace's module-basename cache collisions (for
example our api/main.py versus installed packages also named main.py).
"""
import json
import os
from pathlib import Path
import sys
import threading
import trace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/predictive_maintenance"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    tracer = trace.Trace(count=True, trace=False)
    prefix = os.path.normcase(str(SOURCE)) + os.sep

    def only_project(frame, event, arg):
        if event == "call" and os.path.normcase(frame.f_code.co_filename).startswith(prefix):
            return tracer.localtrace
        return None

    tracer.globaltrace = only_project

    def run():
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        return unittest.TextTestRunner(verbosity=2).run(suite)

    previous = threading.gettrace()
    threading.settrace(only_project)
    try:
        outcome = tracer.runfunc(run)
    finally:
        threading.settrace(previous)
    results = tracer.results()
    coverdir = ROOT / "reports/api_line_coverage"
    results.write_results(show_missing=True, summary=True, coverdir=str(coverdir))
    rows = []
    for path in sorted(SOURCE.rglob("*.py")):
        # Same executable-line definition used by Python 3.11 trace's own report.
        executable = trace._find_executable_linenos(str(path))
        hit = sum(results.counts.get((str(path), line), 0) > 0 for line in executable)
        if executable:
            rows.append({"file": path.relative_to(ROOT).as_posix(), "executable_lines": len(executable),
                         "covered_lines": hit, "percent": round(hit / len(executable) * 100, 2)})
    total, covered = sum(r["executable_lines"] for r in rows), sum(r["covered_lines"] for r in rows)
    report = {"method": "Python 3.11 stdlib trace; executable line coverage, not branch coverage",
              "tests_run": outcome.testsRun, "successful": outcome.wasSuccessful(), "files": rows,
              "executable_lines": total, "covered_lines": covered, "percent": round(covered / total * 100, 2)}
    (ROOT / "reports/api_coverage_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}))
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
