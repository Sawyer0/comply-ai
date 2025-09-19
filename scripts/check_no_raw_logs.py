#!/usr/bin/env python3
"""
Pre-commit hook: block logging of raw content (PII, prompts, responses).

Scans staged Python files (or provided file paths) and fails if lines
contain suspicious logging patterns that may leak raw content.

Allow bypass with a comment marker on the line:  # ok-to-log
(Use sparingly and only for sanitized/approved messages.)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

# Suspicious tokens that should not appear in log arguments
SUSPICIOUS_TOKENS = [
    "raw_input",
    "detector_input",
    "prompt",
    "response_text",
    "raw_output",
    "original_text",
    "message_content",
    "user_content",
    "request.body",
    "request.json",
    "request.data",
    "payload",
    "body",
]

# Patterns for logger calls and print statements
LOGGER_CALL_RE = re.compile(r"\blogger\.(info|debug|warning|error|exception)\s*\(")
PRINT_CALL_RE = re.compile(r"\bprint\s*\(")

# Additional broad content leak heuristics
FSTRING_OUTPUT_RE = re.compile(r"f\"[^\"]*\{[^\}]*output[^\}]*\}")


def get_staged_files() -> List[Path]:
    # Get staged files from git diff --cached for pre-commit
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    paths = [Path(p.strip()) for p in res.stdout.splitlines() if p.strip()]
    return [p for p in paths if p.suffix == ".py"]


def scan_file(path: Path) -> List[str]:
    violations: List[str] = []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return violations

    # Skip tests by default
    if any(part == "tests" for part in path.parts):
        return violations

    for lineno, line in enumerate(content, start=1):
        if "# ok-to-log" in line:
            continue
        # Only inspect lines with logger or print calls
        if not (LOGGER_CALL_RE.search(line) or PRINT_CALL_RE.search(line) or FSTRING_OUTPUT_RE.search(line)):
            continue
        low = line.lower()
        if any(tok in low for tok in SUSPICIOUS_TOKENS) or FSTRING_OUTPUT_RE.search(line):
            violations.append(f"{path}:{lineno}: potential raw content logging: {line.strip()[:200]}")
    return violations


def main(argv: List[str]) -> int:
    file_args = [Path(a) for a in argv if a.endswith((".py", ".pyi"))]
    targets = file_args or get_staged_files()
    violations: List[str] = []
    for path in targets:
        violations.extend(scan_file(path))

    if violations:
        print("Found potential raw-content logging violations:")
        for v in violations:
            print("  ", v)
        print("\nIf this is a false positive, add '# ok-to-log' to the line after sanitizing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
