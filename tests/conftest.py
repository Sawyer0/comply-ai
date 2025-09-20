from __future__ import annotations

import json
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def _tests_root() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def conflict_scenarios(_tests_root: Path) -> dict:
    fp = _tests_root / "fixtures" / "conflict_scenarios.json"
    return json.loads(fp.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def opa_decision_fixtures(_tests_root: Path) -> dict:
    fp = _tests_root / "fixtures" / "opa_decisions.json"
    return json.loads(fp.read_text(encoding="utf-8"))
