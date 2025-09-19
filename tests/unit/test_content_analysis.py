from __future__ import annotations

import pytest

from detector_orchestration.content_analysis import infer_content_type, validate_content_size
from detector_orchestration.models import ContentType


def test_infer_content_type_text():
    assert infer_content_type("hello world") == ContentType.TEXT


def test_validate_content_size():
    validate_content_size("a" * 10, 100)
    with pytest.raises(ValueError):
        validate_content_size("a" * 101, 100)

