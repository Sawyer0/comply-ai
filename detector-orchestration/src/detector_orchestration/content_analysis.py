from __future__ import annotations

from .models import ContentType


def infer_content_type(_content: str) -> ContentType:
    # Very simple heuristic: treat everything as TEXT for now
    # Future: detect code blocks, images (base64), documents (markdown/pdf text)
    return ContentType.TEXT


def validate_content_size(content: str, max_length: int) -> None:
    if len(content) > max_length:
        raise ValueError("content_too_large")

