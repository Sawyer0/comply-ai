"""Multi-layer input sanitization functionality following SRP.

This module provides ONLY input sanitization - cleaning and validating input data.
Single Responsibility: Sanitize and validate input data for security.
"""

import html
import re
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import ValidationError

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Multi-layer input sanitization for security.

    Single Responsibility: Sanitize input data to prevent security vulnerabilities.
    Does NOT handle: attack detection, logging attacks, blocking requests.
    """

    def __init__(
        self,
        max_length: int = 10000,
        allow_html: bool = False,
        strict_mode: bool = True,
    ):
        """Initialize input sanitizer.

        Args:
            max_length: Maximum allowed input length
            allow_html: Whether to allow HTML content
            strict_mode: Whether to use strict sanitization
        """
        self.max_length = max_length
        self.allow_html = allow_html
        self.strict_mode = strict_mode

        # Dangerous characters to remove/escape
        self._dangerous_chars = {
            "\x00": "",  # Null byte
            "\x08": "",  # Backspace
            "\x0b": "",  # Vertical tab
            "\x0c": "",  # Form feed
            "\x0e": "",  # Shift out
            "\x0f": "",  # Shift in
            "\x7f": "",  # Delete
        }

        # SQL injection patterns to neutralize
        self._sql_patterns = [
            (r"(\bUNION\s+SELECT\b)", r"UNION_SELECT"),
            (r"(\bDROP\s+TABLE\b)", r"DROP_TABLE"),
            (r"(\bDELETE\s+FROM\b)", r"DELETE_FROM"),
            (r"(\bINSERT\s+INTO\b)", r"INSERT_INTO"),
            (r"(\bUPDATE\s+SET\b)", r"UPDATE_SET"),
            (r"(--)", r"COMMENT"),
            (r"(/\*|\*/)", r"COMMENT"),
        ]

        # XSS patterns to neutralize
        self._xss_patterns = [
            (r"(<script[^>]*>.*?</script>)", r"[SCRIPT_REMOVED]"),
            (r"(<iframe[^>]*>.*?</iframe>)", r"[IFRAME_REMOVED]"),
            (r"(javascript\s*:)", r"javascript_"),
            (r"(on\w+\s*=)", r"on_event_"),
        ]

    def sanitize(self, data: Any) -> Any:
        """Sanitize input data recursively.

        Args:
            data: Data to sanitize (can be string, dict, list, or other types)

        Returns:
            Sanitized data
        """
        correlation_id = get_correlation_id()

        try:
            if isinstance(data, str):
                return self._sanitize_string(data)
            if isinstance(data, dict):
                return self._sanitize_dict(data)
            if isinstance(data, list):
                return self._sanitize_list(data)
            if isinstance(data, (int, float, bool, type(None))):
                return data

            logger.debug(
                "Converting unknown type %s to string for sanitization",
                type(data).__name__,
                extra={"correlation_id": correlation_id},
            )
            return self._sanitize_string(str(data))

        except (ValueError, TypeError) as exc:
            logger.error(
                "Error during sanitization: %s",
                exc,
                extra={
                    "correlation_id": correlation_id,
                    "data_type": type(data).__name__,
                    "error": str(exc),
                },
            )
            raise ValidationError(
                f"Input sanitization failed: {exc}", correlation_id=correlation_id
            ) from exc

    def _sanitize_string(self, text: str) -> str:
        """Sanitize a string input.

        Args:
            text: String to sanitize

        Returns:
            Sanitized string
        """
        if not isinstance(text, str):
            text = str(text)

        # Check length limit
        if len(text) > self.max_length:
            logger.warning(
                "Input exceeds maximum length (%d > %d), truncating",
                len(text),
                self.max_length,
                extra={"correlation_id": get_correlation_id()},
            )
            text = text[: self.max_length]

        # Remove dangerous control characters
        for char, replacement in self._dangerous_chars.items():
            text = text.replace(char, replacement)

        # URL decode (handle double encoding)
        text = self._safe_url_decode(text)

        # HTML escape if not allowing HTML
        if not self.allow_html:
            text = html.escape(text, quote=True)

        # Neutralize SQL injection patterns
        if self.strict_mode:
            text = self._neutralize_sql_patterns(text)

        # Neutralize XSS patterns
        if self.strict_mode:
            text = self._neutralize_xss_patterns(text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary data.

        Args:
            data: Dictionary to sanitize

        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        for key, value in data.items():
            # Sanitize both key and value
            sanitized_key = self._sanitize_string(str(key))
            sanitized_value = self.sanitize(value)
            sanitized[sanitized_key] = sanitized_value
        return sanitized

    def _sanitize_list(self, data: List[Any]) -> List[Any]:
        """Sanitize list data.

        Args:
            data: List to sanitize

        Returns:
            Sanitized list
        """
        return [self.sanitize(item) for item in data]

    def _safe_url_decode(self, text: str) -> str:
        """Safely URL decode text, handling multiple encoding layers.

        Args:
            text: Text to decode

        Returns:
            Decoded text
        """
        try:
            # Decode up to 3 layers to handle double/triple encoding
            for _ in range(3):
                decoded = unquote(text)
                if decoded == text:
                    break  # No more decoding needed
                text = decoded
            return text
        except (ValueError, UnicodeDecodeError) as exc:
            logger.debug(
                "URL decoding failed, using original text: %s",
                exc,
                extra={"correlation_id": get_correlation_id()},
            )
            return text

    def _neutralize_sql_patterns(self, text: str) -> str:
        """Neutralize SQL injection patterns.

        Args:
            text: Text to process

        Returns:
            Text with SQL patterns neutralized
        """
        for pattern, replacement in self._sql_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _neutralize_xss_patterns(self, text: str) -> str:
        """Neutralize XSS patterns.

        Args:
            text: Text to process

        Returns:
            Text with XSS patterns neutralized
        """
        for pattern, replacement in self._xss_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.DOTALL)
        return text

    def validate_input_size(self, data: Any) -> bool:
        """Validate that input data is within acceptable size limits.

        Args:
            data: Data to validate

        Returns:
            True if size is acceptable, False otherwise
        """
        try:
            if isinstance(data, str):
                return len(data) <= self.max_length
            if isinstance(data, (dict, list)):
                estimated_size = len(str(data))
                return estimated_size <= self.max_length * 2  # Allow some overhead
            return True  # Other types are typically small
        except (TypeError, ValueError):
            return False

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename for safe storage.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        if not filename:
            return "unnamed_file"

        # Remove path traversal attempts
        filename = filename.replace("..", "").replace("/", "").replace("\\", "")

        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', "_", filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
            max_name_len = 250 - len(ext)
            filename = name[:max_name_len] + ("." + ext if ext else "")

        # Ensure it's not empty after sanitization
        if not filename.strip("._"):
            filename = "sanitized_file"

        return filename

    def sanitize_email(self, email: str) -> Optional[str]:
        """Sanitize and validate an email address.

        Args:
            email: Email address to sanitize

        Returns:
            Sanitized email if valid, None otherwise
        """
        if not email or not isinstance(email, str):
            return None

        # Basic sanitization
        email = email.strip().lower()

        # Remove dangerous characters
        email = re.sub(r'[<>"\'\x00-\x1f]', "", email)

        # Basic email validation
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if re.match(email_pattern, email):
            return email

        return None

    def sanitize_phone(self, phone: str) -> Optional[str]:
        """Sanitize a phone number.

        Args:
            phone: Phone number to sanitize

        Returns:
            Sanitized phone number if valid, None otherwise
        """
        if not phone or not isinstance(phone, str):
            return None

        # Remove all non-digit characters except + at the beginning
        phone = re.sub(r"[^\d+]", "", phone)

        # Ensure + is only at the beginning
        if "+" in phone:
            parts = phone.split("+")
            phone = "+" + "".join(parts[1:])

        # Basic validation (must have at least 7 digits)
        digits_only = re.sub(r"[^\d]", "", phone)
        if len(digits_only) >= 7:
            return phone

        return None

    def get_sanitization_report(self, original: str, sanitized: str) -> Dict[str, Any]:
        """Generate a report of sanitization changes.

        Args:
            original: Original input
            sanitized: Sanitized output

        Returns:
            Dictionary with sanitization report
        """
        return {
            "original_length": len(original),
            "sanitized_length": len(sanitized),
            "length_changed": len(original) != len(sanitized),
            "content_changed": original != sanitized,
            "truncated": len(original) > self.max_length,
            "html_escaped": not self.allow_html
            and ("<" in original or ">" in original),
            "patterns_neutralized": self.strict_mode
            and (
                any(pattern in original.lower() for pattern, _ in self._sql_patterns)
                or any(pattern in original.lower() for pattern, _ in self._xss_patterns)
            ),
        }


# Export only the input sanitization functionality
__all__ = [
    "InputSanitizer",
]
