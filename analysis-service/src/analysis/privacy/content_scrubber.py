"""
Content Scrubber

This module provides content scrubbing functionality to remove sensitive information.
Follows SRP by focusing solely on content sanitization and scrubbing.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Pattern
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScrubRule:
    """Rule for scrubbing sensitive content."""

    name: str
    pattern: Pattern[str]
    replacement: str
    description: str
    enabled: bool = True


class ContentScrubber:
    """
    Scrubs sensitive content from text and data structures.

    Single responsibility: Remove or mask sensitive information from content.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scrubbing_config = config.get("content_scrubbing", {})

        # Configuration
        self.enabled = self.scrubbing_config.get("enabled", True)
        self.aggressive_mode = self.scrubbing_config.get("aggressive_mode", False)
        self.preserve_structure = self.scrubbing_config.get("preserve_structure", True)

        # Scrubbing rules
        self.scrub_rules: List[ScrubRule] = []
        self._initialize_scrub_rules()

        # Statistics
        self.scrub_stats = {
            "total_scrubs": 0,
            "content_scrubbed": 0,
            "rules_triggered": {},
            "bytes_scrubbed": 0,
        }

        logger.info(
            "Content Scrubber initialized",
            enabled=self.enabled,
            rules=len(self.scrub_rules),
            aggressive_mode=self.aggressive_mode,
        )

    def scrub_text(self, text: str) -> str:
        """
        Scrub sensitive content from text.

        Args:
            text: Text to scrub

        Returns:
            Scrubbed text with sensitive content removed/masked
        """
        if not self.enabled or not text:
            return text

        try:
            original_length = len(text)
            scrubbed_text = text
            rules_applied = []

            # Apply scrubbing rules
            for rule in self.scrub_rules:
                if rule.enabled:
                    matches = rule.pattern.findall(scrubbed_text)
                    if matches:
                        scrubbed_text = rule.pattern.sub(
                            rule.replacement, scrubbed_text
                        )
                        rules_applied.append(rule.name)

                        # Update statistics
                        self.scrub_stats["rules_triggered"][rule.name] = (
                            self.scrub_stats["rules_triggered"].get(rule.name, 0)
                            + len(matches)
                        )

            # Update statistics
            self.scrub_stats["total_scrubs"] += 1
            if rules_applied:
                self.scrub_stats["content_scrubbed"] += 1
                self.scrub_stats["bytes_scrubbed"] += original_length - len(
                    scrubbed_text
                )

            if rules_applied:
                logger.debug(
                    "Content scrubbed",
                    rules_applied=rules_applied,
                    original_length=original_length,
                    scrubbed_length=len(scrubbed_text),
                )

            return scrubbed_text

        except Exception as e:
            logger.error("Content scrubbing failed", error=str(e))
            # Return original text if scrubbing fails
            return text

    def scrub_dict(
        self, data: Dict[str, Any], scrub_keys: bool = True, scrub_values: bool = True
    ) -> Dict[str, Any]:
        """
        Scrub sensitive content from dictionary.

        Args:
            data: Dictionary to scrub
            scrub_keys: Whether to scrub dictionary keys
            scrub_values: Whether to scrub dictionary values

        Returns:
            Scrubbed dictionary
        """
        if not self.enabled or not data:
            return data

        try:
            scrubbed_data = {}

            for key, value in data.items():
                # Scrub key if requested
                scrubbed_key = (
                    self.scrub_text(key) if scrub_keys and isinstance(key, str) else key
                )

                # Scrub value based on type
                if isinstance(value, str) and scrub_values:
                    scrubbed_value = self.scrub_text(value)
                elif isinstance(value, dict):
                    scrubbed_value = self.scrub_dict(value, scrub_keys, scrub_values)
                elif isinstance(value, list):
                    scrubbed_value = self.scrub_list(value, scrub_values)
                else:
                    scrubbed_value = value

                scrubbed_data[scrubbed_key] = scrubbed_value

            return scrubbed_data

        except Exception as e:
            logger.error("Dictionary scrubbing failed", error=str(e))
            return data

    def scrub_list(self, data: List[Any], scrub_items: bool = True) -> List[Any]:
        """
        Scrub sensitive content from list.

        Args:
            data: List to scrub
            scrub_items: Whether to scrub list items

        Returns:
            Scrubbed list
        """
        if not self.enabled or not data or not scrub_items:
            return data

        try:
            scrubbed_list = []

            for item in data:
                if isinstance(item, str):
                    scrubbed_item = self.scrub_text(item)
                elif isinstance(item, dict):
                    scrubbed_item = self.scrub_dict(item)
                elif isinstance(item, list):
                    scrubbed_item = self.scrub_list(item)
                else:
                    scrubbed_item = item

                scrubbed_list.append(scrubbed_item)

            return scrubbed_list

        except Exception as e:
            logger.error("List scrubbing failed", error=str(e))
            return data

    def scrub_log_message(
        self, message: str, extra_data: Optional[Dict[str, Any]] = None
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Scrub log message and extra data for safe logging.

        Args:
            message: Log message to scrub
            extra_data: Extra logging data to scrub

        Returns:
            Tuple of (scrubbed_message, scrubbed_extra_data)
        """
        try:
            scrubbed_message = self.scrub_text(message)
            scrubbed_extra = self.scrub_dict(extra_data) if extra_data else None

            return scrubbed_message, scrubbed_extra

        except Exception as e:
            logger.error("Log message scrubbing failed", error=str(e))
            return message, extra_data

    def add_scrub_rule(
        self, name: str, pattern: str, replacement: str, description: str = ""
    ):
        """
        Add custom scrubbing rule.

        Args:
            name: Rule name
            pattern: Regex pattern to match
            replacement: Replacement text
            description: Rule description
        """
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

            rule = ScrubRule(
                name=name,
                pattern=compiled_pattern,
                replacement=replacement,
                description=description,
            )

            self.scrub_rules.append(rule)
            self.scrub_stats["rules_triggered"][name] = 0

            logger.info("Scrub rule added", name=name, description=description)

        except Exception as e:
            logger.error("Failed to add scrub rule", name=name, error=str(e))

    def remove_scrub_rule(self, name: str) -> bool:
        """
        Remove scrubbing rule.

        Args:
            name: Rule name to remove

        Returns:
            True if rule was removed, False otherwise
        """
        try:
            original_count = len(self.scrub_rules)
            self.scrub_rules = [rule for rule in self.scrub_rules if rule.name != name]

            if len(self.scrub_rules) < original_count:
                if name in self.scrub_stats["rules_triggered"]:
                    del self.scrub_stats["rules_triggered"][name]

                logger.info("Scrub rule removed", name=name)
                return True
            else:
                logger.warning("Scrub rule not found", name=name)
                return False

        except Exception as e:
            logger.error("Failed to remove scrub rule", name=name, error=str(e))
            return False

    def enable_rule(self, name: str) -> bool:
        """Enable a scrubbing rule."""
        for rule in self.scrub_rules:
            if rule.name == name:
                rule.enabled = True
                logger.info("Scrub rule enabled", name=name)
                return True

        logger.warning("Scrub rule not found", name=name)
        return False

    def disable_rule(self, name: str) -> bool:
        """Disable a scrubbing rule."""
        for rule in self.scrub_rules:
            if rule.name == name:
                rule.enabled = False
                logger.info("Scrub rule disabled", name=name)
                return True

        logger.warning("Scrub rule not found", name=name)
        return False

    def get_scrub_statistics(self) -> Dict[str, Any]:
        """Get content scrubbing statistics."""
        scrub_rate = self.scrub_stats["content_scrubbed"] / max(
            1, self.scrub_stats["total_scrubs"]
        )

        return {
            "enabled": self.enabled,
            "total_scrubs": self.scrub_stats["total_scrubs"],
            "content_scrubbed": self.scrub_stats["content_scrubbed"],
            "scrub_rate": scrub_rate,
            "bytes_scrubbed": self.scrub_stats["bytes_scrubbed"],
            "rules_triggered": self.scrub_stats["rules_triggered"].copy(),
            "active_rules": len([r for r in self.scrub_rules if r.enabled]),
            "total_rules": len(self.scrub_rules),
            "configuration": {
                "aggressive_mode": self.aggressive_mode,
                "preserve_structure": self.preserve_structure,
            },
        }

    def _initialize_scrub_rules(self):
        """Initialize default scrubbing rules."""
        default_rules = [
            # Email addresses
            {
                "name": "email_addresses",
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "replacement": "[EMAIL_REDACTED]",
                "description": "Scrub email addresses",
            },
            # Phone numbers (US format)
            {
                "name": "phone_numbers",
                "pattern": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
                "replacement": "[PHONE_REDACTED]",
                "description": "Scrub US phone numbers",
            },
            # Social Security Numbers
            {
                "name": "ssn",
                "pattern": r"\b\d{3}-?\d{2}-?\d{4}\b",
                "replacement": "[SSN_REDACTED]",
                "description": "Scrub Social Security Numbers",
            },
            # Credit card numbers (basic pattern)
            {
                "name": "credit_cards",
                "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                "replacement": "[CARD_REDACTED]",
                "description": "Scrub credit card numbers",
            },
            # IP addresses
            {
                "name": "ip_addresses",
                "pattern": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                "replacement": "[IP_REDACTED]",
                "description": "Scrub IP addresses",
            },
            # API keys and tokens (common patterns)
            {
                "name": "api_keys",
                "pattern": r'\b(?:api[_-]?key|token|secret)["\s]*[:=]["\s]*([A-Za-z0-9+/=]{20,})\b',
                "replacement": r"\1[API_KEY_REDACTED]",
                "description": "Scrub API keys and tokens",
            },
            # Passwords in URLs or config
            {
                "name": "passwords",
                "pattern": r'\b(?:password|pwd)["\s]*[:=]["\s]*([^\s"]+)',
                "replacement": r"password=[PASSWORD_REDACTED]",
                "description": "Scrub passwords",
            },
        ]

        # Add custom rules from configuration
        custom_rules = self.scrubbing_config.get("custom_rules", [])
        all_rules = default_rules + custom_rules

        for rule_config in all_rules:
            try:
                pattern = re.compile(
                    rule_config["pattern"], re.IGNORECASE | re.MULTILINE
                )

                rule = ScrubRule(
                    name=rule_config["name"],
                    pattern=pattern,
                    replacement=rule_config["replacement"],
                    description=rule_config.get("description", ""),
                    enabled=rule_config.get("enabled", True),
                )

                self.scrub_rules.append(rule)
                self.scrub_stats["rules_triggered"][rule.name] = 0

            except Exception as e:
                logger.error(
                    "Failed to initialize scrub rule",
                    rule=rule_config.get("name", "unknown"),
                    error=str(e),
                )

        logger.info("Initialized scrub rules", count=len(self.scrub_rules))
