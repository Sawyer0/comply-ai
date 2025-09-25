"""
Base pattern collection for WAF security patterns.

This module provides the base class for organizing security patterns
by attack type with consistent severity and description handling.
"""

import logging
from typing import Dict, List, Optional

from ..interfaces import AttackType, ViolationSeverity, WAFRule

logger = logging.getLogger(__name__)


class PatternCollection:
    """Base class for organizing security patterns by attack type."""

    def __init__(self, attack_type: AttackType):
        """
        Initialize pattern collection.

        Args:
            attack_type: Type of attack this collection detects
        """
        self.attack_type = attack_type
        self.patterns: Dict[str, WAFRule] = {}

    def add_pattern(
        self,
        name: str,
        pattern: str,
        severity: ViolationSeverity,
        description: str = "",
        case_sensitive: bool = False,
    ) -> None:
        """
        Add a pattern to the collection.

        Args:
            name: Pattern name
            pattern: Regex pattern
            severity: Pattern severity
            description: Human-readable description
            case_sensitive: Whether pattern matching is case sensitive
        """
        try:
            rule = WAFRule(
                name=name,
                pattern=pattern,
                attack_type=self.attack_type,
                severity=severity,
                description=description,
                case_sensitive=case_sensitive,
            )
            self.patterns[name] = rule
            logger.debug("Added pattern", name=name, attack_type=self.attack_type.value)

        except Exception as e:
            logger.error("Failed to add pattern", name=name, error=str(e))

    def get_pattern(self, name: str) -> Optional[WAFRule]:
        """Get a pattern by name."""
        return self.patterns.get(name)

    def get_all_patterns(self) -> List[WAFRule]:
        """Get all patterns in this collection."""
        return list(self.patterns.values())

    def get_patterns_by_severity(self, severity: ViolationSeverity) -> List[WAFRule]:
        """Get patterns filtered by severity."""
        return [rule for rule in self.patterns.values() if rule.severity == severity]

    def remove_pattern(self, name: str) -> bool:
        """
        Remove a pattern by name.

        Returns:
            True if pattern was removed, False if not found
        """
        if name in self.patterns:
            del self.patterns[name]
            logger.debug("Removed pattern", name=name)
            return True
        return False

    def get_pattern_count(self) -> int:
        """Get total number of patterns in this collection."""
        return len(self.patterns)

    def get_severity_distribution(self) -> Dict[str, int]:
        """Get distribution of patterns by severity."""
        distribution = {}
        for rule in self.patterns.values():
            severity = rule.severity.value
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution

    def validate_patterns(self) -> List[str]:
        """
        Validate all patterns in the collection.

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        for name, rule in self.patterns.items():
            try:
                # Test pattern compilation
                rule.pattern.search("test")
            except Exception as e:
                errors.append(f"Pattern '{name}' is invalid: {str(e)}")
        return errors
