"""
WAF Rule implementation.

This module provides the concrete implementation of WAF rules.
"""

import re
from typing import List, Tuple
from .interfaces import IWAFRule, AttackType, ViolationSeverity


class WAFRule(IWAFRule):
    """
    Concrete implementation of WAF rule.
    
    This class provides a concrete implementation of WAF rules with
    pattern matching and violation detection.
    """
    
    def __init__(
        self,
        name: str,
        pattern: str,
        attack_type: AttackType,
        severity: ViolationSeverity,
        description: str = "",
        action: str = "block"
    ):
        """
        Initialize WAF rule.
        
        Args:
            name: Rule name
            pattern: Regex pattern to match
            attack_type: Type of attack this rule detects
            severity: Severity level
            description: Rule description
            action: Action to take when rule matches
        """
        self._name = name
        self._pattern = pattern
        self._attack_type = attack_type
        self._severity = severity
        self._description = description or f"Rule to detect {attack_type.value}"
        self._action = action
        
        # Compile regex pattern
        try:
            self._compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    
    @property
    def name(self) -> str:
        """Rule name."""
        return self._name
    
    @property
    def pattern(self) -> str:
        """Rule pattern."""
        return self._pattern
    
    @property
    def attack_type(self) -> AttackType:
        """Attack type this rule detects."""
        return self._attack_type
    
    @property
    def severity(self) -> ViolationSeverity:
        """Rule severity."""
        return self._severity
    
    @property
    def description(self) -> str:
        """Rule description."""
        return self._description
    
    @property
    def action(self) -> str:
        """Action to take when rule matches."""
        return self._action
    
    def matches(self, text: str) -> bool:
        """
        Check if rule matches the given text.
        
        Args:
            text: Text to check
            
        Returns:
            True if rule matches, False otherwise
        """
        return bool(self._compiled_pattern.search(text))
    
    def get_matches(self, text: str) -> List[Tuple[str, str]]:
        """
        Get all matches in the text with their positions.
        
        Args:
            text: Text to search
            
        Returns:
            List of tuples containing (match, position)
        """
        matches = []
        for match in self._compiled_pattern.finditer(text):
            matches.append((match.group(), f"{match.start()}-{match.end()}"))
        return matches
