"""
Base classes for WAF security patterns.

This module provides base classes and utilities for creating
and managing security patterns.
"""

from typing import Dict, List, Tuple
from ..interfaces import AttackType, ViolationSeverity


class SecurityPattern:
    """Base class for security patterns."""
    
    def __init__(
        self,
        name: str,
        pattern: str,
        attack_type: AttackType,
        severity: ViolationSeverity,
        description: str,
        case_sensitive: bool = False
    ):
        """
        Initialize security pattern.
        
        Args:
            name: Pattern name
            pattern: Regex pattern
            attack_type: Type of attack
            severity: Pattern severity
            description: Human-readable description
            case_sensitive: Whether pattern is case sensitive
        """
        self.name = name
        self.pattern = pattern
        self.attack_type = attack_type
        self.severity = severity
        self.description = description
        self.case_sensitive = case_sensitive
    
    def to_dict(self) -> Dict[str, str]:
        """Convert pattern to dictionary."""
        return {
            "name": self.name,
            "pattern": self.pattern,
            "attack_type": self.attack_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "case_sensitive": str(self.case_sensitive)
        }


class PatternCollection:
    """Base class for collections of security patterns."""
    
    def __init__(self, attack_type: AttackType):
        """
        Initialize pattern collection.
        
        Args:
            attack_type: Type of attack this collection handles
        """
        self.attack_type = attack_type
        self.patterns: List[SecurityPattern] = []
    
    def add_pattern(
        self,
        name: str,
        pattern: str,
        severity: ViolationSeverity,
        description: str,
        case_sensitive: bool = False
    ) -> None:
        """
        Add a pattern to the collection.
        
        Args:
            name: Pattern name
            pattern: Regex pattern
            severity: Pattern severity
            description: Human-readable description
            case_sensitive: Whether pattern is case sensitive
        """
        security_pattern = SecurityPattern(
            name=name,
            pattern=pattern,
            attack_type=self.attack_type,
            severity=severity,
            description=description,
            case_sensitive=case_sensitive
        )
        self.patterns.append(security_pattern)
    
    def get_patterns(self) -> List[SecurityPattern]:
        """Get all patterns in the collection."""
        return self.patterns.copy()
    
    def get_pattern_by_name(self, name: str) -> SecurityPattern:
        """Get pattern by name."""
        for pattern in self.patterns:
            if pattern.name == name:
                return pattern
        raise ValueError(f"Pattern '{name}' not found")
    
    def remove_pattern(self, name: str) -> bool:
        """
        Remove pattern by name.
        
        Args:
            name: Pattern name to remove
            
        Returns:
            True if pattern was removed, False if not found
        """
        for i, pattern in enumerate(self.patterns):
            if pattern.name == name:
                del self.patterns[i]
                return True
        return False
    
    def get_patterns_by_severity(self, severity: ViolationSeverity) -> List[SecurityPattern]:
        """Get patterns by severity level."""
        return [p for p in self.patterns if p.severity == severity]
    
    def to_dict(self) -> Dict[str, any]:
        """Convert collection to dictionary."""
        return {
            "attack_type": self.attack_type.value,
            "pattern_count": len(self.patterns),
            "patterns": [p.to_dict() for p in self.patterns]
        }
