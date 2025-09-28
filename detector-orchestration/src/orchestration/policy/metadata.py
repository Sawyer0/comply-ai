"""Policy metadata extraction and analysis.

Extracted from PolicyLoader to handle metadata extraction,
policy structure analysis, and statistics generation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List
from functools import lru_cache

from .parser import classify_rule_type

logger = logging.getLogger(__name__)

__all__ = [
    "extract_policy_metadata",
    "parse_policy_structure", 
    "generate_policy_statistics",
    "PolicyMetadata",
]


class PolicyMetadata:
    """Container for policy metadata and statistics."""
    
    def __init__(self):
        self.loaded_policies: Dict[str, str] = {}
        self.policy_metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_policy(self, name: str, content: str, file_path: Path | None = None):
        """Add a policy and extract its metadata."""
        self.loaded_policies[name] = content
        if file_path:
            self.policy_metadata[name] = extract_policy_metadata(content, file_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all loaded policies."""
        return generate_policy_statistics(self.loaded_policies, self.policy_metadata)


def extract_policy_metadata(content: str, policy_file: Path) -> Dict[str, Any]:
    """Extract metadata from policy content.

    Args:
        content: Policy file content
        policy_file: Path to policy file

    Returns:
        Dictionary containing policy metadata
    """
    metadata = {
        "file_path": str(policy_file),
        "file_size": len(content),
        "package_name": None,
        "rules_count": 0,
        "imports": [],
    }

    try:
        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            # Extract package name
            if line.startswith("package "):
                metadata["package_name"] = line.replace("package ", "").strip()

            # Count rules (simple heuristic)
            if (
                line.endswith(" if {")
                or line.endswith(" := ")
                or line == "allow"
                or line == "deny"
            ):
                metadata["rules_count"] += 1

            # Extract imports
            if line.startswith("import "):
                import_statement = line.replace("import ", "").strip()
                metadata["imports"].append(import_statement)

    except (IndexError, AttributeError) as e:
        logger.warning("Failed to extract policy metadata", extra={"error": str(e)})

    return metadata


@lru_cache(maxsize=100)
def parse_policy_structure(policy_content: str) -> Dict[str, Any]:
    """Parse and cache policy structure analysis.

    Args:
        policy_content: Policy content to analyze

    Returns:
        Dictionary containing policy structure information
    """
    structure = {
        "package_name": None,
        "rules": [],
        "imports": [],
        "rule_count": 0,
        "complexity_score": 0,
    }

    try:
        lines = policy_content.split("\n")
        complexity = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Extract package name
            if line.startswith("package "):
                structure["package_name"] = line.replace("package ", "").strip()

            # Extract imports
            if line.startswith("import "):
                import_statement = line.replace("import ", "").strip()
                structure["imports"].append(import_statement)

            # Extract rules and calculate complexity
            if " if {" in line or line.endswith(" := ") or line in ["allow", "deny"]:
                rule_name = line.split(" ")[0] if " " in line else line
                structure["rules"].append(
                    {
                        "name": rule_name,
                        "line": line_num,
                        "type": classify_rule_type(line),
                    }
                )
                structure["rule_count"] += 1

            # Add complexity for control structures
            complexity += line.count("if ") + line.count("else") + line.count("count(")

        structure["complexity_score"] = complexity

    except (AttributeError, IndexError, ValueError) as e:
        logger.warning("Failed to parse policy structure", extra={"error": str(e)})

    return structure


def generate_policy_statistics(
    loaded_policies: Dict[str, str], 
    policy_metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate comprehensive statistics about loaded policies.

    Args:
        loaded_policies: Dictionary mapping policy names to their content
        policy_metadata: Dictionary mapping policy names to their metadata

    Returns:
        Dictionary containing policy loading statistics
    """
    total_rules = 0
    total_complexity = 0
    package_names = []

    for content in loaded_policies.values():
        structure = parse_policy_structure(content)
        total_rules += structure.get("rule_count", 0)
        total_complexity += structure.get("complexity_score", 0)

        if structure.get("package_name"):
            package_names.append(structure["package_name"])

    return {
        "total_policies": len(loaded_policies),
        "policy_names": list(loaded_policies.keys()),
        "total_content_size": sum(len(content) for content in loaded_policies.values()),
        "total_rules": total_rules,
        "total_complexity_score": total_complexity,
        "average_complexity": (
            total_complexity / len(loaded_policies) if loaded_policies else 0
        ),
        "package_names": package_names,
        "metadata_available": len(policy_metadata),
    }


def analyze_policy_complexity(policy_content: str) -> Dict[str, Any]:
    """Analyze complexity metrics for a single policy.
    
    Args:
        policy_content: Policy content to analyze
        
    Returns:
        Dictionary containing complexity analysis
    """
    structure = parse_policy_structure(policy_content)
    
    lines = policy_content.split("\n")
    
    # Count different types of constructs
    conditionals = sum(1 for line in lines if " if " in line)
    loops = sum(1 for line in lines if "some " in line or "every " in line)
    functions = sum(1 for line in lines if line.strip().endswith(") {"))
    
    # Calculate complexity score
    base_complexity = structure.get("complexity_score", 0)
    adjusted_complexity = base_complexity + (loops * 2) + (functions * 1.5)
    
    return {
        "base_complexity": base_complexity,
        "adjusted_complexity": adjusted_complexity,
        "conditionals": conditionals,
        "loops": loops,
        "functions": functions,
        "rule_count": structure.get("rule_count", 0),
        "lines_of_code": len([line for line in lines if line.strip()]),
        "complexity_per_rule": (
            adjusted_complexity / structure.get("rule_count", 1)
            if structure.get("rule_count", 0) > 0 else 0
        ),
    }


def get_policy_dependencies(policy_content: str) -> List[str]:
    """Extract policy dependencies from imports and references.
    
    Args:
        policy_content: Policy content to analyze
        
    Returns:
        List of dependency identifiers
    """
    dependencies = []
    
    lines = policy_content.split("\n")
    
    for line in lines:
        line = line.strip()
        
        # Extract imports
        if line.startswith("import "):
            import_path = line.replace("import ", "").strip()
            # Extract package name from import path
            if "." in import_path:
                dependencies.append(import_path.split(".")[-1])
            else:
                dependencies.append(import_path)
    
    return list(set(dependencies))  # Remove duplicates
