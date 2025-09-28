"""Policy loader utility for OPA policy templates.

This module provides a thin orchestration layer that coordinates
file loading, parsing, and metadata extraction via specialized modules.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .file_loader import (
    PolicyLoadError,
    handle_policy_errors,
    load_all_policy_files,
    load_single_policy_by_name,
)
from .parser import validate_tenant_policy_data, validate_policy_syntax
from .metadata import PolicyMetadata, parse_policy_structure

logger = logging.getLogger(__name__)


class PolicyLoader:
    """Loads and manages OPA policy templates.

    Thin orchestration layer that coordinates file loading, parsing, and metadata
    extraction via specialized modules. Maintains backward compatibility.
    """

    def __init__(self, policies_directory: str = "policies"):
        """Initialize policy loader.

        Args:
            policies_directory: Directory containing policy files
        """
        self.policies_directory = Path(policies_directory)
        self.metadata = PolicyMetadata()

        logger.info(
            "Policy loader initialized",
            extra={"directory": str(self.policies_directory)},
        )

    def load_all_policies(self) -> Dict[str, str]:
        """Load all .rego policy files from the policies directory.

        Returns:
            Dictionary mapping policy names to their content

        Raises:
            PolicyLoadError: If policy loading fails
        """
        loaded_policies = load_all_policy_files(self.policies_directory)
        
        # Extract metadata for each loaded policy
        for policy_name, content in loaded_policies.items():
            policy_file = self.policies_directory / f"{policy_name}.rego"
            self.metadata.add_policy(policy_name, content, policy_file)

        return loaded_policies

    @handle_policy_errors(default_return=None)
    def load_policy(self, policy_name: str) -> Optional[str]:
        """Load a specific policy by name.

        Args:
            policy_name: Name of the policy to load (without .rego extension)

        Returns:
            Policy content if found, None otherwise
        """
        content = load_single_policy_by_name(self.policies_directory, policy_name)
        
        if content:
            policy_file = self.policies_directory / f"{policy_name}.rego"
            self.metadata.add_policy(policy_name, content, policy_file)
        
        return content

    # File operations and metadata extraction moved to specialized modules

    # Policy parsing and validation moved to specialized modules
    
    def parse_policy_structure(self, policy_content: str) -> Dict[str, Any]:
        """Parse policy structure (delegates to metadata module)."""
        return parse_policy_structure(policy_content)
    
    def validate_policy_syntax(self, policy_content: str) -> List[str]:
        """Validate policy syntax (delegates to parser module)."""
        return validate_policy_syntax(policy_content)

    def get_loaded_policies(self) -> Dict[str, str]:
        """Get all currently loaded policies."""
        return self.metadata.loaded_policies.copy()

    def get_policy_metadata(self, policy_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific policy."""
        return self.metadata.policy_metadata.get(policy_name)

    def get_all_policy_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all loaded policies."""
        return self.metadata.policy_metadata.copy()

    @handle_policy_errors(default_return=None)
    def load_tenant_policy_data(self) -> Optional[Dict[str, Any]]:
        """Load tenant policy configuration data."""
        data_file = self.policies_directory / "tenant_policies_data.json"

        if not data_file.exists():
            logger.warning("Tenant policy data file not found")
            return None

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("Tenant policy data loaded successfully")
        return data

    @handle_policy_errors(default_return=[])
    def validate_tenant_policy_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate tenant policy data structure (delegates to parser module)."""
        return validate_tenant_policy_data(data)

    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded policies (delegates to metadata module)."""
        stats = self.metadata.get_statistics()
        stats["policies_directory"] = str(self.policies_directory)
        return stats


# Export the main policy loading functionality
__all__ = [
    "PolicyLoader",
    "PolicyLoadError",
]
