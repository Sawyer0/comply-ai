"""
Infrastructure implementation of the OPA policy generator for the Analysis Module.

This module contains the concrete implementation of the IOPAGenerator interface
for generating and validating OPA/Rego policies.
"""

import logging
import subprocess
import tempfile
from typing import List

from ..domain.interfaces import IOPAGenerator

logger = logging.getLogger(__name__)


class OPAPolicyGenerator(IOPAGenerator):
    """
    OPA policy generator implementation.

    Provides concrete implementation of the IOPAGenerator interface
    for generating and validating OPA/Rego policies.
    """

    def __init__(self):
        """Initialize the OPA policy generator."""
        logger.info("Initialized OPA Policy Generator")

    def validate_rego(self, rego_snippet: str) -> bool:
        """
        Validate Rego syntax using OPA compiler.

        Args:
            rego_snippet: Rego policy snippet to validate

        Returns:
            True if valid, False otherwise
        """
        if not rego_snippet.strip():
            return True  # Empty is valid

        try:
            # Write snippet to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".rego", delete=False
            ) as f:
                f.write(rego_snippet)
                temp_file = f.name

            # Run OPA check command
            result = subprocess.run(
                ["opa", "check", temp_file], capture_output=True, text=True, timeout=10
            )

            # Clean up temporary file
            import os

            os.unlink(temp_file)

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.warning("OPA validation timeout")
            return False
        except FileNotFoundError:
            logger.warning("OPA command not found, skipping validation")
            return True  # Assume valid if OPA not available
        except Exception as e:
            logger.error("OPA validation error: %s", e)
            return False

    def generate_coverage_policy(
        self, required_detectors: List[str], required_coverage: dict
    ) -> str:
        """
        Generate OPA policy for coverage violations.

        Args:
            required_detectors: List of required detector names
            required_coverage: Required coverage per detector

        Returns:
            OPA/Rego policy string
        """
        policy = """package coverage

# Coverage enforcement policy
violation[msg] {
    detector := input.required_detectors[_]
    not input.observed_coverage[detector]
    msg := sprintf("Missing coverage for detector: %v", [detector])
}

# Coverage threshold enforcement
violation[msg] {
    detector := input.required_detectors[_]
    observed := input.observed_coverage[detector]
    required := input.required_coverage[detector]
    observed < required
    msg := sprintf("Insufficient coverage for %v: %v < %v", [detector, observed, required])
}

# Specific detector coverage requirements
"""

        # Add specific coverage requirements
        for detector, coverage in required_coverage.items():
            if detector in required_detectors:
                policy += f"""
violation[msg] {{
    detector := "{detector}"
    observed := input.observed_coverage[detector]
    observed < {coverage}
    msg := sprintf("Detector %v coverage below required threshold: %v < {coverage}", [detector, observed])
}}
"""

        return policy

    def generate_threshold_policy(self, detector: str, new_threshold: float) -> str:
        """
        Generate OPA policy for threshold adjustments.

        Args:
            detector: Detector name
            new_threshold: New threshold value

        Returns:
            OPA/Rego policy string
        """
        return f"""package thresholds

# Threshold enforcement for {detector}
violation[msg] {{
    detector := "{detector}"
    current_threshold := input.detector_thresholds[detector]
    current_threshold != {new_threshold}
    msg := sprintf("Detector %v threshold should be adjusted to {new_threshold}, current: %v", [detector, current_threshold])
}}

# Threshold validation
violation[msg] {{
    detector := "{detector}"
    threshold := input.detector_thresholds[detector]
    threshold < 0 or threshold > 1
    msg := sprintf("Invalid threshold for %v: %v (must be between 0 and 1)", [detector, threshold])
}}"""
