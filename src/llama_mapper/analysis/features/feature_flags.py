"""
Feature Flag System for gradual rollout of refactored components.

This system allows controlled rollout of new analysis engines while
maintaining the ability to rollback to the original implementation.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class FeatureFlagState(Enum):
    """Feature flag states."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    CANARY = "canary"  # Enabled for subset of traffic
    ROLLBACK = "rollback"  # Temporarily disabled due to issues


class FeatureFlag:
    """Individual feature flag configuration."""
    
    def __init__(self, name: str, state: FeatureFlagState = FeatureFlagState.DISABLED,
                 rollout_percentage: float = 0.0, description: str = ""):
        self.name = name
        self.state = state
        self.rollout_percentage = rollout_percentage  # 0.0 to 1.0
        self.description = description
        self.enabled_tenants: Set[str] = set()
        self.disabled_tenants: Set[str] = set()
        self.metadata: Dict[str, Any] = {}


class FeatureFlagManager:
    """
    Manages feature flags for gradual rollout of refactored components.
    
    Supports percentage-based rollouts, tenant-specific overrides,
    and emergency rollback capabilities.
    """
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self.default_state = FeatureFlagState.DISABLED
        
        # Initialize default flags for refactored components
        self._initialize_default_flags()
    
    def _initialize_default_flags(self) -> None:
        """Initialize default feature flags for refactored components."""
        self.register_flag(
            "enhanced_pattern_recognition",
            FeatureFlagState.DISABLED,
            description="Use new PatternRecognitionEngine instead of legacy pattern detection"
        )
        
        self.register_flag(
            "enhanced_risk_scoring", 
            FeatureFlagState.DISABLED,
            description="Use new RiskScoringEngine with business context"
        )
        
        self.register_flag(
            "enhanced_compliance_intelligence",
            FeatureFlagState.DISABLED,
            description="Use new ComplianceIntelligenceEngine for regulatory mapping"
        )
        
        self.register_flag(
            "template_orchestrator",
            FeatureFlagState.DISABLED,
            description="Use new TemplateOrchestrator instead of monolithic provider"
        )
        
        self.register_flag(
            "full_refactored_system",
            FeatureFlagState.DISABLED,
            description="Enable complete refactored analysis system"
        )
    
    def register_flag(self, name: str, state: FeatureFlagState = FeatureFlagState.DISABLED,
                     rollout_percentage: float = 0.0, description: str = "") -> None:
        """
        Register a new feature flag.
        
        Args:
            name: Flag name
            state: Initial state
            rollout_percentage: Percentage of traffic to enable (0.0-1.0)
            description: Flag description
        """
        self.flags[name] = FeatureFlag(name, state, rollout_percentage, description)
        logger.info(f"Registered feature flag: {name} (state: {state.value})")
    
    def is_enabled(self, flag_name: str, tenant_id: Optional[str] = None,
                  user_id: Optional[str] = None) -> bool:
        """
        Check if a feature flag is enabled for the given context.
        
        Args:
            flag_name: Name of the feature flag
            tenant_id: Optional tenant ID for tenant-specific overrides
            user_id: Optional user ID for user-based rollouts
            
        Returns:
            True if feature is enabled, False otherwise
        """
        if flag_name not in self.flags:
            logger.warning(f"Unknown feature flag: {flag_name}")
            return self.default_state == FeatureFlagState.ENABLED
        
        flag = self.flags[flag_name]
        
        # Check if flag is in rollback state
        if flag.state == FeatureFlagState.ROLLBACK:
            return False
        
        # Check if flag is completely disabled
        if flag.state == FeatureFlagState.DISABLED:
            return False
        
        # Check tenant-specific overrides
        if tenant_id:
            if tenant_id in flag.disabled_tenants:
                return False
            if tenant_id in flag.enabled_tenants:
                return True
        
        # Check if flag is fully enabled
        if flag.state == FeatureFlagState.ENABLED:
            return True
        
        # Handle canary rollout
        if flag.state == FeatureFlagState.CANARY:
            return self._is_in_rollout_percentage(flag, tenant_id, user_id)
        
        return False
    
    def enable_flag(self, flag_name: str, rollout_percentage: float = 1.0) -> bool:
        """
        Enable a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            rollout_percentage: Percentage to enable (0.0-1.0)
            
        Returns:
            True if flag was enabled successfully
        """
        if flag_name not in self.flags:
            logger.error(f"Cannot enable unknown flag: {flag_name}")
            return False
        
        flag = self.flags[flag_name]
        
        if rollout_percentage >= 1.0:
            flag.state = FeatureFlagState.ENABLED
            flag.rollout_percentage = 1.0
        else:
            flag.state = FeatureFlagState.CANARY
            flag.rollout_percentage = rollout_percentage
        
        logger.info(f"Enabled feature flag: {flag_name} (rollout: {rollout_percentage:.1%})")
        return True
    
    def disable_flag(self, flag_name: str) -> bool:
        """
        Disable a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            True if flag was disabled successfully
        """
        if flag_name not in self.flags:
            logger.error(f"Cannot disable unknown flag: {flag_name}")
            return False
        
        flag = self.flags[flag_name]
        flag.state = FeatureFlagState.DISABLED
        flag.rollout_percentage = 0.0
        
        logger.info(f"Disabled feature flag: {flag_name}")
        return True
    
    def rollback_flag(self, flag_name: str, reason: str = "") -> bool:
        """
        Emergency rollback of a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            reason: Reason for rollback
            
        Returns:
            True if flag was rolled back successfully
        """
        if flag_name not in self.flags:
            logger.error(f"Cannot rollback unknown flag: {flag_name}")
            return False
        
        flag = self.flags[flag_name]
        flag.state = FeatureFlagState.ROLLBACK
        flag.metadata['rollback_reason'] = reason
        flag.metadata['rollback_time'] = logger.info.__globals__.get('datetime', __import__('datetime')).datetime.now().isoformat()
        
        logger.warning(f"Rolled back feature flag: {flag_name} (reason: {reason})")
        return True
    
    def enable_for_tenant(self, flag_name: str, tenant_id: str) -> bool:
        """
        Enable a feature flag for a specific tenant.
        
        Args:
            flag_name: Name of the feature flag
            tenant_id: Tenant ID to enable for
            
        Returns:
            True if successfully enabled for tenant
        """
        if flag_name not in self.flags:
            logger.error(f"Cannot enable unknown flag for tenant: {flag_name}")
            return False
        
        flag = self.flags[flag_name]
        flag.enabled_tenants.add(tenant_id)
        flag.disabled_tenants.discard(tenant_id)  # Remove from disabled if present
        
        logger.info(f"Enabled feature flag {flag_name} for tenant: {tenant_id}")
        return True
    
    def disable_for_tenant(self, flag_name: str, tenant_id: str) -> bool:
        """
        Disable a feature flag for a specific tenant.
        
        Args:
            flag_name: Name of the feature flag
            tenant_id: Tenant ID to disable for
            
        Returns:
            True if successfully disabled for tenant
        """
        if flag_name not in self.flags:
            logger.error(f"Cannot disable unknown flag for tenant: {flag_name}")
            return False
        
        flag = self.flags[flag_name]
        flag.disabled_tenants.add(tenant_id)
        flag.enabled_tenants.discard(tenant_id)  # Remove from enabled if present
        
        logger.info(f"Disabled feature flag {flag_name} for tenant: {tenant_id}")
        return True
    
    def get_flag_status(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            Flag status dictionary or None if not found
        """
        if flag_name not in self.flags:
            return None
        
        flag = self.flags[flag_name]
        
        return {
            'name': flag.name,
            'state': flag.state.value,
            'rollout_percentage': flag.rollout_percentage,
            'description': flag.description,
            'enabled_tenants': list(flag.enabled_tenants),
            'disabled_tenants': list(flag.disabled_tenants),
            'metadata': flag.metadata
        }
    
    def get_all_flags_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status for all feature flags.
        
        Returns:
            Dictionary mapping flag names to status information
        """
        return {name: self.get_flag_status(name) for name in self.flags.keys()}
    
    def _is_in_rollout_percentage(self, flag: FeatureFlag, tenant_id: Optional[str],
                                user_id: Optional[str]) -> bool:
        """
        Determine if request is within rollout percentage.
        
        Uses consistent hashing to ensure same tenant/user always gets same result.
        """
        # Create a consistent hash input
        hash_input = f"{flag.name}:{tenant_id or 'unknown'}:{user_id or 'unknown'}"
        
        # Simple hash function (in production, use a proper hash function)
        hash_value = sum(ord(c) for c in hash_input) % 100
        
        # Check if hash falls within rollout percentage
        return (hash_value / 100.0) < flag.rollout_percentage


# Global feature flag manager instance
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance."""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager


def is_feature_enabled(flag_name: str, tenant_id: Optional[str] = None,
                      user_id: Optional[str] = None) -> bool:
    """
    Convenience function to check if a feature is enabled.
    
    Args:
        flag_name: Name of the feature flag
        tenant_id: Optional tenant ID
        user_id: Optional user ID
        
    Returns:
        True if feature is enabled
    """
    manager = get_feature_flag_manager()
    return manager.is_enabled(flag_name, tenant_id, user_id)


def enable_refactored_components_gradually() -> None:
    """
    Enable refactored components with a gradual rollout plan.
    
    This function demonstrates a typical rollout strategy.
    """
    manager = get_feature_flag_manager()
    
    # Phase 1: Enable pattern recognition for 10% of traffic
    manager.enable_flag("enhanced_pattern_recognition", rollout_percentage=0.1)
    
    # Phase 2: Enable risk scoring for 5% of traffic (more conservative)
    manager.enable_flag("enhanced_risk_scoring", rollout_percentage=0.05)
    
    # Phase 3: Enable compliance intelligence for 5% of traffic
    manager.enable_flag("enhanced_compliance_intelligence", rollout_percentage=0.05)
    
    logger.info("Initiated gradual rollout of refactored components")


def enable_full_refactored_system(rollout_percentage: float = 0.01) -> None:
    """
    Enable the full refactored system for a small percentage of traffic.
    
    Args:
        rollout_percentage: Percentage of traffic to enable (default 1%)
    """
    manager = get_feature_flag_manager()
    
    # Enable orchestrator with very conservative rollout
    manager.enable_flag("template_orchestrator", rollout_percentage)
    manager.enable_flag("full_refactored_system", rollout_percentage)
    
    logger.info(f"Enabled full refactored system for {rollout_percentage:.1%} of traffic")


def emergency_rollback_all() -> None:
    """Emergency rollback of all refactored components."""
    manager = get_feature_flag_manager()
    
    flags_to_rollback = [
        "enhanced_pattern_recognition",
        "enhanced_risk_scoring", 
        "enhanced_compliance_intelligence",
        "template_orchestrator",
        "full_refactored_system"
    ]
    
    for flag_name in flags_to_rollback:
        manager.rollback_flag(flag_name, "Emergency rollback - system issues detected")
    
    logger.warning("Emergency rollback completed for all refactored components")