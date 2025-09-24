"""
Production-grade configuration for Risk Scoring Framework.

This module provides comprehensive configuration management for the risk scoring
system with validation, defaults, and environment-based overrides.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import os
import yaml
import logging

logger = logging.getLogger(__name__)


class RiskCalculationMethod(Enum):
    """Risk calculation methodologies."""
    CVSS_ENHANCED = "cvss_enhanced"
    BUSINESS_WEIGHTED = "business_weighted"
    STATISTICAL = "statistical"
    HYBRID = "hybrid"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    FEDRAMP = "fedramp"


@dataclass
class RiskWeightsConfig:
    """Configuration for risk dimension weights."""
    technical: float = 0.3
    business: float = 0.3
    regulatory: float = 0.25
    temporal: float = 0.15
    
    def __post_init__(self):
        """Validate weights sum to approximately 1.0."""
        total = self.technical + self.business + self.regulatory + self.temporal
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Risk weights must sum to 1.0, got {total}")


@dataclass
class CVSSConfig:
    """Configuration for CVSS-like technical risk scoring."""
    confidentiality_weight: float = 0.3
    integrity_weight: float = 0.3
    availability_weight: float = 0.3
    scope_weight: float = 0.1
    
    # Attack vector multipliers
    attack_vector_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'physical': 0.2,
        'local': 0.55,
        'adjacent': 0.62,
        'network': 0.85
    })
    
    # Attack complexity multipliers
    attack_complexity_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.77,
        'high': 0.44
    })


@dataclass
class BusinessContextConfig:
    """Configuration for business context evaluation."""
    size_multiplier: float = 1.0
    industry_type: str = "technology"
    critical_processes: List[str] = field(default_factory=list)
    system_criticality: Dict[str, float] = field(default_factory=dict)
    process_criticality: Dict[str, float] = field(default_factory=dict)
    
    # Business impact thresholds
    revenue_impact_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 10000,
        'medium': 100000,
        'high': 1000000,
        'critical': 10000000
    })


@dataclass
class RegulatoryConfig:
    """Configuration for regulatory compliance weights."""
    applicable_frameworks: List[ComplianceFramework] = field(default_factory=list)
    
    # Framework-specific weights
    framework_weights: Dict[str, float] = field(default_factory=lambda: {
        'soc2': 1.2,
        'iso27001': 1.1,
        'hipaa': 1.5,
        'gdpr': 1.4,
        'pci_dss': 1.3,
        'sox': 1.4,
        'fedramp': 1.6
    })
    
    # Compliance posture per framework (0.0-1.0)
    compliance_posture: Dict[str, float] = field(default_factory=lambda: {
        'soc2': 0.8,
        'iso27001': 0.8,
        'hipaa': 0.9,
        'gdpr': 0.7,
        'pci_dss': 0.8,
        'sox': 0.9,
        'fedramp': 0.8
    })
    
    # Audit frequency multipliers
    audit_frequency: Dict[str, float] = field(default_factory=lambda: {
        'soc2': 1.2,
        'iso27001': 1.1,
        'hipaa': 1.4,
        'gdpr': 1.3,
        'pci_dss': 1.5,
        'sox': 1.6,
        'fedramp': 1.8
    })


@dataclass
class TemporalConfig:
    """Configuration for temporal risk assessment."""
    decay_days: int = 30
    critical_window_hours: int = 4
    high_priority_window_hours: int = 24
    
    # Urgency weights
    urgency_weights: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.8,
        'medium': 1.0,
        'high': 1.2,
        'critical': 1.5
    })
    
    # Age-based risk decay function parameters
    decay_function: str = "exponential"  # linear, exponential, logarithmic
    decay_rate: float = 0.1


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_concurrent_calculations: int = 100
    calculation_timeout_seconds: int = 30
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60


@dataclass
class ValidationConfig:
    """Configuration for input validation and error handling."""
    strict_validation: bool = True
    max_findings_per_request: int = 1000
    min_confidence_threshold: float = 0.0
    max_confidence_threshold: float = 1.0
    
    # Error handling behavior
    fail_on_invalid_findings: bool = False
    use_degraded_scoring_on_errors: bool = True
    log_validation_errors: bool = True


@dataclass
class RiskScoringConfiguration:
    """Comprehensive configuration for the Risk Scoring Framework."""
    
    # Core calculation method
    calculation_method: RiskCalculationMethod = RiskCalculationMethod.CVSS_ENHANCED
    
    # Component configurations
    risk_weights: RiskWeightsConfig = field(default_factory=RiskWeightsConfig)
    cvss_config: CVSSConfig = field(default_factory=CVSSConfig)
    business_config: BusinessContextConfig = field(default_factory=BusinessContextConfig)
    regulatory_config: RegulatoryConfig = field(default_factory=RegulatoryConfig)
    temporal_config: TemporalConfig = field(default_factory=TemporalConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Advanced features
    enable_predictive_scoring: bool = False
    enable_machine_learning_enhancements: bool = False
    enable_external_threat_intelligence: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RiskScoringConfiguration':
        """Create configuration from dictionary."""
        try:
            return cls(
                calculation_method=RiskCalculationMethod(
                    config_dict.get('calculation_method', 'cvss_enhanced')
                ),
                risk_weights=RiskWeightsConfig(**config_dict.get('risk_weights', {})),
                cvss_config=CVSSConfig(**config_dict.get('cvss_config', {})),
                business_config=BusinessContextConfig(**config_dict.get('business_config', {})),
                regulatory_config=RegulatoryConfig(**config_dict.get('regulatory_config', {})),
                temporal_config=TemporalConfig(**config_dict.get('temporal_config', {})),
                performance_config=PerformanceConfig(**config_dict.get('performance_config', {})),
                validation_config=ValidationConfig(**config_dict.get('validation_config', {})),
                enable_predictive_scoring=config_dict.get('enable_predictive_scoring', False),
                enable_machine_learning_enhancements=config_dict.get('enable_machine_learning_enhancements', False),
                enable_external_threat_intelligence=config_dict.get('enable_external_threat_intelligence', False)
            )
        except Exception as e:
            logger.error(f"Error creating RiskScoringConfiguration from dict: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    @classmethod
    def from_yaml_file(cls, file_path: str) -> 'RiskScoringConfiguration':
        """Load configuration from YAML file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict.get('risk_scoring', {}))
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {file_path}, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            raise
    
    @classmethod
    def from_environment(cls) -> 'RiskScoringConfiguration':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        if os.getenv('RISK_SCORING_METHOD'):
            config.calculation_method = RiskCalculationMethod(os.getenv('RISK_SCORING_METHOD'))
        
        # Risk weights from environment
        if os.getenv('RISK_WEIGHT_TECHNICAL'):
            config.risk_weights.technical = float(os.getenv('RISK_WEIGHT_TECHNICAL'))
        if os.getenv('RISK_WEIGHT_BUSINESS'):
            config.risk_weights.business = float(os.getenv('RISK_WEIGHT_BUSINESS'))
        if os.getenv('RISK_WEIGHT_REGULATORY'):
            config.risk_weights.regulatory = float(os.getenv('RISK_WEIGHT_REGULATORY'))
        if os.getenv('RISK_WEIGHT_TEMPORAL'):
            config.risk_weights.temporal = float(os.getenv('RISK_WEIGHT_TEMPORAL'))
        
        # Business context from environment
        if os.getenv('BUSINESS_SIZE_MULTIPLIER'):
            config.business_config.size_multiplier = float(os.getenv('BUSINESS_SIZE_MULTIPLIER'))
        if os.getenv('BUSINESS_INDUSTRY_TYPE'):
            config.business_config.industry_type = os.getenv('BUSINESS_INDUSTRY_TYPE')
        
        # Performance settings from environment
        if os.getenv('RISK_SCORING_CACHE_ENABLED'):
            config.performance_config.enable_caching = os.getenv('RISK_SCORING_CACHE_ENABLED').lower() == 'true'
        if os.getenv('RISK_SCORING_CACHE_TTL'):
            config.performance_config.cache_ttl_seconds = int(os.getenv('RISK_SCORING_CACHE_TTL'))
        
        return config
    
    def validate(self) -> None:
        """Validate configuration consistency."""
        # Validate risk weights
        self.risk_weights.__post_init__()
        
        # Validate CVSS weights
        cvss_total = (self.cvss_config.confidentiality_weight + 
                      self.cvss_config.integrity_weight + 
                      self.cvss_config.availability_weight + 
                      self.cvss_config.scope_weight)
        if abs(cvss_total - 1.0) > 0.01:
            raise ValueError(f"CVSS weights must sum to 1.0, got {cvss_total}")
        
        # Validate confidence thresholds
        if self.validation_config.min_confidence_threshold < 0.0:
            raise ValueError("min_confidence_threshold must be >= 0.0")
        if self.validation_config.max_confidence_threshold > 1.0:
            raise ValueError("max_confidence_threshold must be <= 1.0")
        if self.validation_config.min_confidence_threshold >= self.validation_config.max_confidence_threshold:
            raise ValueError("min_confidence_threshold must be < max_confidence_threshold")
        
        # Validate performance settings
        if self.performance_config.calculation_timeout_seconds <= 0:
            raise ValueError("calculation_timeout_seconds must be > 0")
        if self.performance_config.max_concurrent_calculations <= 0:
            raise ValueError("max_concurrent_calculations must be > 0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'calculation_method': self.calculation_method.value,
            'risk_weights': {
                'technical': self.risk_weights.technical,
                'business': self.risk_weights.business,
                'regulatory': self.risk_weights.regulatory,
                'temporal': self.risk_weights.temporal
            },
            'cvss_config': {
                'confidentiality_weight': self.cvss_config.confidentiality_weight,
                'integrity_weight': self.cvss_config.integrity_weight,
                'availability_weight': self.cvss_config.availability_weight,
                'scope_weight': self.cvss_config.scope_weight,
                'attack_vector_multipliers': self.cvss_config.attack_vector_multipliers,
                'attack_complexity_multipliers': self.cvss_config.attack_complexity_multipliers
            },
            'business_config': {
                'size_multiplier': self.business_config.size_multiplier,
                'industry_type': self.business_config.industry_type,
                'critical_processes': self.business_config.critical_processes,
                'system_criticality': self.business_config.system_criticality,
                'process_criticality': self.business_config.process_criticality,
                'revenue_impact_thresholds': self.business_config.revenue_impact_thresholds
            },
            'regulatory_config': {
                'applicable_frameworks': [f.value for f in self.regulatory_config.applicable_frameworks],
                'framework_weights': self.regulatory_config.framework_weights,
                'compliance_posture': self.regulatory_config.compliance_posture,
                'audit_frequency': self.regulatory_config.audit_frequency
            },
            'temporal_config': {
                'decay_days': self.temporal_config.decay_days,
                'critical_window_hours': self.temporal_config.critical_window_hours,
                'high_priority_window_hours': self.temporal_config.high_priority_window_hours,
                'urgency_weights': self.temporal_config.urgency_weights,
                'decay_function': self.temporal_config.decay_function,
                'decay_rate': self.temporal_config.decay_rate
            },
            'performance_config': {
                'enable_caching': self.performance_config.enable_caching,
                'cache_ttl_seconds': self.performance_config.cache_ttl_seconds,
                'max_concurrent_calculations': self.performance_config.max_concurrent_calculations,
                'calculation_timeout_seconds': self.performance_config.calculation_timeout_seconds,
                'circuit_breaker_enabled': self.performance_config.circuit_breaker_enabled,
                'circuit_breaker_failure_threshold': self.performance_config.circuit_breaker_failure_threshold,
                'circuit_breaker_timeout_seconds': self.performance_config.circuit_breaker_timeout_seconds
            },
            'validation_config': {
                'strict_validation': self.validation_config.strict_validation,
                'max_findings_per_request': self.validation_config.max_findings_per_request,
                'min_confidence_threshold': self.validation_config.min_confidence_threshold,
                'max_confidence_threshold': self.validation_config.max_confidence_threshold,
                'fail_on_invalid_findings': self.validation_config.fail_on_invalid_findings,
                'use_degraded_scoring_on_errors': self.validation_config.use_degraded_scoring_on_errors,
                'log_validation_errors': self.validation_config.log_validation_errors
            },
            'enable_predictive_scoring': self.enable_predictive_scoring,
            'enable_machine_learning_enhancements': self.enable_machine_learning_enhancements,
            'enable_external_threat_intelligence': self.enable_external_threat_intelligence
        }


def get_default_config() -> RiskScoringConfiguration:
    """Get default production-ready configuration."""
    return RiskScoringConfiguration()


def load_config_from_file(file_path: str = "config/risk_scoring.yaml") -> RiskScoringConfiguration:
    """Load configuration from file with fallback to environment and defaults."""
    try:
        # Try to load from file first
        config = RiskScoringConfiguration.from_yaml_file(file_path)
        logger.info(f"Loaded risk scoring configuration from {file_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load configuration from file: {e}")
        
        # Fallback to environment variables
        try:
            config = RiskScoringConfiguration.from_environment()
            logger.info("Loaded risk scoring configuration from environment variables")
            return config
        except Exception as e:
            logger.warning(f"Failed to load configuration from environment: {e}")
            
            # Final fallback to defaults
            config = get_default_config()
            logger.info("Using default risk scoring configuration")
            return config
