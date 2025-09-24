"""
Production-grade validation and error handling for Risk Scoring Framework.

This module provides comprehensive input validation, error recovery, and
robustness features for the risk scoring system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
from contextlib import asynccontextmanager

from ..domain.analysis_models import SecurityFinding, RiskLevel, RiskScore, BusinessImpact
from ..config.risk_scoring_config import RiskScoringConfiguration

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation with details about issues found."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_data: Optional[Any] = None


class SecurityFindingValidator:
    """Validator for SecurityFinding objects."""
    
    def __init__(self, config: RiskScoringConfiguration):
        self.config = config
    
    def validate_finding(self, finding: SecurityFinding) -> ValidationResult:
        """Validate a single security finding."""
        errors = []
        warnings = []
        
        # Validate required fields
        if not finding.finding_id:
            errors.append("finding_id is required")
        
        if not finding.detector_id:
            errors.append("detector_id is required")
        
        if not finding.category:
            warnings.append("category is empty, using 'unknown'")
        
        if not finding.description:
            warnings.append("description is empty, using default")
        
        # Validate severity
        if finding.severity not in RiskLevel:
            errors.append(f"Invalid severity: {finding.severity}")
        
        # Validate confidence
        if not (0.0 <= finding.confidence <= 1.0):
            errors.append(f"confidence must be between 0.0 and 1.0, got {finding.confidence}")
        
        # Validate timestamp
        if finding.timestamp > datetime.now(timezone.utc):
            warnings.append("finding timestamp is in the future")
        
        # Validate metadata
        if finding.metadata:
            metadata_validation = self._validate_metadata(finding.metadata)
            errors.extend(metadata_validation.errors)
            warnings.extend(metadata_validation.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> ValidationResult:
        """Validate finding metadata."""
        errors = []
        warnings = []
        
        # Validate CVSS-related fields
        cvss_fields = [
            'confidentiality_impact', 'integrity_impact', 'availability_impact',
            'attack_vector', 'attack_complexity', 'privileges_required', 'user_interaction'
        ]
        
        valid_impact_values = ['none', 'low', 'medium', 'high', 'complete']
        for field in ['confidentiality_impact', 'integrity_impact', 'availability_impact']:
            if field in metadata:
                value = metadata[field]
                if value not in valid_impact_values:
                    warnings.append(f"Invalid {field}: {value}, using 'medium'")
        
        valid_attack_vectors = ['physical', 'local', 'adjacent', 'network']
        if 'attack_vector' in metadata:
            if metadata['attack_vector'] not in valid_attack_vectors:
                warnings.append(f"Invalid attack_vector: {metadata['attack_vector']}, using 'network'")
        
        valid_complexities = ['low', 'high']
        if 'attack_complexity' in metadata:
            if metadata['attack_complexity'] not in valid_complexities:
                warnings.append(f"Invalid attack_complexity: {metadata['attack_complexity']}, using 'low'")
        
        # Validate lists
        list_fields = ['affected_systems', 'affected_processes', 'applicable_regulations']
        for field in list_fields:
            if field in metadata:
                if not isinstance(metadata[field], list):
                    warnings.append(f"{field} should be a list, converting from {type(metadata[field])}")
        
        # Validate data classification
        valid_classifications = ['public', 'internal', 'confidential', 'restricted', 'pii', 'financial', 'health']
        if 'data_classification' in metadata:
            if metadata['data_classification'].lower() not in valid_classifications:
                warnings.append(f"Unknown data_classification: {metadata['data_classification']}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_findings_batch(self, findings: List[SecurityFinding]) -> ValidationResult:
        """Validate a batch of security findings."""
        if not findings:
            return ValidationResult(
                is_valid=False,
                errors=["Empty findings list"],
                warnings=[]
            )
        
        if len(findings) > self.config.validation_config.max_findings_per_request:
            return ValidationResult(
                is_valid=False,
                errors=[f"Too many findings: {len(findings)}, max allowed: {self.config.validation_config.max_findings_per_request}"],
                warnings=[]
            )
        
        all_errors = []
        all_warnings = []
        valid_count = 0
        
        for i, finding in enumerate(findings):
            result = self.validate_finding(finding)
            if result.is_valid:
                valid_count += 1
            
            # Prefix errors/warnings with finding index
            all_errors.extend([f"Finding {i}: {error}" for error in result.errors])
            all_warnings.extend([f"Finding {i}: {warning}" for warning in result.warnings])
        
        # Check if we have enough valid findings
        valid_ratio = valid_count / len(findings) if findings else 0
        if valid_ratio < 0.5:
            all_errors.append(f"Too many invalid findings: {valid_count}/{len(findings)} valid (minimum 50% required)")
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )


class RiskScoreValidator:
    """Validator for RiskScore objects."""
    
    def __init__(self, config: RiskScoringConfiguration):
        self.config = config
    
    def validate_risk_score(self, risk_score: RiskScore) -> ValidationResult:
        """Validate a risk score object."""
        errors = []
        warnings = []
        
        # Validate composite score
        if not (0.0 <= risk_score.composite_score <= 1.0):
            errors.append(f"composite_score must be between 0.0 and 1.0, got {risk_score.composite_score}")
        
        # Validate confidence
        if not (0.0 <= risk_score.confidence <= 1.0):
            errors.append(f"confidence must be between 0.0 and 1.0, got {risk_score.confidence}")
        
        # Validate risk level consistency
        expected_level = self._determine_expected_risk_level(risk_score.composite_score)
        if risk_score.risk_level != expected_level:
            warnings.append(f"risk_level {risk_score.risk_level} inconsistent with composite_score {risk_score.composite_score}")
        
        # Validate breakdown
        if risk_score.breakdown:
            breakdown_validation = self._validate_risk_breakdown(risk_score.breakdown)
            errors.extend(breakdown_validation.errors)
            warnings.extend(breakdown_validation.warnings)
        
        # Validate timestamp
        if risk_score.timestamp > datetime.now(timezone.utc):
            warnings.append("risk_score timestamp is in the future")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _determine_expected_risk_level(self, score: float) -> RiskLevel:
        """Determine expected risk level from score."""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _validate_risk_breakdown(self, breakdown) -> ValidationResult:
        """Validate risk breakdown."""
        errors = []
        warnings = []
        
        # Validate individual risk components
        components = ['technical_risk', 'business_risk', 'regulatory_risk', 'temporal_risk']
        for component in components:
            if hasattr(breakdown, component):
                value = getattr(breakdown, component)
                if not (0.0 <= value <= 1.0):
                    errors.append(f"{component} must be between 0.0 and 1.0, got {value}")
        
        # Validate contributing factors
        if hasattr(breakdown, 'contributing_factors') and breakdown.contributing_factors:
            total_contribution = sum(factor.contribution for factor in breakdown.contributing_factors)
            total_weight = sum(factor.weight for factor in breakdown.contributing_factors)
            
            if abs(total_weight - 1.0) > 0.1:
                warnings.append(f"contributing_factors weights sum to {total_weight}, expected ~1.0")
            
            # Check for negative contributions
            for factor in breakdown.contributing_factors:
                if factor.contribution < 0:
                    errors.append(f"negative contribution in factor {factor.factor_name}: {factor.contribution}")
                if factor.weight < 0:
                    errors.append(f"negative weight in factor {factor.factor_name}: {factor.weight}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class BusinessImpactValidator:
    """Validator for BusinessImpact objects."""
    
    def __init__(self, config: RiskScoringConfiguration):
        self.config = config
    
    def validate_business_impact(self, impact: BusinessImpact) -> ValidationResult:
        """Validate a business impact object."""
        errors = []
        warnings = []
        
        # Validate total risk value
        if impact.total_risk_value < 0:
            errors.append(f"total_risk_value cannot be negative: {impact.total_risk_value}")
        
        # Validate financial impact
        if impact.financial_impact:
            financial_validation = self._validate_financial_impact(impact.financial_impact)
            errors.extend(financial_validation.errors)
            warnings.extend(financial_validation.warnings)
        
        # Validate confidence interval
        if impact.confidence_interval:
            ci_validation = self._validate_confidence_interval(impact.confidence_interval)
            errors.extend(ci_validation.errors)
            warnings.extend(ci_validation.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_financial_impact(self, financial_impact: Dict[str, Any]) -> ValidationResult:
        """Validate financial impact data."""
        errors = []
        warnings = []
        
        # Check for required fields
        if 'total_estimated_cost' not in financial_impact:
            errors.append("financial_impact missing 'total_estimated_cost'")
        else:
            cost = financial_impact['total_estimated_cost']
            if cost < 0:
                errors.append(f"total_estimated_cost cannot be negative: {cost}")
        
        # Validate cost breakdown if present
        if 'cost_breakdown' in financial_impact:
            breakdown = financial_impact['cost_breakdown']
            if isinstance(breakdown, dict):
                for key, value in breakdown.items():
                    if isinstance(value, (int, float)) and value < 0:
                        warnings.append(f"negative cost in breakdown {key}: {value}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_confidence_interval(self, confidence_interval: Dict[str, Any]) -> ValidationResult:
        """Validate confidence interval data."""
        errors = []
        warnings = []
        
        required_fields = ['lower', 'upper']
        for field in required_fields:
            if field not in confidence_interval:
                errors.append(f"confidence_interval missing '{field}'")
        
        if 'lower' in confidence_interval and 'upper' in confidence_interval:
            lower = confidence_interval['lower']
            upper = confidence_interval['upper']
            
            if lower > upper:
                errors.append(f"confidence interval lower bound ({lower}) > upper bound ({upper})")
            
            if lower < 0:
                warnings.append(f"negative lower bound in confidence interval: {lower}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class CircuitBreaker:
    """Circuit breaker for risk scoring operations."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    @asynccontextmanager
    async def protect(self):
        """Context manager for circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            yield
            if self.state == "HALF_OPEN":
                self._reset()
        except Exception as e:
            self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = datetime.now().timestamp() - self.last_failure_time
        return elapsed >= self.timeout_seconds
    
    def _record_failure(self):
        """Record a failure in the circuit breaker."""
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _reset(self):
        """Reset the circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        logger.info("Circuit breaker reset to CLOSED state")


class RiskScoringValidator:
    """Main validator for the Risk Scoring Framework."""
    
    def __init__(self, config: RiskScoringConfiguration):
        self.config = config
        self.finding_validator = SecurityFindingValidator(config)
        self.risk_score_validator = RiskScoreValidator(config)
        self.business_impact_validator = BusinessImpactValidator(config)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.performance_config.circuit_breaker_failure_threshold,
            timeout_seconds=config.performance_config.circuit_breaker_timeout_seconds
        ) if config.performance_config.circuit_breaker_enabled else None
    
    async def validate_analysis_input(self, findings: List[SecurityFinding]) -> ValidationResult:
        """Validate input for risk analysis."""
        with_circuit_breaker = self.circuit_breaker is not None
        
        if with_circuit_breaker:
            async with self.circuit_breaker.protect():
                return await self._validate_input_internal(findings)
        else:
            return await self._validate_input_internal(findings)
    
    async def _validate_input_internal(self, findings: List[SecurityFinding]) -> ValidationResult:
        """Internal validation logic."""
        if not findings:
            return ValidationResult(
                is_valid=False,
                errors=["No findings provided for analysis"],
                warnings=[]
            )
        
        # Validate the batch
        batch_result = self.finding_validator.validate_findings_batch(findings)
        
        # If strict validation is enabled, fail on any errors
        if self.config.validation_config.strict_validation and not batch_result.is_valid:
            return batch_result
        
        # Otherwise, filter out invalid findings and continue
        if not batch_result.is_valid:
            valid_findings = []
            for finding in findings:
                result = self.finding_validator.validate_finding(finding)
                if result.is_valid:
                    valid_findings.append(finding)
            
            if not valid_findings:
                return ValidationResult(
                    is_valid=False,
                    errors=["No valid findings after validation"],
                    warnings=batch_result.warnings
                )
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=batch_result.warnings + [f"Filtered out {len(findings) - len(valid_findings)} invalid findings"],
                corrected_data=valid_findings
            )
        
        return batch_result
    
    async def validate_analysis_output(self, risk_score: RiskScore, 
                                     business_impact: Optional[BusinessImpact] = None) -> ValidationResult:
        """Validate output from risk analysis."""
        all_errors = []
        all_warnings = []
        
        # Validate risk score
        risk_result = self.risk_score_validator.validate_risk_score(risk_score)
        all_errors.extend(risk_result.errors)
        all_warnings.extend(risk_result.warnings)
        
        # Validate business impact if provided
        if business_impact:
            impact_result = self.business_impact_validator.validate_business_impact(business_impact)
            all_errors.extend([f"BusinessImpact: {error}" for error in impact_result.errors])
            all_warnings.extend([f"BusinessImpact: {warning}" for warning in impact_result.warnings])
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )
    
    def sanitize_findings(self, findings: List[SecurityFinding]) -> List[SecurityFinding]:
        """Sanitize and correct common issues in findings."""
        sanitized = []
        
        for finding in findings:
            # Create a copy to avoid modifying original
            sanitized_finding = SecurityFinding(
                finding_id=finding.finding_id or f"auto_generated_{datetime.now().timestamp()}",
                detector_id=finding.detector_id or "unknown",
                severity=finding.severity,
                category=finding.category or "unknown",
                description=finding.description or "Security finding detected",
                timestamp=finding.timestamp,
                metadata=self._sanitize_metadata(finding.metadata),
                confidence=max(0.0, min(1.0, finding.confidence))  # Clamp to [0,1]
            )
            
            sanitized.append(sanitized_finding)
        
        return sanitized
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata by correcting common issues."""
        if not metadata:
            return {}
        
        sanitized = metadata.copy()
        
        # Sanitize CVSS fields
        valid_impact_values = ['none', 'low', 'medium', 'high', 'complete']
        for field in ['confidentiality_impact', 'integrity_impact', 'availability_impact']:
            if field in sanitized:
                value = str(sanitized[field]).lower()
                if value not in valid_impact_values:
                    sanitized[field] = 'medium'  # Default fallback
        
        # Sanitize attack vector
        valid_attack_vectors = ['physical', 'local', 'adjacent', 'network']
        if 'attack_vector' in sanitized:
            value = str(sanitized['attack_vector']).lower()
            if value not in valid_attack_vectors:
                sanitized['attack_vector'] = 'network'  # Default fallback
        
        # Sanitize lists
        list_fields = ['affected_systems', 'affected_processes', 'applicable_regulations']
        for field in list_fields:
            if field in sanitized and not isinstance(sanitized[field], list):
                # Try to convert to list
                try:
                    if isinstance(sanitized[field], str):
                        sanitized[field] = [sanitized[field]]
                    else:
                        sanitized[field] = list(sanitized[field])
                except:
                    sanitized[field] = []
        
        return sanitized
