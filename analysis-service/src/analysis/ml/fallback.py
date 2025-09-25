"""
ML Fallback Manager

This module provides fallback mechanisms for ML models in the Analysis Service,
ensuring system reliability when ML components fail or are unavailable.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reasons for using fallback mechanisms."""

    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    LOW_CONFIDENCE = "low_confidence"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"


class MLFallbackManager:
    """
    ML fallback management system for Analysis Service.

    Provides:
    - Rule-based fallbacks for ML models
    - Graceful degradation strategies
    - Fallback performance tracking
    - Automatic fallback selection
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fallback_config = config.get("fallback", {})

        # Configuration
        self.enable_fallbacks = self.fallback_config.get("enable_fallbacks", True)
        self.confidence_threshold = self.fallback_config.get(
            "confidence_threshold", 0.5
        )
        self.timeout_seconds = self.fallback_config.get("timeout_seconds", 30)

        # Fallback strategies
        self.fallback_strategies = {}
        self._initialize_fallback_strategies()

        # Performance tracking
        self.fallback_usage = {}
        self.total_requests = 0
        self.fallback_requests = 0

        logger.info(
            "ML Fallback Manager initialized",
            enable_fallbacks=self.enable_fallbacks,
            strategies=len(self.fallback_strategies),
        )

    async def execute_with_fallback(
        self, primary_func: Callable, fallback_type: str, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute function with fallback support.

        Args:
            primary_func: Primary ML function to execute
            fallback_type: Type of fallback to use if primary fails
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result with metadata about execution
        """
        self.total_requests += 1
        start_time = datetime.now(timezone.utc)

        try:
            # Try primary function with timeout
            result = await asyncio.wait_for(
                primary_func(*args, **kwargs), timeout=self.timeout_seconds
            )

            # Check confidence if available
            confidence = (
                result.get("confidence", 1.0) if isinstance(result, dict) else 1.0
            )

            if confidence < self.confidence_threshold:
                logger.warning(
                    "Low confidence result, using fallback",
                    confidence=confidence,
                    threshold=self.confidence_threshold,
                )

                return await self._execute_fallback(
                    fallback_type, FallbackReason.LOW_CONFIDENCE, *args, **kwargs
                )

            # Primary function succeeded
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return {
                "result": result,
                "execution_method": "primary",
                "execution_time": execution_time,
                "confidence": confidence,
                "fallback_used": False,
            }

        except asyncio.TimeoutError:
            logger.warning(
                "Primary function timeout, using fallback", timeout=self.timeout_seconds
            )

            return await self._execute_fallback(
                fallback_type, FallbackReason.TIMEOUT, *args, **kwargs
            )

        except Exception as e:
            logger.error("Primary function failed, using fallback", error=str(e))

            return await self._execute_fallback(
                fallback_type, FallbackReason.MODEL_ERROR, *args, **kwargs
            )

    async def _execute_fallback(
        self, fallback_type: str, reason: FallbackReason, *args, **kwargs
    ) -> Dict[str, Any]:
        """Execute fallback strategy."""
        if not self.enable_fallbacks:
            raise RuntimeError(f"Fallbacks disabled, cannot handle {reason.value}")

        self.fallback_requests += 1
        self.fallback_usage[reason.value] = self.fallback_usage.get(reason.value, 0) + 1

        start_time = datetime.now(timezone.utc)

        try:
            # Get fallback strategy
            fallback_func = self.fallback_strategies.get(fallback_type)

            if not fallback_func:
                raise ValueError(f"No fallback strategy for type: {fallback_type}")

            # Execute fallback
            result = await fallback_func(*args, **kwargs)

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.info(
                "Fallback executed successfully",
                fallback_type=fallback_type,
                reason=reason.value,
                execution_time=execution_time,
            )

            return {
                "result": result,
                "execution_method": "fallback",
                "fallback_type": fallback_type,
                "fallback_reason": reason.value,
                "execution_time": execution_time,
                "confidence": 0.3,  # Lower confidence for fallback results
                "fallback_used": True,
            }

        except Exception as e:
            logger.error(
                "Fallback execution failed",
                fallback_type=fallback_type,
                reason=reason.value,
                error=str(e),
            )

            # Return minimal error response
            return {
                "result": self._create_error_result(fallback_type, str(e)),
                "execution_method": "error_fallback",
                "fallback_type": fallback_type,
                "fallback_reason": reason.value,
                "execution_time": (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds(),
                "confidence": 0.0,
                "fallback_used": True,
                "error": str(e),
            }

    def _initialize_fallback_strategies(self):
        """Initialize fallback strategies for different ML tasks."""

        # Analysis generation fallback
        self.fallback_strategies["analysis_generation"] = (
            self._fallback_analysis_generation
        )

        # Pattern recognition fallback
        self.fallback_strategies["pattern_recognition"] = (
            self._fallback_pattern_recognition
        )

        # Risk scoring fallback
        self.fallback_strategies["risk_scoring"] = self._fallback_risk_scoring

        # Compliance mapping fallback
        self.fallback_strategies["compliance_mapping"] = (
            self._fallback_compliance_mapping
        )

        # Embeddings fallback
        self.fallback_strategies["embeddings"] = self._fallback_embeddings

        # RAG query fallback
        self.fallback_strategies["rag_query"] = self._fallback_rag_query

    async def _fallback_analysis_generation(
        self, prompt: str, **kwargs
    ) -> Dict[str, Any]:
        """Rule-based fallback for analysis generation."""
        try:
            # Simple rule-based analysis
            analysis_text = self._generate_rule_based_analysis(prompt)

            return {
                "generated_text": analysis_text,
                "finish_reason": "stop",
                "tokens_generated": len(analysis_text.split()),
                "method": "rule_based",
            }

        except Exception as e:
            logger.error("Analysis generation fallback failed", error=str(e))
            raise

    async def _fallback_pattern_recognition(
        self, data: Any, **kwargs
    ) -> Dict[str, Any]:
        """Rule-based fallback for pattern recognition."""
        try:
            # Simple pattern detection based on data characteristics
            patterns = []

            if isinstance(data, dict):
                # Check for common patterns in data structure
                if "high_sev_hits" in data and len(data.get("high_sev_hits", [])) > 0:
                    patterns.append(
                        {
                            "pattern_type": "security_alert",
                            "confidence": 0.7,
                            "description": "High severity security findings detected",
                        }
                    )

                if (
                    "detector_errors" in data
                    and len(data.get("detector_errors", {})) > 2
                ):
                    patterns.append(
                        {
                            "pattern_type": "system_degradation",
                            "confidence": 0.6,
                            "description": "Multiple detector errors indicating system issues",
                        }
                    )

            return {
                "patterns": patterns,
                "analysis_type": "pattern_recognition",
                "method": "rule_based",
                "confidence": 0.5,
            }

        except Exception as e:
            logger.error("Pattern recognition fallback failed", error=str(e))
            raise

    async def _fallback_risk_scoring(
        self, findings: List[Any], **kwargs
    ) -> Dict[str, Any]:
        """Rule-based fallback for risk scoring."""
        try:
            # Simple risk scoring based on finding characteristics
            if not findings:
                risk_score = 0.0
                risk_level = "low"
            else:
                # Count high severity findings
                high_severity_count = sum(
                    1
                    for f in findings
                    if isinstance(f, dict)
                    and f.get("severity", "").lower() in ["high", "critical"]
                )

                # Simple risk calculation
                risk_score = min(
                    1.0, (len(findings) * 0.1) + (high_severity_count * 0.3)
                )

                if risk_score >= 0.8:
                    risk_level = "critical"
                elif risk_score >= 0.6:
                    risk_level = "high"
                elif risk_score >= 0.3:
                    risk_level = "medium"
                else:
                    risk_level = "low"

            return {
                "composite_score": risk_score,
                "risk_level": risk_level,
                "breakdown": {
                    "finding_count": len(findings),
                    "high_severity_count": (
                        high_severity_count if "high_severity_count" in locals() else 0
                    ),
                    "method": "rule_based",
                },
                "confidence": 0.4,
            }

        except Exception as e:
            logger.error("Risk scoring fallback failed", error=str(e))
            raise

    async def _fallback_compliance_mapping(
        self, findings: List[Any], **kwargs
    ) -> Dict[str, Any]:
        """Rule-based fallback for compliance mapping."""
        try:
            # Simple compliance mapping based on finding types
            mappings = []

            for finding in findings:
                if isinstance(finding, dict):
                    detector = finding.get("detector", "").lower()

                    # Simple mapping rules
                    if "pii" in detector or "privacy" in detector:
                        mappings.append(
                            {
                                "framework": "GDPR",
                                "control": "Article 32",
                                "description": "Security of processing",
                                "compliance_status": "requires_attention",
                            }
                        )

                    if "access" in detector or "auth" in detector:
                        mappings.append(
                            {
                                "framework": "SOC2",
                                "control": "CC6.1",
                                "description": "Logical and Physical Access Controls",
                                "compliance_status": "requires_attention",
                            }
                        )

            return {
                "mappings": mappings,
                "frameworks_covered": list(set(m["framework"] for m in mappings)),
                "method": "rule_based",
                "confidence": 0.3,
            }

        except Exception as e:
            logger.error("Compliance mapping fallback failed", error=str(e))
            raise

    async def _fallback_embeddings(
        self, texts: List[str], **kwargs
    ) -> List[List[float]]:
        """Simple fallback for embeddings generation."""
        try:
            # Generate simple hash-based embeddings
            embeddings = []

            for text in texts:
                # Create simple embedding from text characteristics
                embedding = self._create_simple_embedding(text)
                embeddings.append(embedding)

            return embeddings

        except Exception as e:
            logger.error("Embeddings fallback failed", error=str(e))
            raise

    async def _fallback_rag_query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Rule-based fallback for RAG queries."""
        try:
            # Simple keyword-based response
            answer = self._generate_keyword_based_answer(question)

            return {
                "answer": answer,
                "question": question,
                "context_used": "",
                "citations": [],
                "retrieved_documents": 0,
                "confidence": 0.2,
                "method": "keyword_fallback",
            }

        except Exception as e:
            logger.error("RAG query fallback failed", error=str(e))
            raise

    def _generate_rule_based_analysis(self, prompt: str) -> str:
        """Generate rule-based analysis text."""
        prompt_lower = prompt.lower()

        if "risk" in prompt_lower:
            return (
                "Based on the available data, a risk assessment indicates moderate concern. "
                "Recommend implementing appropriate monitoring and control measures."
            )

        elif "pattern" in prompt_lower:
            return (
                "Pattern analysis shows typical operational characteristics. "
                "No significant anomalies detected in the current data set."
            )

        elif "compliance" in prompt_lower:
            return (
                "Compliance review indicates adherence to standard frameworks. "
                "Continue monitoring for any regulatory requirement changes."
            )

        elif "security" in prompt_lower:
            return (
                "Security analysis shows standard protective measures are in place. "
                "Regular security assessments are recommended."
            )

        else:
            return (
                "Analysis completed using available data. "
                "Results indicate normal operational parameters within expected ranges."
            )

    def _create_simple_embedding(self, text: str, dimension: int = 384) -> List[float]:
        """Create simple embedding from text."""
        import hashlib

        # Create hash-based embedding
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Convert to numbers
        embedding = []
        for i in range(0, min(len(text_hash), dimension * 2), 2):
            hex_pair = text_hash[i : i + 2]
            value = int(hex_pair, 16) / 255.0
            embedding.append(value)

        # Pad to desired dimension
        while len(embedding) < dimension:
            embedding.append(0.0)

        return embedding[:dimension]

    def _generate_keyword_based_answer(self, question: str) -> str:
        """Generate keyword-based answer."""
        question_lower = question.lower()

        if any(word in question_lower for word in ["soc2", "soc 2"]):
            return (
                "SOC 2 is a compliance framework that focuses on security, availability, "
                "processing integrity, confidentiality, and privacy controls."
            )

        elif any(word in question_lower for word in ["gdpr", "privacy"]):
            return (
                "GDPR is the General Data Protection Regulation that governs data protection "
                "and privacy in the European Union and European Economic Area."
            )

        elif any(word in question_lower for word in ["hipaa", "healthcare"]):
            return (
                "HIPAA is the Health Insurance Portability and Accountability Act that "
                "provides data privacy and security provisions for safeguarding medical information."
            )

        elif any(word in question_lower for word in ["risk", "assessment"]):
            return (
                "Risk assessment involves identifying, analyzing, and evaluating potential "
                "risks to determine appropriate mitigation strategies."
            )

        else:
            return (
                "I don't have specific information to answer this question. "
                "Please consult relevant documentation or contact a subject matter expert."
            )

    def _create_error_result(self, fallback_type: str, error_message: str) -> Any:
        """Create minimal error result."""
        return {
            "error": f"Fallback failed for {fallback_type}: {error_message}",
            "fallback_type": fallback_type,
            "success": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_fallback_metrics(self) -> Dict[str, Any]:
        """Get fallback usage metrics."""
        fallback_rate = (
            self.fallback_requests / self.total_requests
            if self.total_requests > 0
            else 0
        )

        return {
            "total_requests": self.total_requests,
            "fallback_requests": self.fallback_requests,
            "fallback_rate": fallback_rate,
            "fallback_usage_by_reason": self.fallback_usage.copy(),
            "available_strategies": list(self.fallback_strategies.keys()),
            "configuration": {
                "enable_fallbacks": self.enable_fallbacks,
                "confidence_threshold": self.confidence_threshold,
                "timeout_seconds": self.timeout_seconds,
            },
        }

    async def test_fallback_strategy(
        self, fallback_type: str, *args, **kwargs
    ) -> Dict[str, Any]:
        """Test a specific fallback strategy."""
        try:
            if fallback_type not in self.fallback_strategies:
                raise ValueError(f"Unknown fallback type: {fallback_type}")

            start_time = datetime.now(timezone.utc)

            fallback_func = self.fallback_strategies[fallback_type]
            result = await fallback_func(*args, **kwargs)

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return {
                "fallback_type": fallback_type,
                "result": result,
                "execution_time": execution_time,
                "success": True,
                "test_timestamp": start_time.isoformat(),
            }

        except Exception as e:
            logger.error(
                "Fallback strategy test failed",
                fallback_type=fallback_type,
                error=str(e),
            )

            return {
                "fallback_type": fallback_type,
                "result": None,
                "execution_time": 0.0,
                "success": False,
                "error": str(e),
                "test_timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def shutdown(self):
        """Gracefully shutdown fallback manager."""
        try:
            logger.info("Shutting down ML Fallback Manager...")

            # Log final statistics
            metrics = self.get_fallback_metrics()
            logger.info("Final fallback metrics", **metrics)

            logger.info("ML Fallback Manager shutdown complete")

        except Exception as e:
            logger.error("Error during ML Fallback Manager shutdown", error=str(e))
