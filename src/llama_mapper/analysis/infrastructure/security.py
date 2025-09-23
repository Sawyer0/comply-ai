"""
Infrastructure implementation of the security validator for the Analysis Module.

This module contains the concrete implementation of the ISecurityValidator interface
for PII redaction and security validation.
"""

import logging
import re
from typing import Any, Dict, List

from ..domain.interfaces import ISecurityValidator

logger = logging.getLogger(__name__)


class AnalysisSecurityValidator(ISecurityValidator):
    """
    Analysis security validator implementation.
    
    Provides concrete implementation of the ISecurityValidator interface
    for PII redaction and security validation.
    """
    
    def __init__(self):
        """Initialize the analysis security validator."""
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'mac_address': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
        }
        
        logger.info("Initialized Analysis Security Validator")
    
    def validate_response_security(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and redact PII from analysis response.
        
        Args:
            response: Analysis response dictionary
            
        Returns:
            Response with PII redacted
        """
        try:
            # Create a copy to avoid modifying the original
            redacted_response = response.copy()
            
            # Redact PII from text fields
            text_fields = ['reason', 'remediation', 'notes']
            for field in text_fields:
                if field in redacted_response and isinstance(redacted_response[field], str):
                    redacted_response[field] = self._redact_pii(redacted_response[field])
            
            # Redact PII from OPA diff if present
            if 'opa_diff' in redacted_response and isinstance(redacted_response['opa_diff'], str):
                redacted_response['opa_diff'] = self._redact_pii(redacted_response['opa_diff'])
            
            # Redact PII from evidence references
            if 'evidence_refs' in redacted_response and isinstance(redacted_response['evidence_refs'], list):
                redacted_response['evidence_refs'] = [
                    self._redact_pii(str(ref)) for ref in redacted_response['evidence_refs']
                ]
            
            return redacted_response
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            # Return original response if security validation fails
            return response
    
    def validate_request_security(self, request_data: Dict[str, Any]) -> bool:
        """
        Validate request data for security issues.
        
        Args:
            request_data: Request data to validate
            
        Returns:
            True if secure, False otherwise
        """
        try:
            # Check for suspicious patterns in request data
            request_str = str(request_data)
            
            # Check for potential injection attempts
            injection_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'data:text/html',
                r'eval\s*\(',
                r'exec\s*\(',
            ]
            
            for pattern in injection_patterns:
                if re.search(pattern, request_str, re.IGNORECASE):
                    logger.warning(f"Potential injection attempt detected: {pattern}")
                    return False
            
            # Check for excessive data size
            if len(request_str) > 100000:  # 100KB limit
                logger.warning("Request data size exceeds limit")
                return False
            
            # Check for suspicious file paths
            suspicious_paths = [
                '../', '..\\', '/etc/', 'C:\\Windows\\',
                '/bin/', '/usr/bin/', 'C:\\System32\\'
            ]
            
            for path in suspicious_paths:
                if path in request_str:
                    logger.warning(f"Suspicious path detected: {path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Request security validation error: {e}")
            return False
    
    def get_security_headers(self) -> Dict[str, str]:
        """
        Get security headers for API responses.
        
        Returns:
            Dictionary of security headers
        """
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    def _redact_pii(self, text: str) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Text with PII redacted
        """
        if not isinstance(text, str):
            return text
        
        redacted_text = text
        
        # Redact different types of PII
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type == 'email':
                redacted_text = re.sub(pattern, '[EMAIL_REDACTED]', redacted_text)
            elif pii_type == 'phone':
                redacted_text = re.sub(pattern, '[PHONE_REDACTED]', redacted_text)
            elif pii_type == 'ssn':
                redacted_text = re.sub(pattern, '[SSN_REDACTED]', redacted_text)
            elif pii_type == 'credit_card':
                redacted_text = re.sub(pattern, '[CARD_REDACTED]', redacted_text)
            elif pii_type == 'ip_address':
                redacted_text = re.sub(pattern, '[IP_REDACTED]', redacted_text)
            elif pii_type == 'mac_address':
                redacted_text = re.sub(pattern, '[MAC_REDACTED]', redacted_text)
        
        return redacted_text


class PIIRedactor:
    """
    PII redaction utility class.
    
    Provides additional PII redaction capabilities beyond the security validator.
    """
    
    def __init__(self):
        """Initialize the PII redactor."""
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'mac_address': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',  # Generic API key pattern
            'token': r'\b[A-Za-z0-9+/]{40,}={0,2}\b',  # Generic token pattern
        }
    
    def redact_text(self, text: str) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Text with PII redacted
        """
        if not isinstance(text, str):
            return text
        
        redacted_text = text
        
        # Redact different types of PII
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type == 'email':
                redacted_text = re.sub(pattern, '[EMAIL_REDACTED]', redacted_text)
            elif pii_type == 'phone':
                redacted_text = re.sub(pattern, '[PHONE_REDACTED]', redacted_text)
            elif pii_type == 'ssn':
                redacted_text = re.sub(pattern, '[SSN_REDACTED]', redacted_text)
            elif pii_type == 'credit_card':
                redacted_text = re.sub(pattern, '[CARD_REDACTED]', redacted_text)
            elif pii_type == 'ip_address':
                redacted_text = re.sub(pattern, '[IP_REDACTED]', redacted_text)
            elif pii_type == 'mac_address':
                redacted_text = re.sub(pattern, '[MAC_REDACTED]', redacted_text)
            elif pii_type == 'api_key':
                redacted_text = re.sub(pattern, '[API_KEY_REDACTED]', redacted_text)
            elif pii_type == 'token':
                redacted_text = re.sub(pattern, '[TOKEN_REDACTED]', redacted_text)
        
        return redacted_text
    
    def detect_pii(self, text: str) -> List[str]:
        """
        Detect PII in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected PII types
        """
        if not isinstance(text, str):
            return []
        
        detected_pii = []
        
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                detected_pii.append(pii_type)
        
        return detected_pii