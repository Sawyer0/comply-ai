"""
Multi-layer input sanitization for comprehensive security protection.

Provides protection against SQL injection, XSS, path traversal, and other attacks.
"""

import html
import re
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import structlog
from pydantic import BaseModel, validator

from ..utils.correlation import get_correlation_id

logger = structlog.get_logger(__name__).bind(component="input_sanitization")


class AttackType(Enum):
    """Types of detected attacks."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    LDAP_INJECTION = "ldap_injection"
    XML_INJECTION = "xml_injection"
    SCRIPT_INJECTION = "script_injection"


class SanitizationLevel(Enum):
    """Levels of sanitization."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class SecuritySanitizer:
    """Multi-layer input sanitization with threat detection."""
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\bUNION\s+SELECT\b)",
        r"(\bINTO\s+OUTFILE\b)",
        r"(\bLOAD_FILE\b)",
        r"(\bCHAR\s*\(\s*\d+\s*\))",
        r"(\bCONCAT\s*\()",
        r"(\bSUBSTRING\s*\()",
        r"(\bCAST\s*\()",
        r"(\bCONVERT\s*\()",
        r"(\bDECLARE\s+@)",
        r"(\bEXEC\s*\()",
        r"(\bSP_EXECUTESQL\b)"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>.*?</embed>",
        r"<applet[^>]*>.*?</applet>",
        r"javascript:",
        r"vbscript:",
        r"data:text/html",
        r"on\w+\s*=",
        r"<img[^>]*src\s*=\s*[\"']?javascript:",
        r"<link[^>]*href\s*=\s*[\"']?javascript:",
        r"<style[^>]*>.*?expression\s*\(",
        r"<meta[^>]*http-equiv\s*=\s*[\"']?refresh"
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"/etc/passwd",
        r"\\windows\\system32",
        r"/proc/self/environ",
        r"\\boot\.ini",
        r"/var/log/",
        r"\\system32\\drivers\\etc\\hosts",
        r"\.\.%2f",
        r"\.\.%5c",
        r"%2e%2e%2f",
        r"%2e%2e%5c"
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}[\]<>]",
        r"\b(cat|ls|dir|type|more|less|head|tail|grep|find|locate)\b",
        r"\b(rm|del|rmdir|rd|mkdir|md|copy|cp|move|mv)\b",
        r"\b(wget|curl|nc|netcat|telnet|ssh|ftp)\b",
        r"\b(chmod|chown|su|sudo|passwd)\b",
        r"\b(ps|kill|killall|pkill|top|htop)\b",
        r"\b(ping|nslookup|dig|host|traceroute)\b"
    ]
    
    # LDAP injection patterns
    LDAP_INJECTION_PATTERNS = [
        r"[()&|!*]",
        r"\\[0-9a-fA-F]{2}",
        r"\*\)",
        r"\(\|",
        r"\(&",
        r"\(!"
    ]
    
    # XML injection patterns
    XML_INJECTION_PATTERNS = [
        r"<!ENTITY",
        r"<!DOCTYPE",
        r"<\?xml",
        r"SYSTEM\s+[\"']",
        r"PUBLIC\s+[\"']",
        r"&\w+;",
        r"<!\[CDATA\["
    ]
    
    def __init__(self, level: SanitizationLevel = SanitizationLevel.STRICT):
        """
        Initialize security sanitizer.
        
        Args:
            level: Sanitization level
        """
        self.level = level
        self.max_length_limits = {
            SanitizationLevel.BASIC: 10000,
            SanitizationLevel.STRICT: 5000,
            SanitizationLevel.PARANOID: 1000
        }
        
        # Compile regex patterns for performance
        self.compiled_patterns = {
            AttackType.SQL_INJECTION: [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS],
            AttackType.XSS: [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS],
            AttackType.PATH_TRAVERSAL: [re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS],
            AttackType.COMMAND_INJECTION: [re.compile(p, re.IGNORECASE) for p in self.COMMAND_INJECTION_PATTERNS],
            AttackType.LDAP_INJECTION: [re.compile(p, re.IGNORECASE) for p in self.LDAP_INJECTION_PATTERNS],
            AttackType.XML_INJECTION: [re.compile(p, re.IGNORECASE) for p in self.XML_INJECTION_PATTERNS]
        }
    
    def sanitize_input(self, data: Any, field_name: str = "unknown") -> Any:
        """
        Comprehensive input sanitization.
        
        Args:
            data: Input data to sanitize
            field_name: Name of the field being sanitized
            
        Returns:
            Sanitized data
            
        Raises:
            ValueError: If input contains malicious content and cannot be sanitized
        """
        if isinstance(data, str):
            return self.sanitize_string(data, field_name)
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v, f"{field_name}.{k}") for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item, f"{field_name}[{i}]") for i, item in enumerate(data)]
        else:
            return data
    
    def sanitize_string(self, text: str, field_name: str = "unknown") -> str:
        """
        Sanitize string input with multi-layer protection.
        
        Args:
            text: String to sanitize
            field_name: Name of the field
            
        Returns:
            Sanitized string
            
        Raises:
            ValueError: If input contains malicious content
        """
        if not isinstance(text, str):
            return str(text)
        
        original_text = text
        
        # Length validation
        max_length = self.max_length_limits[self.level]
        if len(text) > max_length:
            if self.level == SanitizationLevel.PARANOID:
                raise ValueError(f"Input exceeds maximum length of {max_length} characters")
            else:
                text = text[:max_length]
                logger.warning("Input truncated due to length",
                             field_name=field_name,
                             original_length=len(original_text),
                             truncated_length=len(text),
                             correlation_id=get_correlation_id())
        
        # Detect malicious patterns
        detected_attacks = self.detect_malicious_patterns(text, field_name)
        
        if detected_attacks:
            if self.level == SanitizationLevel.PARANOID:
                raise ValueError(f"Malicious input detected: {', '.join([a.value for a in detected_attacks])}")
            else:
                # Log the attack attempt
                logger.warning("Malicious input detected and sanitized",
                             field_name=field_name,
                             attack_types=[a.value for a in detected_attacks],
                             correlation_id=get_correlation_id())
        
        # Apply sanitization based on level
        if self.level in [SanitizationLevel.STRICT, SanitizationLevel.PARANOID]:
            # HTML escape
            text = html.escape(text, quote=True)
            
            # Remove null bytes and control characters
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
        
        if self.level == SanitizationLevel.PARANOID:
            # Additional paranoid-level sanitization
            # Remove any remaining special characters that could be dangerous
            text = re.sub(r'[<>"\';\\&|`$(){}[\]*?]', '', text)
        
        return text
    
    def detect_malicious_patterns(self, text: str, field_name: str = "unknown") -> List[AttackType]:
        """
        Detect malicious patterns in input text.
        
        Args:
            text: Text to analyze
            field_name: Name of the field
            
        Returns:
            List of detected attack types
        """
        detected_attacks = []
        
        for attack_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected_attacks.append(attack_type)
                    
                    # Log detailed information about the attack
                    logger.warning("Attack pattern detected",
                                 field_name=field_name,
                                 attack_type=attack_type.value,
                                 pattern=pattern.pattern,
                                 correlation_id=get_correlation_id())
                    break  # Only log first match per attack type
        
        return detected_attacks
    
    def validate_file_path(self, file_path: str) -> str:
        """
        Validate and sanitize file paths.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Sanitized file path
            
        Raises:
            ValueError: If path contains traversal attempts
        """
        # Normalize path separators
        normalized_path = file_path.replace('\\', '/').replace('//', '/')
        
        # Check for path traversal
        if '..' in normalized_path or normalized_path.startswith('/'):
            logger.warning("Path traversal attempt detected",
                         file_path=file_path,
                         correlation_id=get_correlation_id())
            raise ValueError("Invalid file path: path traversal detected")
        
        # Remove any dangerous characters
        sanitized_path = re.sub(r'[^a-zA-Z0-9._/-]', '', normalized_path)
        
        return sanitized_path
    
    def validate_email(self, email: str) -> str:
        """
        Validate and sanitize email addresses.
        
        Args:
            email: Email address to validate
            
        Returns:
            Sanitized email address
            
        Raises:
            ValueError: If email format is invalid
        """
        # Basic email regex (not RFC compliant but secure)
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        if not email_pattern.match(email):
            raise ValueError("Invalid email format")
        
        # Additional sanitization
        sanitized_email = email.lower().strip()
        
        # Check length
        if len(sanitized_email) > 254:  # RFC 5321 limit
            raise ValueError("Email address too long")
        
        return sanitized_email
    
    def validate_url(self, url: str) -> str:
        """
        Validate and sanitize URLs.
        
        Args:
            url: URL to validate
            
        Returns:
            Sanitized URL
            
        Raises:
            ValueError: If URL is invalid or dangerous
        """
        # Allow only HTTP and HTTPS
        if not url.startswith(('http://', 'https://')):
            raise ValueError("Only HTTP and HTTPS URLs are allowed")
        
        # Check for dangerous schemes
        dangerous_schemes = ['javascript:', 'data:', 'vbscript:', 'file:', 'ftp:']
        for scheme in dangerous_schemes:
            if scheme in url.lower():
                raise ValueError(f"Dangerous URL scheme detected: {scheme}")
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(url):
            raise ValueError("Invalid URL format")
        
        return url


# Enhanced Pydantic models with security validation
class SecureDetectorRequest(BaseModel):
    """Secure detector request with comprehensive input validation."""
    
    detector_type: str
    content: str
    metadata: Dict[str, Any] = {}
    tenant_id: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        # Enable validation on assignment
        validate_assignment = True
        # Use enum values
        use_enum_values = True
    
    @validator('detector_type')
    def validate_detector_type(cls, v):
        """Validate detector type."""
        allowed_types = ['presidio', 'deberta', 'custom', 'toxicity', 'pii']
        if v not in allowed_types:
            raise ValueError(f"Invalid detector type: {v}. Allowed: {', '.join(allowed_types)}")
        return v
    
    @validator('content')
    def sanitize_content(cls, v):
        """Sanitize content field."""
        if not isinstance(v, str):
            raise ValueError("Content must be a string")
        
        sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)
        return sanitizer.sanitize_string(v, "content")
    
    @validator('metadata')
    def sanitize_metadata(cls, v):
        """Sanitize metadata dictionary."""
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        
        sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)
        return sanitizer.sanitize_input(v, "metadata")
    
    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        """Validate tenant ID format."""
        if v is None:
            return v
        
        if not isinstance(v, str):
            raise ValueError("Tenant ID must be a string")
        
        # Tenant ID should be alphanumeric with hyphens/underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid tenant ID format")
        
        if len(v) > 64:
            raise ValueError("Tenant ID too long")
        
        return v


class SecureFileUploadRequest(BaseModel):
    """Secure file upload request."""
    
    filename: str
    content_type: str
    file_size: int
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename."""
        sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)
        return sanitizer.validate_file_path(v)
    
    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate content type."""
        allowed_types = [
            'text/plain',
            'text/csv',
            'application/json',
            'application/pdf',
            'image/jpeg',
            'image/png'
        ]
        
        if v not in allowed_types:
            raise ValueError(f"Content type not allowed: {v}")
        
        return v
    
    @validator('file_size')
    def validate_file_size(cls, v):
        """Validate file size."""
        max_size = 10 * 1024 * 1024  # 10MB
        
        if v > max_size:
            raise ValueError(f"File size exceeds maximum of {max_size} bytes")
        
        if v <= 0:
            raise ValueError("File size must be positive")
        
        return v


# Middleware integration
def create_sanitization_middleware(level: SanitizationLevel = SanitizationLevel.STRICT):
    """
    Create FastAPI middleware for input sanitization.
    
    Args:
        level: Sanitization level
        
    Returns:
        Middleware function
    """
    sanitizer = SecuritySanitizer(level)
    
    async def sanitization_middleware(request, call_next):
        """Sanitize request data."""
        # This would integrate with FastAPI request processing
        # For now, we'll just pass through
        response = await call_next(request)
        return response
    
    return sanitization_middleware