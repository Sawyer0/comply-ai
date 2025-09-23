"""
XSS (Cross-Site Scripting) security patterns.

This module contains comprehensive patterns for detecting
XSS attacks across different vectors and techniques.
"""

from .base import PatternCollection
from ..interfaces import AttackType, ViolationSeverity


class XSSPatterns(PatternCollection):
    """Collection of XSS security patterns."""
    
    def __init__(self):
        """Initialize XSS patterns."""
        super().__init__(AttackType.XSS)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize all XSS patterns."""
        
        # Script tag patterns
        self.add_pattern(
            name="xss_script_tag",
            pattern=r"<script[^>]*>.*?</script>",
            severity=ViolationSeverity.HIGH,
            description="XSS - Script tags"
        )
        
        self.add_pattern(
            name="xss_script_src",
            pattern=r"<script[^>]*src\s*=\s*['\"][^'\"]*['\"]",
            severity=ViolationSeverity.HIGH,
            description="XSS - Script with external source"
        )
        
        # JavaScript protocol
        self.add_pattern(
            name="xss_javascript_protocol",
            pattern=r"javascript:",
            severity=ViolationSeverity.HIGH,
            description="XSS - JavaScript protocol"
        )
        
        self.add_pattern(
            name="xss_javascript_encoded",
            pattern=r"javascript\s*:",
            severity=ViolationSeverity.HIGH,
            description="XSS - JavaScript protocol (encoded)"
        )
        
        # Event handlers
        self.add_pattern(
            name="xss_event_handlers",
            pattern=r"on\w+\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - Event handlers"
        )
        
        self.add_pattern(
            name="xss_onclick",
            pattern=r"onclick\s*=\s*['\"][^'\"]*['\"]",
            severity=ViolationSeverity.HIGH,
            description="XSS - onclick event handler"
        )
        
        self.add_pattern(
            name="xss_onerror",
            pattern=r"onerror\s*=\s*['\"][^'\"]*['\"]",
            severity=ViolationSeverity.HIGH,
            description="XSS - onerror event handler"
        )
        
        self.add_pattern(
            name="xss_onload",
            pattern=r"onload\s*=\s*['\"][^'\"]*['\"]",
            severity=ViolationSeverity.HIGH,
            description="XSS - onload event handler"
        )
        
        # HTML elements
        self.add_pattern(
            name="xss_iframe",
            pattern=r"<iframe[^>]*>",
            severity=ViolationSeverity.HIGH,
            description="XSS - Iframe element"
        )
        
        self.add_pattern(
            name="xss_object",
            pattern=r"<object[^>]*>",
            severity=ViolationSeverity.HIGH,
            description="XSS - Object element"
        )
        
        self.add_pattern(
            name="xss_embed",
            pattern=r"<embed[^>]*>",
            severity=ViolationSeverity.HIGH,
            description="XSS - Embed element"
        )
        
        self.add_pattern(
            name="xss_link",
            pattern=r"<link[^>]*>",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Link element"
        )
        
        self.add_pattern(
            name="xss_meta",
            pattern=r"<meta[^>]*>",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Meta element"
        )
        
        self.add_pattern(
            name="xss_style",
            pattern=r"<style[^>]*>",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Style element"
        )
        
        # CSS-based XSS
        self.add_pattern(
            name="xss_css_expression",
            pattern=r"expression\s*\(",
            severity=ViolationSeverity.HIGH,
            description="XSS - CSS expression"
        )
        
        self.add_pattern(
            name="xss_css_import",
            pattern=r"@import\s+['\"][^'\"]*['\"]",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - CSS import"
        )
        
        self.add_pattern(
            name="xss_css_url",
            pattern=r"url\s*\(\s*['\"]?javascript:",
            severity=ViolationSeverity.HIGH,
            description="XSS - CSS url with javascript"
        )
        
        # VBScript
        self.add_pattern(
            name="xss_vbscript",
            pattern=r"vbscript:",
            severity=ViolationSeverity.HIGH,
            description="XSS - VBScript protocol"
        )
        
        # Data URIs
        self.add_pattern(
            name="xss_data_uri_html",
            pattern=r"data:text/html",
            severity=ViolationSeverity.HIGH,
            description="XSS - Data URI with HTML"
        )
        
        self.add_pattern(
            name="xss_data_uri_javascript",
            pattern=r"data:text/javascript",
            severity=ViolationSeverity.HIGH,
            description="XSS - Data URI with JavaScript"
        )
        
        # SVG-based XSS
        self.add_pattern(
            name="xss_svg_element",
            pattern=r"<svg[^>]*>",
            severity=ViolationSeverity.HIGH,
            description="XSS - SVG element"
        )
        
        self.add_pattern(
            name="xss_svg_script",
            pattern=r"<svg[^>]*>.*<script",
            severity=ViolationSeverity.HIGH,
            description="XSS - SVG with script"
        )
        
        # MathML-based XSS
        self.add_pattern(
            name="xss_mathml_element",
            pattern=r"<math[^>]*>",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - MathML element"
        )
        
        # Form-based XSS
        self.add_pattern(
            name="xss_form_action",
            pattern=r"<form[^>]*action\s*=\s*['\"]javascript:",
            severity=ViolationSeverity.HIGH,
            description="XSS - Form action with javascript"
        )
        
        # Input-based XSS
        self.add_pattern(
            name="xss_input_onfocus",
            pattern=r"<input[^>]*onfocus\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - Input with onfocus"
        )
        
        # Image-based XSS
        self.add_pattern(
            name="xss_img_onerror",
            pattern=r"<img[^>]*onerror\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - Image with onerror"
        )
        
        self.add_pattern(
            name="xss_img_src_javascript",
            pattern=r"<img[^>]*src\s*=\s*['\"]javascript:",
            severity=ViolationSeverity.HIGH,
            description="XSS - Image src with javascript"
        )
        
        # Body-based XSS
        self.add_pattern(
            name="xss_body_onload",
            pattern=r"<body[^>]*onload\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - Body with onload"
        )
        
        # Div-based XSS
        self.add_pattern(
            name="xss_div_onclick",
            pattern=r"<div[^>]*onclick\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - Div with onclick"
        )
        
        # Encoded XSS attempts
        self.add_pattern(
            name="xss_encoded_script",
            pattern=r"&#x3C;script|&#60;script|%3Cscript|%3cscript",
            severity=ViolationSeverity.HIGH,
            description="XSS - Encoded script tag"
        )
        
        self.add_pattern(
            name="xss_encoded_javascript",
            pattern=r"&#x6A;avascript|&#106;avascript|%6Aavascript|%6aavascript",
            severity=ViolationSeverity.HIGH,
            description="XSS - Encoded javascript"
        )
        
        # DOM-based XSS patterns
        self.add_pattern(
            name="xss_dom_document_write",
            pattern=r"document\.write\s*\(",
            severity=ViolationSeverity.HIGH,
            description="XSS - DOM document.write"
        )
        
        self.add_pattern(
            name="xss_dom_innerhtml",
            pattern=r"\.innerHTML\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - DOM innerHTML"
        )
        
        self.add_pattern(
            name="xss_dom_outerhtml",
            pattern=r"\.outerHTML\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - DOM outerHTML"
        )
        
        # Filter bypass attempts
        self.add_pattern(
            name="xss_filter_bypass_case",
            pattern=r"[Ss][Cc][Rr][Ii][Pp][Tt]",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Case variation filter bypass"
        )
        
        self.add_pattern(
            name="xss_filter_bypass_space",
            pattern=r"<script\s+",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Space insertion filter bypass"
        )
        
        self.add_pattern(
            name="xss_filter_bypass_tab",
            pattern=r"<script\t+",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Tab insertion filter bypass"
        )
        
        # Advanced XSS techniques
        self.add_pattern(
            name="xss_advanced_unicode",
            pattern=r"\\u003cscript|\\u003c/script",
            severity=ViolationSeverity.HIGH,
            description="XSS - Unicode encoded script tags"
        )
        
        self.add_pattern(
            name="xss_advanced_hex",
            pattern=r"\\x3cscript|\\x3c/script",
            severity=ViolationSeverity.HIGH,
            description="XSS - Hex encoded script tags"
        )
