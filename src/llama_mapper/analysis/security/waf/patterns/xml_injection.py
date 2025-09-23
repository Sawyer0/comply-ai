"""XML injection security patterns."""

from .base import PatternCollection
from ..interfaces import AttackType, ViolationSeverity


class XMLInjectionPatterns(PatternCollection):
    """Collection of XML injection security patterns."""
    
    def __init__(self):
        super().__init__(AttackType.XML_INJECTION)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        # XML DOCTYPE
        self.add_pattern(
            name="xml_doctype",
            pattern=r"<!DOCTYPE",
            severity=ViolationSeverity.MEDIUM,
            description="XML Injection - DOCTYPE"
        )
        
        # XML entities
        self.add_pattern(
            name="xml_entity",
            pattern=r"<!ENTITY",
            severity=ViolationSeverity.MEDIUM,
            description="XML Injection - Entity"
        )
        
        # Entity references
        self.add_pattern(
            name="xml_entity_ref",
            pattern=r"&[a-zA-Z0-9_]+;",
            severity=ViolationSeverity.MEDIUM,
            description="XML Injection - Entity reference"
        )
        
        # CDATA sections
        self.add_pattern(
            name="xml_cdata",
            pattern=r"<![CDATA\[",
            severity=ViolationSeverity.MEDIUM,
            description="XML Injection - CDATA"
        )
        
        # Namespaces
        self.add_pattern(
            name="xml_namespace",
            pattern=r"xmlns:",
            severity=ViolationSeverity.MEDIUM,
            description="XML Injection - Namespace"
        )
        
        # Schema instances
        self.add_pattern(
            name="xml_schema_instance",
            pattern=r"xsi:",
            severity=ViolationSeverity.MEDIUM,
            description="XML Injection - Schema instance"
        )
