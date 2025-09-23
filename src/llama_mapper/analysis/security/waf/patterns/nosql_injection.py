"""NoSQL injection security patterns."""

from .base import PatternCollection
from ..interfaces import AttackType, ViolationSeverity


class NoSQLInjectionPatterns(PatternCollection):
    """Collection of NoSQL injection security patterns."""
    
    def __init__(self):
        super().__init__(AttackType.NO_SQL_INJECTION)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        # MongoDB operators
        self.add_pattern(
            name="nosql_where",
            pattern=r"\$where",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - MongoDB where"
        )
        
        self.add_pattern(
            name="nosql_ne",
            pattern=r"\$ne",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - Not equal"
        )
        
        self.add_pattern(
            name="nosql_comparison",
            pattern=r"\$gt|\$gte|\$lt|\$lte",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - Comparison operators"
        )
        
        self.add_pattern(
            name="nosql_in_nin",
            pattern=r"\$in|\$nin",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - In/not in"
        )
        
        self.add_pattern(
            name="nosql_regex",
            pattern=r"\$regex",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - Regex"
        )
        
        self.add_pattern(
            name="nosql_exists",
            pattern=r"\$exists",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - Exists"
        )
        
        self.add_pattern(
            name="nosql_or_and",
            pattern=r"\$or|\$and",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - Boolean operators"
        )
        
        self.add_pattern(
            name="nosql_not",
            pattern=r"\$not",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - Not operator"
        )
