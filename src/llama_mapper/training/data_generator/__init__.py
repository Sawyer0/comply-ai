"""Training data generation modules for enhanced domain specificity."""

from .advanced_training_generator import AdvancedTrainingDataGenerator
from .analysis_module_generator import AnalysisModuleDataGenerator
from .evaluation_metrics import ComplianceModelEvaluator
from .hybrid_generator import HybridTrainingDataGenerator
from .models import MapperCanonicalEvent, TrainingExample
from .real_world_collector import RealWorldDataCollector
from .synthetic import SyntheticDataGenerator

# Add missing classes for backward compatibility
TrainingDataGenerator = AdvancedTrainingDataGenerator
DatasetValidator = ComplianceModelEvaluator

__all__ = [
    "SyntheticDataGenerator",
    "RealWorldDataCollector",
    "HybridTrainingDataGenerator",
    "AnalysisModuleDataGenerator",
    "AdvancedTrainingDataGenerator",
    "ComplianceModelEvaluator",
    "MapperCanonicalEvent",
    "TrainingExample",
    # Backward compatibility aliases
    "TrainingDataGenerator",
    "DatasetValidator",
]
