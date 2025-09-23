"""Training data generation modules for enhanced domain specificity."""

from .synthetic import SyntheticDataGenerator
from .real_world_collector import RealWorldDataCollector
from .hybrid_generator import HybridTrainingDataGenerator
from .analysis_module_generator import AnalysisModuleDataGenerator
from .advanced_training_generator import AdvancedTrainingDataGenerator
from .evaluation_metrics import ComplianceModelEvaluator
from .models import MapperCanonicalEvent, TrainingExample

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