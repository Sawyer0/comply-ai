"""
Training infrastructure for the Analysis Service.

Single responsibility: Training pipeline coordination and management for Phi-3-Mini
models used in risk assessment and compliance analysis.

Note: Some components require ML dependencies. Install with:
pip install -r requirements-training.txt
"""

# Import with error handling for optional ML dependencies
try:
    from .trainer import Phi3Trainer, TrainingConfig

    _TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import trainer: {e}")
    Phi3Trainer = None
    TrainingConfig = None
    _TRAINER_AVAILABLE = False

try:
    from .checkpoint_manager import CheckpointManager, ModelVersion

    _CHECKPOINT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import checkpoint_manager: {e}")
    CheckpointManager = None
    ModelVersion = None
    _CHECKPOINT_AVAILABLE = False

try:
    from .model_loader import ModelLoader

    _MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import model_loader: {e}")
    ModelLoader = None
    _MODEL_LOADER_AVAILABLE = False

try:
    from .version_manager import ModelVersionManager, DeploymentManager

    _VERSION_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import version_manager: {e}")
    ModelVersionManager = None
    DeploymentManager = None
    _VERSION_MANAGER_AVAILABLE = False

try:
    from .data_processor import AnalysisDataProcessor, RiskAssessmentDataset

    _DATA_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import data_processor: {e}")
    AnalysisDataProcessor = None
    RiskAssessmentDataset = None
    _DATA_PROCESSOR_AVAILABLE = False

# Build __all__ list dynamically based on successful imports
__all__ = []

if _TRAINER_AVAILABLE:
    __all__.extend(["Phi3Trainer", "TrainingConfig"])

if _CHECKPOINT_AVAILABLE:
    __all__.extend(["CheckpointManager", "ModelVersion"])

if _MODEL_LOADER_AVAILABLE:
    __all__.extend(["ModelLoader"])

if _VERSION_MANAGER_AVAILABLE:
    __all__.extend(["ModelVersionManager", "DeploymentManager"])

if _DATA_PROCESSOR_AVAILABLE:
    __all__.extend(["AnalysisDataProcessor", "RiskAssessmentDataset"])
