"""
Model version management and deployment coordination.

Single responsibility: Manage model versions and deployment lifecycle.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status for model versions."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class ModelType(Enum):
    """Model types for versioning."""

    MAPPER = "mapper"
    ANALYST = "analyst"


@dataclass
class ModelCard:
    """Model card with essential information."""

    model_name: str
    model_type: ModelType
    version: str
    created_at: str
    created_by: str

    # Training information
    training_datasets: List[str]
    training_steps: int
    training_epochs: int
    lora_config: Dict[str, Any]

    # Performance metrics
    performance_metrics: Dict[str, Any]
    golden_set_metrics: Optional[Dict[str, Any]] = None

    # Known issues
    known_failure_modes: Optional[List[str]] = None
    limitations: Optional[List[str]] = None

    # Deployment information
    deployment_status: DeploymentStatus = DeploymentStatus.DEVELOPMENT
    canary_percentage: float = 0.0
    baseline_version: Optional[str] = None

    # Metadata
    model_size: Optional[str] = None
    memory_requirements: Optional[str] = None
    inference_latency: Optional[float] = None
    throughput: Optional[float] = None


@dataclass
class KPIMetrics:
    """KPI metrics for deployment evaluation."""

    p95_latency: float
    schema_pass_rate: float
    f1_score: float
    cache_hit_rate: float
    error_rate: float
    throughput: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""

    canary_percentage: float = 5.0
    evaluation_duration: int = 3600  # 1 hour
    kpi_thresholds: Optional[Dict[str, float]] = None
    auto_promote: bool = False

    def __post_init__(self):
        if self.kpi_thresholds is None:
            self.kpi_thresholds = {
                "p95_latency_improvement": 0.1,
                "schema_pass_rate_min": 0.95,
                "f1_score_improvement": 0.02,
                "error_rate_max": 0.01,
            }


class ModelVersionManager:
    """
    Manages model versions and deployments.

    Single responsibility: Model version lifecycle management.
    """

    def __init__(
        self, models_dir: str = "models", registry_file: str = "model_registry.json"
    ):
        """
        Initialize model version manager.

        Args:
            models_dir: Directory for storing models
            registry_file: Registry file name
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.registry_file = self.models_dir / registry_file
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {
            "models": {},
            "deployments": {},
            "last_updated": datetime.now().isoformat(),
        }

    def _save_registry(self) -> None:
        """Save model registry to file."""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def create_model_version(
        self,
        model_name: str,
        model_type: ModelType,
        training_datasets: List[str],
        training_config: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        golden_set_metrics: Dict[str, Any],
        created_by: str = "system",
    ) -> str:
        """Create a new model version."""

        # Generate version number
        version = self._generate_version_number(model_name, model_type)

        # Create model card
        model_card = ModelCard(
            model_name=model_name,
            model_type=model_type,
            version=version,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            training_datasets=training_datasets,
            training_steps=training_config.get("max_steps", 0),
            training_epochs=training_config.get("num_train_epochs", 0),
            lora_config=training_config.get("lora_config", {}),
            performance_metrics=performance_metrics,
            golden_set_metrics=golden_set_metrics,
            known_failure_modes=self._identify_failure_modes(performance_metrics),
            limitations=self._identify_limitations(performance_metrics),
            deployment_status=DeploymentStatus.DEVELOPMENT,
        )

        # Save model card
        model_dir = self.models_dir / f"{model_name}_{version}"
        model_dir.mkdir(exist_ok=True)

        model_card_file = model_dir / "model_card.json"
        with open(model_card_file, "w") as f:
            json.dump(asdict(model_card), f, indent=2)

        # Update registry
        model_key = f"{model_name}@{version}"
        self.registry["models"][model_key] = {
            "model_card": asdict(model_card),
            "model_dir": str(model_dir),
            "created_at": model_card.created_at,
        }

        self._save_registry()

        logger.info("Created model version: %s", model_key)
        return version

    def _generate_version_number(self, model_name: str, model_type: ModelType) -> str:
        """Generate semantic version number."""
        # Get existing versions for this model
        existing_versions = []
        for key in self.registry["models"].keys():
            if key.startswith(f"{model_name}@"):
                version_str = key.split("@")[1]
                existing_versions.append(version_str)

        if not existing_versions:
            return "1.0.0"

        # Simple increment for now
        latest_version = max(existing_versions)
        parts = latest_version.split(".")
        if len(parts) == 3:
            try:
                patch = int(parts[2]) + 1
                return f"{parts[0]}.{parts[1]}.{patch}"
            except ValueError:
                pass

        return "1.0.1"

    def _identify_failure_modes(self, performance_metrics: Dict[str, Any]) -> List[str]:
        """Identify known failure modes from performance metrics."""
        failure_modes = []

        # Check for low performance
        if "f1_score" in performance_metrics and performance_metrics["f1_score"] < 0.8:
            failure_modes.append("Low F1 score on evaluation set")

        if (
            "schema_pass_rate" in performance_metrics
            and performance_metrics["schema_pass_rate"] < 0.9
        ):
            failure_modes.append("Schema validation failures")

        if (
            "latency_p95" in performance_metrics
            and performance_metrics["latency_p95"] > 200
        ):
            failure_modes.append("High latency (>200ms p95)")

        return failure_modes

    def _identify_limitations(self, performance_metrics: Dict[str, Any]) -> List[str]:
        """Identify model limitations from performance metrics."""
        limitations = []

        # Check taxonomy coverage
        if "taxonomy_coverage" in performance_metrics:
            coverage = performance_metrics["taxonomy_coverage"]
            if coverage.get("total_branches", 0) < 20:
                limitations.append("Limited taxonomy coverage")

        # Check confidence distribution
        if "confidence_statistics" in performance_metrics:
            conf_stats = performance_metrics["confidence_statistics"]
            if conf_stats.get("mean", 0) < 0.8:
                limitations.append("Low average confidence scores")

        return limitations

    def get_model_info(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        model_key = f"{model_name}@{version}"
        return self.registry["models"].get(model_key)

    def list_model_versions(self, model_name: str) -> List[str]:
        """List all versions of a model."""
        versions = []
        for key in self.registry["models"].keys():
            if key.startswith(f"{model_name}@"):
                version = key.split("@")[1]
                versions.append(version)

        versions.sort(reverse=True)
        return versions

    def get_deployment_status(self, model_name: str, version: str) -> Optional[str]:
        """Get deployment status of a model version."""
        model_key = f"{model_name}@{version}"
        if model_key in self.registry["deployments"]:
            return self.registry["deployments"][model_key]["status"]
        return None


class DeploymentManager:
    """
    Manages model deployments and A/B testing.

    Single responsibility: Deployment lifecycle management.
    """

    def __init__(self, version_manager: ModelVersionManager):
        """
        Initialize deployment manager.

        Args:
            version_manager: Model version manager instance
        """
        self.version_manager = version_manager

    def deploy_canary(
        self,
        model_name: str,
        version: str,
        canary_config: CanaryConfig,
        baseline_version: Optional[str] = None,
    ) -> bool:
        """Deploy model version as canary."""

        model_key = f"{model_name}@{version}"
        if model_key not in self.version_manager.registry["models"]:
            logger.error("Model %s not found in registry", model_key)
            return False

        # Update model card
        model_info = self.version_manager.registry["models"][model_key]
        model_card = ModelCard(**model_info["model_card"])
        model_card.deployment_status = DeploymentStatus.CANARY
        model_card.canary_percentage = canary_config.canary_percentage
        model_card.baseline_version = baseline_version

        # Save updated model card
        model_dir = Path(model_info["model_dir"])
        model_card_file = model_dir / "model_card.json"
        with open(model_card_file, "w") as f:
            json.dump(asdict(model_card), f, indent=2)

        # Update registry
        model_info["model_card"] = asdict(model_card)
        self.version_manager.registry["deployments"][model_key] = {
            "status": "canary",
            "canary_config": asdict(canary_config),
            "deployed_at": datetime.now().isoformat(),
            "baseline_version": baseline_version,
        }

        self.version_manager._save_registry()

        logger.info(
            "Deployed canary: %s (%s%% traffic)",
            model_key,
            canary_config.canary_percentage,
        )
        return True

    def promote_to_production(
        self, model_name: str, version: str, kpi_metrics: KPIMetrics
    ) -> bool:
        """Promote canary to production based on KPI metrics."""

        model_key = f"{model_name}@{version}"
        if model_key not in self.version_manager.registry["models"]:
            logger.error("Model %s not found in registry", model_key)
            return False

        # Check if model is in canary
        if model_key not in self.version_manager.registry["deployments"]:
            logger.error("Model %s not deployed as canary", model_key)
            return False

        deployment_info = self.version_manager.registry["deployments"][model_key]
        if deployment_info["status"] != "canary":
            logger.error("Model %s is not in canary status", model_key)
            return False

        # Evaluate KPI metrics
        canary_config = CanaryConfig(**deployment_info["canary_config"])
        if not self._evaluate_kpi_metrics(kpi_metrics, canary_config):
            logger.warning(
                f"KPI metrics for {model_key} do not meet promotion criteria"
            )
            return False

        # Promote to production
        model_info = self.version_manager.registry["models"][model_key]
        model_card = ModelCard(**model_info["model_card"])
        model_card.deployment_status = DeploymentStatus.PRODUCTION

        # Save updated model card
        model_dir = Path(model_info["model_dir"])
        model_card_file = model_dir / "model_card.json"
        with open(model_card_file, "w") as f:
            json.dump(asdict(model_card), f, indent=2)

        # Update registry
        model_info["model_card"] = asdict(model_card)
        deployment_info["status"] = "production"
        deployment_info["promoted_at"] = datetime.now().isoformat()
        deployment_info["kpi_metrics"] = asdict(kpi_metrics)

        self.version_manager._save_registry()

        logger.info("Promoted to production: %s", model_key)
        return True

    def _evaluate_kpi_metrics(
        self, kpi_metrics: KPIMetrics, canary_config: CanaryConfig
    ) -> bool:
        """Evaluate KPI metrics against canary configuration thresholds."""
        thresholds = canary_config.kpi_thresholds
        if not thresholds:
            return False

        # Check schema pass rate
        if kpi_metrics.schema_pass_rate < thresholds["schema_pass_rate_min"]:
            return False

        # Check error rate
        if kpi_metrics.error_rate > thresholds["error_rate_max"]:
            return False

        return True

    def create_ab_test(
        self,
        test_name: str,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
    ) -> Dict[str, Any]:
        """Create A/B test configuration."""

        ab_test_config = {
            "test_name": test_name,
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
            "status": "active",
        }

        # Save A/B test configuration
        ab_test_file = self.version_manager.models_dir / f"ab_test_{test_name}.json"
        with open(ab_test_file, "w") as f:
            json.dump(ab_test_config, f, indent=2)

        logger.info(
            "A/B test configuration created",
            test_name=test_name,
            version_a=version_a,
            version_b=version_b,
        )

        return ab_test_config
