"""
Model version management and deployment for Analysis Service.

Handles model versioning, A/B testing, and deployment coordination.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..shared_integration import get_shared_logger

logger = get_shared_logger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""

    model_version: str
    deployment_type: str  # "blue_green", "canary", "rolling"
    traffic_split: float = 1.0  # Fraction of traffic for this version
    health_check_endpoint: str = "/health"
    rollback_threshold: float = 0.95  # Minimum success rate before rollback
    monitoring_duration_minutes: int = 60
    auto_promote: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "deployment_type": self.deployment_type,
            "traffic_split": self.traffic_split,
            "health_check_endpoint": self.health_check_endpoint,
            "rollback_threshold": self.rollback_threshold,
            "monitoring_duration_minutes": self.monitoring_duration_minutes,
            "auto_promote": self.auto_promote,
        }


class ModelVersionManager:
    """Manages model versions and deployment strategies."""

    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = registry_path
        self.logger = logger.bind(component="model_version_manager")

        # Load existing registry
        self.registry = self._load_registry()

    def register_model(
        self,
        version: str,
        model_path: str,
        base_model: str,
        training_data_version: str,
        performance_metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a new model version.

        Args:
            version: Version identifier
            model_path: Path to the trained model
            base_model: Base model used for training
            training_data_version: Version of training data
            performance_metrics: Model performance metrics
            metadata: Additional metadata
        """
        model_info = {
            "version": version,
            "model_path": model_path,
            "base_model": base_model,
            "training_data_version": training_data_version,
            "performance_metrics": performance_metrics,
            "metadata": metadata or {},
            "registered_at": datetime.utcnow().isoformat(),
            "status": "registered",
        }

        self.registry[version] = model_info
        self._save_registry()

        self.logger.info(
            "Model version registered",
            version=version,
            model_path=model_path,
            metrics=performance_metrics,
        )

    def get_model_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model version."""
        return self.registry.get(version)

    def list_models(
        self, status: Optional[str] = None, min_accuracy: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        List registered models with optional filtering.

        Args:
            status: Filter by status
            min_accuracy: Minimum accuracy threshold

        Returns:
            List of model information dictionaries
        """
        models = list(self.registry.values())

        # Filter by status
        if status:
            models = [m for m in models if m.get("status") == status]

        # Filter by accuracy
        if min_accuracy is not None:
            models = [
                m
                for m in models
                if m.get("performance_metrics", {}).get("accuracy", 0) >= min_accuracy
            ]

        # Sort by registration date (newest first)
        models.sort(key=lambda m: m.get("registered_at", ""), reverse=True)

        return models

    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """Get the currently deployed production model."""
        production_models = [
            m for m in self.registry.values() if m.get("status") == "production"
        ]

        if not production_models:
            return None

        # Return the most recently deployed production model
        return max(production_models, key=lambda m: m.get("deployed_at", ""))

    def promote_to_production(self, version: str) -> bool:
        """
        Promote a model version to production.

        Args:
            version: Version to promote

        Returns:
            True if successful, False otherwise
        """
        if version not in self.registry:
            self.logger.error("Model version not found", version=version)
            return False

        try:
            # Demote current production model
            current_production = self.get_production_model()
            if current_production:
                current_version = current_production["version"]
                self.registry[current_version]["status"] = "archived"
                self.registry[current_version][
                    "archived_at"
                ] = datetime.utcnow().isoformat()

            # Promote new model
            self.registry[version]["status"] = "production"
            self.registry[version]["deployed_at"] = datetime.utcnow().isoformat()

            self._save_registry()

            self.logger.info(
                "Model promoted to production",
                version=version,
                previous_production=(
                    current_production["version"] if current_production else None
                ),
            )

            return True

        except Exception as e:
            self.logger.error("Failed to promote model", version=version, error=str(e))
            return False

    def rollback_production(self) -> Optional[str]:
        """
        Rollback to the previous production model.

        Returns:
            Version of the rolled back model, or None if no rollback available
        """
        # Find archived models (previously in production)
        archived_models = [
            m
            for m in self.registry.values()
            if m.get("status") == "archived" and "deployed_at" in m
        ]

        if not archived_models:
            self.logger.warning("No archived models available for rollback")
            return None

        # Get the most recently archived model
        rollback_model = max(archived_models, key=lambda m: m.get("archived_at", ""))

        rollback_version = rollback_model["version"]

        # Promote the rollback model
        if self.promote_to_production(rollback_version):
            self.logger.info("Rollback completed", rollback_version=rollback_version)
            return rollback_version

        return None

    def update_model_status(
        self, version: str, status: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update the status of a model version.

        Args:
            version: Model version
            status: New status
            metadata: Optional additional metadata

        Returns:
            True if successful, False otherwise
        """
        if version not in self.registry:
            self.logger.error("Model version not found", version=version)
            return False

        self.registry[version]["status"] = status
        self.registry[version]["status_updated_at"] = datetime.utcnow().isoformat()

        if metadata:
            self.registry[version]["metadata"].update(metadata)

        self._save_registry()

        self.logger.info("Model status updated", version=version, status=status)
        return True

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load model registry from disk."""
        if not os.path.exists(self.registry_path):
            return {}

        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning("Failed to load model registry", error=str(e))
            return {}

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            self.logger.error("Failed to save model registry", error=str(e))
            raise


class DeploymentManager:
    """Manages model deployments and A/B testing."""

    def __init__(self, version_manager: ModelVersionManager):
        self.version_manager = version_manager
        self.logger = logger.bind(component="deployment_manager")
        self.active_deployments: Dict[str, DeploymentConfig] = {}

    def deploy_model(self, version: str, deployment_config: DeploymentConfig) -> bool:
        """
        Deploy a model version with specified configuration.

        Args:
            version: Model version to deploy
            deployment_config: Deployment configuration

        Returns:
            True if deployment initiated successfully
        """
        model_info = self.version_manager.get_model_info(version)
        if not model_info:
            self.logger.error("Model version not found", version=version)
            return False

        try:
            # Validate deployment configuration
            if not self._validate_deployment_config(deployment_config):
                return False

            # Execute deployment based on type
            if deployment_config.deployment_type == "blue_green":
                success = self._deploy_blue_green(version, deployment_config)
            elif deployment_config.deployment_type == "canary":
                success = self._deploy_canary(version, deployment_config)
            elif deployment_config.deployment_type == "rolling":
                success = self._deploy_rolling(version, deployment_config)
            else:
                self.logger.error(
                    "Unknown deployment type", type=deployment_config.deployment_type
                )
                return False

            if success:
                self.active_deployments[version] = deployment_config
                self.version_manager.update_model_status(
                    version,
                    "deploying",
                    {"deployment_config": deployment_config.to_dict()},
                )

            return success

        except Exception as e:
            self.logger.error("Deployment failed", version=version, error=str(e))
            return False

    def _validate_deployment_config(self, config: DeploymentConfig) -> bool:
        """Validate deployment configuration."""
        if config.traffic_split < 0 or config.traffic_split > 1:
            self.logger.error(
                "Invalid traffic split", traffic_split=config.traffic_split
            )
            return False

        if config.rollback_threshold < 0 or config.rollback_threshold > 1:
            self.logger.error(
                "Invalid rollback threshold", threshold=config.rollback_threshold
            )
            return False

        return True

    def _deploy_blue_green(self, version: str, config: DeploymentConfig) -> bool:
        """Execute blue-green deployment."""
        self.logger.info("Starting blue-green deployment", version=version)

        # In a real implementation, this would:
        # 1. Deploy to green environment
        # 2. Run health checks
        # 3. Switch traffic from blue to green
        # 4. Keep blue as rollback option

        # For now, simulate successful deployment
        return True

    def _deploy_canary(self, version: str, config: DeploymentConfig) -> bool:
        """Execute canary deployment."""
        self.logger.info(
            "Starting canary deployment",
            version=version,
            traffic_split=config.traffic_split,
        )

        # In a real implementation, this would:
        # 1. Deploy canary version
        # 2. Route specified traffic percentage to canary
        # 3. Monitor metrics
        # 4. Gradually increase traffic or rollback

        return True

    def _deploy_rolling(self, version: str, config: DeploymentConfig) -> bool:
        """Execute rolling deployment."""
        self.logger.info("Starting rolling deployment", version=version)

        # In a real implementation, this would:
        # 1. Replace instances one by one
        # 2. Health check each instance
        # 3. Continue or rollback based on health

        return True

    def get_deployment_status(self, version: str) -> Optional[Dict[str, Any]]:
        """Get status of a deployment."""
        if version not in self.active_deployments:
            return None

        config = self.active_deployments[version]
        model_info = self.version_manager.get_model_info(version)

        return {
            "version": version,
            "deployment_config": config.to_dict(),
            "model_status": model_info.get("status") if model_info else "unknown",
            "deployment_time": model_info.get("deployed_at") if model_info else None,
        }

    def list_active_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments."""
        return [
            self.get_deployment_status(version)
            for version in self.active_deployments.keys()
        ]
