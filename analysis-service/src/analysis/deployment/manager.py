"""
Deployment management for Analysis Service.

Handles model deployments, version management, and rollbacks.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..shared_integration import get_shared_logger
from ..serving.model_server import ModelManager, ModelConfig

logger = get_shared_logger(__name__)


class DeploymentStatus(Enum):
    """Deployment status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentRecord:
    """Record of a deployment."""

    deployment_id: str
    version: str
    model_path: Optional[str]
    config_updates: Optional[Dict[str, Any]]
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None


class DeploymentManager:
    """Manages deployments for the Analysis Service."""

    def __init__(self):
        self.logger = logger.bind(component="deployment_manager")
        self.model_manager = ModelManager()

        # Deployment tracking
        self.current_deployment: Optional[DeploymentRecord] = None
        self.deployment_history: List[DeploymentRecord] = []
        self.current_version = "1.0.0"

    async def deploy_version(
        self,
        version: str,
        model_path: Optional[str] = None,
        config_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Deploy a new version."""
        try:
            deployment_id = f"deploy_{datetime.utcnow().timestamp()}"

            self.logger.info(
                "Starting deployment",
                deployment_id=deployment_id,
                version=version,
                model_path=model_path,
            )

            # Create deployment record
            deployment = DeploymentRecord(
                deployment_id=deployment_id,
                version=version,
                model_path=model_path,
                config_updates=config_updates,
                status=DeploymentStatus.PENDING,
                start_time=datetime.utcnow(),
            )

            self.current_deployment = deployment
            deployment.status = DeploymentStatus.IN_PROGRESS

            # Deploy new model if provided
            if model_path:
                success = await self._deploy_model(version, model_path)
                if not success:
                    deployment.status = DeploymentStatus.FAILED
                    deployment.error_message = "Model deployment failed"
                    deployment.end_time = datetime.utcnow()
                    self.deployment_history.append(deployment)
                    return {
                        "success": False,
                        "deployment_id": deployment_id,
                        "error": "Model deployment failed",
                    }

            # Apply configuration updates
            if config_updates:
                await self._apply_config_updates(config_updates)

            # Complete deployment
            deployment.status = DeploymentStatus.COMPLETED
            deployment.end_time = datetime.utcnow()
            self.current_version = version

            # Add to history
            self.deployment_history.append(deployment)
            self.current_deployment = None

            self.logger.info(
                "Deployment completed successfully",
                deployment_id=deployment_id,
                version=version,
            )

            return {
                "success": True,
                "deployment_id": deployment_id,
                "version": version,
                "status": deployment.status.value,
            }

        except Exception as e:
            self.logger.error("Deployment failed", error=str(e))

            if self.current_deployment:
                self.current_deployment.status = DeploymentStatus.FAILED
                self.current_deployment.error_message = str(e)
                self.current_deployment.end_time = datetime.utcnow()
                self.deployment_history.append(self.current_deployment)
                self.current_deployment = None

            return {"success": False, "error": str(e)}

    async def rollback_deployment(
        self, deployment_id: Optional[str] = None, reason: str = "Manual rollback"
    ) -> Dict[str, Any]:
        """Rollback a deployment."""
        try:
            # Find deployment to rollback
            if deployment_id:
                deployment = next(
                    (
                        d
                        for d in self.deployment_history
                        if d.deployment_id == deployment_id
                    ),
                    None,
                )
                if not deployment:
                    return {
                        "success": False,
                        "error": f"Deployment {deployment_id} not found",
                    }
            else:
                # Rollback latest deployment
                if not self.deployment_history:
                    return {"success": False, "error": "No deployments to rollback"}
                deployment = self.deployment_history[-1]

            self.logger.info(
                "Starting rollback",
                deployment_id=deployment.deployment_id,
                reason=reason,
            )

            # Find previous successful deployment
            previous_deployment = None
            for i in range(len(self.deployment_history) - 1, -1, -1):
                if (
                    self.deployment_history[i].deployment_id != deployment.deployment_id
                    and self.deployment_history[i].status == DeploymentStatus.COMPLETED
                ):
                    previous_deployment = self.deployment_history[i]
                    break

            if not previous_deployment:
                return {
                    "success": False,
                    "error": "No previous successful deployment found",
                }

            # Perform rollback
            if previous_deployment.model_path:
                success = await self._deploy_model(
                    previous_deployment.version, previous_deployment.model_path
                )
                if not success:
                    return {
                        "success": False,
                        "error": "Rollback model deployment failed",
                    }

            # Update deployment status
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.rollback_reason = reason
            self.current_version = previous_deployment.version

            self.logger.info(
                "Rollback completed successfully",
                deployment_id=deployment.deployment_id,
                rolled_back_to=previous_deployment.version,
            )

            return {
                "success": True,
                "deployment_id": deployment.deployment_id,
                "rolled_back_to": previous_deployment.version,
                "reason": reason,
            }

        except Exception as e:
            self.logger.error("Rollback failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        status = {
            "current_version": self.current_version,
            "deployment_in_progress": self.current_deployment is not None,
            "total_deployments": len(self.deployment_history),
        }

        if self.current_deployment:
            status["current_deployment"] = {
                "deployment_id": self.current_deployment.deployment_id,
                "version": self.current_deployment.version,
                "status": self.current_deployment.status.value,
                "start_time": self.current_deployment.start_time.isoformat(),
            }

        # Recent deployment history
        recent_deployments = self.deployment_history[-5:]  # Last 5 deployments
        status["recent_deployments"] = [
            {
                "deployment_id": d.deployment_id,
                "version": d.version,
                "status": d.status.value,
                "start_time": d.start_time.isoformat(),
                "end_time": d.end_time.isoformat() if d.end_time else None,
                "error_message": d.error_message,
            }
            for d in recent_deployments
        ]

        return status

    async def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments."""
        return [
            {
                "deployment_id": d.deployment_id,
                "version": d.version,
                "status": d.status.value,
                "start_time": d.start_time.isoformat(),
                "end_time": d.end_time.isoformat() if d.end_time else None,
                "model_path": d.model_path,
                "error_message": d.error_message,
                "rollback_reason": d.rollback_reason,
            }
            for d in self.deployment_history
        ]

    async def _deploy_model(self, version: str, model_path: str) -> bool:
        """Deploy a model."""
        try:
            # Create model configuration
            config = ModelConfig(
                model_name=f"phi3_analysis_{version}",
                model_path=model_path,
                max_tokens=2048,
                temperature=0.1,
                device="cpu",
            )

            # Add model to manager
            success = await self.model_manager.add_model(
                f"analysis_model_{version}", config
            )

            if success:
                self.logger.info("Model deployed successfully", version=version)
            else:
                self.logger.error("Model deployment failed", version=version)

            return success

        except Exception as e:
            self.logger.error("Model deployment error", version=version, error=str(e))
            return False

    async def _apply_config_updates(self, config_updates: Dict[str, Any]) -> None:
        """Apply configuration updates."""
        try:
            self.logger.info("Applying configuration updates", updates=config_updates)

            # In a real implementation, this would update service configuration
            # For now, just log the updates

            for key, value in config_updates.items():
                self.logger.info(f"Config update: {key} = {value}")

        except Exception as e:
            self.logger.error("Configuration update failed", error=str(e))
            raise
