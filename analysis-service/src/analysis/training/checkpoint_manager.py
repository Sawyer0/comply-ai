"""
Checkpoint management for Analysis Service training.

Handles saving, loading, and versioning of model checkpoints.
"""

import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..shared_integration import get_shared_logger

logger = get_shared_logger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata."""

    version: str
    model_name: str
    base_model: str
    training_data_version: str
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    created_at: datetime
    created_by: str
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class CheckpointManager:
    """Manages model checkpoints and versions."""

    def __init__(self, base_checkpoint_dir: str = "checkpoints"):
        self.base_checkpoint_dir = base_checkpoint_dir
        self.logger = logger.bind(component="checkpoint_manager")

        # Ensure checkpoint directory exists
        os.makedirs(base_checkpoint_dir, exist_ok=True)

        # Load existing versions
        self.versions = self._load_versions()

    def save_checkpoint(
        self,
        model,
        tokenizer,
        version: str,
        model_name: str,
        base_model: str,
        training_data_version: str,
        performance_metrics: Dict[str, float],
        training_config: Dict[str, Any],
        created_by: str,
        description: str = "",
        tags: List[str] = None,
    ) -> str:
        """
        Save a model checkpoint with version metadata.

        Args:
            model: Trained model to save
            tokenizer: Tokenizer to save
            version: Version identifier
            model_name: Name of the model
            base_model: Base model used for training
            training_data_version: Version of training data used
            performance_metrics: Model performance metrics
            training_config: Training configuration used
            created_by: User who created this version
            description: Optional description
            tags: Optional tags for categorization

        Returns:
            Path to saved checkpoint
        """
        try:
            # Create version metadata
            model_version = ModelVersion(
                version=version,
                model_name=model_name,
                base_model=base_model,
                training_data_version=training_data_version,
                performance_metrics=performance_metrics,
                training_config=training_config,
                created_at=datetime.utcnow(),
                created_by=created_by,
                description=description,
                tags=tags or [],
            )

            # Create checkpoint directory
            checkpoint_path = os.path.join(self.base_checkpoint_dir, version)
            os.makedirs(checkpoint_path, exist_ok=True)

            # Save model and tokenizer
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

            # Save version metadata
            metadata_path = os.path.join(checkpoint_path, "version_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(model_version.to_dict(), f, indent=2)

            # Update versions registry
            self.versions[version] = model_version
            self._save_versions()

            self.logger.info(
                "Checkpoint saved successfully",
                version=version,
                checkpoint_path=checkpoint_path,
                model_name=model_name,
            )

            return checkpoint_path

        except Exception as e:
            self.logger.error(
                "Failed to save checkpoint", version=version, error=str(e)
            )
            raise

    def load_checkpoint(self, version: str) -> Optional[str]:
        """
        Load a checkpoint by version.

        Args:
            version: Version to load

        Returns:
            Path to checkpoint if found, None otherwise
        """
        checkpoint_path = os.path.join(self.base_checkpoint_dir, version)

        if not os.path.exists(checkpoint_path):
            self.logger.warning("Checkpoint not found", version=version)
            return None

        # Verify metadata exists
        metadata_path = os.path.join(checkpoint_path, "version_metadata.json")
        if not os.path.exists(metadata_path):
            self.logger.warning("Checkpoint metadata not found", version=version)
            return None

        self.logger.info("Checkpoint loaded", version=version, path=checkpoint_path)
        return checkpoint_path

    def get_version_metadata(self, version: str) -> Optional[ModelVersion]:
        """Get metadata for a specific version."""
        return self.versions.get(version)

    def list_versions(
        self, model_name: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> List[ModelVersion]:
        """
        List available versions with optional filtering.

        Args:
            model_name: Filter by model name
            tags: Filter by tags (must have all specified tags)

        Returns:
            List of matching model versions
        """
        versions = list(self.versions.values())

        # Filter by model name
        if model_name:
            versions = [v for v in versions if v.model_name == model_name]

        # Filter by tags
        if tags:
            versions = [v for v in versions if all(tag in v.tags for tag in tags)]

        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        return versions

    def get_latest_version(
        self, model_name: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> Optional[ModelVersion]:
        """Get the latest version matching criteria."""
        versions = self.list_versions(model_name=model_name, tags=tags)
        return versions[0] if versions else None

    def delete_checkpoint(self, version: str) -> bool:
        """
        Delete a checkpoint and its metadata.

        Args:
            version: Version to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            checkpoint_path = os.path.join(self.base_checkpoint_dir, version)

            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)

            # Remove from versions registry
            if version in self.versions:
                del self.versions[version]
                self._save_versions()

            self.logger.info("Checkpoint deleted", version=version)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to delete checkpoint", version=version, error=str(e)
            )
            return False

    def cleanup_old_checkpoints(
        self, keep_latest: int = 5, model_name: Optional[str] = None
    ) -> int:
        """
        Clean up old checkpoints, keeping only the latest N versions.

        Args:
            keep_latest: Number of latest versions to keep
            model_name: Optional model name filter

        Returns:
            Number of checkpoints deleted
        """
        versions = self.list_versions(model_name=model_name)

        if len(versions) <= keep_latest:
            return 0

        # Delete older versions
        versions_to_delete = versions[keep_latest:]
        deleted_count = 0

        for version in versions_to_delete:
            if self.delete_checkpoint(version.version):
                deleted_count += 1

        self.logger.info(
            "Checkpoint cleanup completed",
            deleted_count=deleted_count,
            remaining_count=len(versions) - deleted_count,
        )

        return deleted_count

    def export_version_metadata(self, output_path: str) -> None:
        """Export all version metadata to a file."""
        try:
            metadata = {
                version: model_version.to_dict()
                for version, model_version in self.versions.items()
            }

            with open(output_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info("Version metadata exported", output_path=output_path)

        except Exception as e:
            self.logger.error("Failed to export metadata", error=str(e))
            raise

    def _load_versions(self) -> Dict[str, ModelVersion]:
        """Load versions registry from disk."""
        versions_file = os.path.join(self.base_checkpoint_dir, "versions.json")

        if not os.path.exists(versions_file):
            return {}

        try:
            with open(versions_file, "r") as f:
                data = json.load(f)

            versions = {}
            for version_id, version_data in data.items():
                versions[version_id] = ModelVersion.from_dict(version_data)

            return versions

        except Exception as e:
            self.logger.warning("Failed to load versions registry", error=str(e))
            return {}

    def _save_versions(self) -> None:
        """Save versions registry to disk."""
        versions_file = os.path.join(self.base_checkpoint_dir, "versions.json")

        try:
            data = {
                version: model_version.to_dict()
                for version, model_version in self.versions.items()
            }

            with open(versions_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error("Failed to save versions registry", error=str(e))
            raise
