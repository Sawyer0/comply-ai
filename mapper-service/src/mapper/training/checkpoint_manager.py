"""
Checkpoint management for model versioning and rollback.

Single responsibility: Manage model checkpoints and versions.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a versioned model checkpoint."""

    def __init__(
        self,
        version: str,
        checkpoint_path: str,
        metadata: Dict[str, Any],
        created_at: datetime,
    ):
        """
        Initialize model version.

        Args:
            version: Version identifier
            checkpoint_path: Path to checkpoint directory
            metadata: Version metadata
            created_at: Creation timestamp
        """
        self.version = version
        self.checkpoint_path = checkpoint_path
        self.metadata = metadata
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "checkpoint_path": self.checkpoint_path,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            checkpoint_path=data["checkpoint_path"],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class CheckpointManager:
    """
    Manages model checkpoint versioning and rollback.

    Single responsibility: Checkpoint storage and retrieval operations.
    """

    def __init__(
        self,
        base_dir: str = "./model_checkpoints",
        version_prefix: str = "mapper-lora",
        max_versions: int = 10,
    ):
        """
        Initialize checkpoint manager.

        Args:
            base_dir: Base directory for storing checkpoints
            version_prefix: Prefix for version tags
            max_versions: Maximum number of versions to keep
        """
        self.base_dir = Path(base_dir)
        self.version_prefix = version_prefix
        self.max_versions = max_versions

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.base_dir / "versions.json"

        # Load existing versions
        self.versions: Dict[str, ModelVersion] = self._load_versions()

        logger.info(
            "CheckpointManager initialized",
            base_dir=str(self.base_dir),
            version_prefix=version_prefix,
            existing_versions=len(self.versions),
        )

    def _load_versions(self) -> Dict[str, ModelVersion]:
        """Load version registry from disk."""
        if not self.versions_file.exists():
            return {}

        try:
            with open(self.versions_file, "r") as f:
                data = json.load(f)

            versions = {}
            for version_id, version_data in data.items():
                versions[version_id] = ModelVersion.from_dict(version_data)

            logger.info("Loaded version registry", num_versions=len(versions))
            return versions

        except Exception as e:
            logger.error("Failed to load version registry", error=str(e))
            return {}

    def _save_versions(self) -> None:
        """Save version registry to disk."""
        try:
            data = {
                version_id: version.to_dict()
                for version_id, version in self.versions.items()
            }

            with open(self.versions_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Saved version registry", num_versions=len(self.versions))

        except Exception as e:
            logger.error("Failed to save version registry", error=str(e))

    def _generate_version_id(
        self, major: int = 1, minor: int = 0, patch: int = 0
    ) -> str:
        """Generate version ID with auto-increment."""
        if not self.versions:
            return f"{self.version_prefix}@v{major}.{minor}.{patch}"

        # Find the highest version number
        max_version = (0, 0, 0)
        for version_id in self.versions:
            if "@v" in version_id:
                try:
                    version_part = version_id.split("@v")[1]
                    parts = version_part.split(".")
                    if len(parts) == 3:
                        v_tuple = (int(parts[0]), int(parts[1]), int(parts[2]))
                        max_version = max(max_version, v_tuple)
                except (ValueError, IndexError):
                    continue

        # Increment patch version
        new_version = (max_version[0], max_version[1], max_version[2] + 1)
        return (
            f"{self.version_prefix}@v{new_version[0]}.{new_version[1]}.{new_version[2]}"
        )

    def save_checkpoint(
        self,
        model: Any,
        tokenizer: Any,
        training_metrics: Dict[str, Any],
        version_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Save model checkpoint with version information.

        Args:
            model: Model to save
            tokenizer: Associated tokenizer
            training_metrics: Training metrics and results
            version_id: Optional version identifier
            metadata: Additional metadata
            tags: Optional tags for the version

        Returns:
            Version identifier of the saved checkpoint
        """
        if version_id is None:
            version_id = self._generate_version_id()

        # Create checkpoint directory
        checkpoint_dir = self.base_dir / version_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Saving checkpoint",
            version_id=version_id,
            checkpoint_dir=str(checkpoint_dir),
        )

        try:
            # Save model
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(checkpoint_dir / "model")

            # Save tokenizer
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

            # Prepare metadata
            full_metadata = {
                "training_metrics": training_metrics,
                "model_config": (
                    getattr(model, "config", {}).to_dict()
                    if hasattr(getattr(model, "config", {}), "to_dict")
                    else {}
                ),
                "peft_config": (
                    getattr(model, "peft_config", {})
                    if hasattr(model, "peft_config")
                    else {}
                ),
                "tags": tags or [],
                "created_by": "CheckpointManager",
                **(metadata or {}),
            }

            # Save metadata
            metadata_file = checkpoint_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(full_metadata, f, indent=2)

            # Create version entry
            model_version = ModelVersion(
                version=version_id,
                checkpoint_path=str(checkpoint_dir),
                metadata=full_metadata,
                created_at=datetime.now(),
            )

            # Add to registry
            self.versions[version_id] = model_version
            self._save_versions()

            # Cleanup old versions if needed
            self._cleanup_old_versions()

            logger.info(
                "Checkpoint saved successfully",
                version_id=version_id,
                checkpoint_size_mb=self._get_directory_size(checkpoint_dir)
                / (1024 * 1024),
            )

            return version_id

        except Exception as e:
            logger.error(
                "Failed to save checkpoint", version_id=version_id, error=str(e)
            )
            # Cleanup failed checkpoint
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            raise

    def load_checkpoint(
        self, version_id: str, base_model: Optional[Any] = None
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load model checkpoint by version ID.

        Args:
            version_id: Version identifier to load
            base_model: Optional base model

        Returns:
            Tuple of (model, tokenizer, metadata)
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        version = self.versions[version_id]
        checkpoint_dir = Path(version.checkpoint_path)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        logger.info(
            "Loading checkpoint",
            version_id=version_id,
            checkpoint_dir=str(checkpoint_dir),
        )

        try:
            # Load tokenizer
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
            except ImportError:
                logger.warning("transformers not available, tokenizer not loaded")
                tokenizer = None

            # Load model
            model = None
            if base_model is not None:
                # Load PEFT adapter onto existing base model
                try:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(
                        base_model, checkpoint_dir / "model"
                    )
                except ImportError:
                    logger.warning("PEFT not available, using base model")
                    model = base_model
            else:
                # Load PEFT model with base model
                try:
                    from peft import AutoPeftModelForCausalLM

                    model = AutoPeftModelForCausalLM.from_pretrained(
                        checkpoint_dir / "model"
                    )
                except ImportError:
                    logger.warning("PEFT not available, model not loaded")

            # Load metadata
            metadata_file = checkpoint_dir / "metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

            logger.info("Checkpoint loaded successfully", version_id=version_id)

            return model, tokenizer, metadata

        except Exception as e:
            logger.error(
                "Failed to load checkpoint", version_id=version_id, error=str(e)
            )
            raise

    def list_versions(self, tags: Optional[List[str]] = None) -> List[ModelVersion]:
        """
        List available model versions.

        Args:
            tags: Optional tags to filter by

        Returns:
            List of model versions, sorted by creation time
        """
        versions = list(self.versions.values())

        # Filter by tags if specified
        if tags:
            filtered_versions = []
            for version in versions:
                version_tags = version.metadata.get("tags", [])
                if any(tag in version_tags for tag in tags):
                    filtered_versions.append(version)
            versions = filtered_versions

        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        return versions

    def get_latest_version(
        self, tags: Optional[List[str]] = None
    ) -> Optional[ModelVersion]:
        """Get the latest model version."""
        versions = self.list_versions(tags=tags)
        return versions[0] if versions else None

    def delete_version(self, version_id: str) -> bool:
        """Delete a model version."""
        if version_id not in self.versions:
            logger.warning("Version not found for deletion", version_id=version_id)
            return False

        version = self.versions[version_id]
        checkpoint_dir = Path(version.checkpoint_path)

        try:
            # Remove checkpoint directory
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)

            # Remove from registry
            del self.versions[version_id]
            self._save_versions()

            logger.info("Version deleted successfully", version_id=version_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to delete version", version_id=version_id, error=str(e)
            )
            return False

    def _cleanup_old_versions(self) -> None:
        """Remove old versions if exceeding max_versions limit."""
        if len(self.versions) <= self.max_versions:
            return

        # Sort versions by creation time (oldest first)
        sorted_versions = sorted(self.versions.values(), key=lambda v: v.created_at)

        # Remove oldest versions
        versions_to_remove = sorted_versions[: len(self.versions) - self.max_versions]

        for version in versions_to_remove:
            logger.info("Cleaning up old version", version_id=version.version)
            self.delete_version(version.version)

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
