"""
Generic version management for taxonomy and schema resources.

Single responsibility: Handle versioning logic for any resource type.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Generic

from .base_models import VersionInfo, ChangeType, CompatibilityCheck, CompatibilityLevel

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VersionManager(Generic[T]):
    """
    Generic version manager for any resource type.

    Single responsibility: Manage versions, checksums, and persistence.
    """

    def __init__(self, resource_name: str, storage_path: Optional[Path] = None):
        """
        Initialize version manager.

        Args:
            resource_name: Name of the resource being versioned
            storage_path: Path to store version information
        """
        self.resource_name = resource_name
        self.storage_path = storage_path or Path(f"config/{resource_name}")
        self.current_version: Optional[str] = None
        self.version_history: List[VersionInfo] = []

        self._load_version_info()

    def _load_version_info(self) -> None:
        """Load version information from storage."""
        version_file = self.storage_path / "version.json"

        if not version_file.exists():
            return

        try:
            with open(version_file, "r") as f:
                data = json.load(f)
                self.current_version = data.get("current_version")

                for version_data in data.get("history", []):
                    version_info = VersionInfo(
                        version=version_data["version"],
                        created_at=datetime.fromisoformat(version_data["created_at"]),
                        created_by=version_data["created_by"],
                        change_type=ChangeType(version_data["change_type"]),
                        changes=version_data.get("changes", []),
                        backward_compatible=version_data.get(
                            "backward_compatible", True
                        ),
                        migration_notes=version_data.get("migration_notes"),
                        checksum=version_data.get("checksum"),
                    )
                    self.version_history.append(version_info)

        except Exception as e:
            logger.error(
                "Failed to load version info for %s: %s", self.resource_name, str(e)
            )

    def _save_version_info(self) -> None:
        """Save version information to storage."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)

            data = {
                "current_version": self.current_version,
                "history": [
                    {
                        "version": v.version,
                        "created_at": v.created_at.isoformat(),
                        "created_by": v.created_by,
                        "change_type": v.change_type.value,
                        "changes": v.changes,
                        "backward_compatible": v.backward_compatible,
                        "migration_notes": v.migration_notes,
                        "checksum": v.checksum,
                    }
                    for v in self.version_history
                ],
            }

            version_file = self.storage_path / "version.json"
            with open(version_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(
                "Failed to save version info for %s: %s", self.resource_name, str(e)
            )
            raise

    def create_version(
        self,
        change_type: ChangeType,
        changes: List[str],
        created_by: str,
        checksum: str,
        migration_notes: Optional[str] = None,
    ) -> str:
        """
        Create a new version.

        Args:
            change_type: Type of change
            changes: List of changes made
            created_by: Who created this version
            checksum: Checksum of the resource
            migration_notes: Optional migration notes

        Returns:
            str: New version number
        """
        new_version_str = self._increment_version(
            self.current_version or "1.0.0", change_type
        )

        version_info = VersionInfo(
            version=new_version_str,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            change_type=change_type,
            changes=changes,
            backward_compatible=change_type != ChangeType.MAJOR,
            migration_notes=migration_notes,
            checksum=checksum,
        )

        self.version_history.append(version_info)
        self.current_version = new_version_str

        self._save_version_info()

        logger.info("Created %s version %s", self.resource_name, new_version_str)
        return new_version_str

    def get_version_info(self, version: Optional[str] = None) -> Optional[VersionInfo]:
        """Get version information."""
        if version is None:
            version = self.current_version

        for v in self.version_history:
            if v.version == version:
                return v
        return None

    def get_all_versions(self) -> List[str]:
        """Get all version numbers sorted."""
        return sorted(
            [v.version for v in self.version_history], key=self._parse_version
        )

    def _parse_version(self, version: str) -> tuple:
        """Parse version string into tuple for sorting."""
        try:
            parts = version.split(".")
            return tuple(int(part) for part in parts)
        except (ValueError, AttributeError):
            return (0, 0, 0)

    def _increment_version(self, current_version: str, change_type: ChangeType) -> str:
        """Increment version based on change type."""
        try:
            major, minor, patch = map(int, current_version.split("."))
        except (ValueError, AttributeError):
            major, minor, patch = 1, 0, 0

        if change_type == ChangeType.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif change_type == ChangeType.MINOR:
            minor += 1
            patch = 0
        else:  # PATCH
            patch += 1

        return f"{major}.{minor}.{patch}"

    def calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data."""
        import hashlib

        if isinstance(data, dict):
            json_str = json.dumps(data, sort_keys=True)
        else:
            json_str = str(data)

        return hashlib.sha256(json_str.encode()).hexdigest()


class TaxonomyVersionManager(VersionManager[Dict[str, Any]]):
    """Version manager specifically for taxonomy resources."""

    def __init__(self):
        super().__init__("taxonomy", Path("config/taxonomy"))
