"""
Integration tests for centralized taxonomy and schema management.

Tests the complete taxonomy system across all microservices.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from shared.taxonomy import (
    canonical_taxonomy,
    schema_evolution_manager,
    framework_mapping_registry,
    ChangeType,
)


class TestTaxonomyIntegration:
    """Integration tests for taxonomy system."""

    def setup_method(self):
        """Setup test environment."""
        # Use temporary directories for testing
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_canonical_taxonomy_basic_operations(self):
        """Test basic canonical taxonomy operations."""
        # Test label validation
        assert canonical_taxonomy.is_valid_label("PII.Contact.Email")
        assert canonical_taxonomy.is_valid_label("SECURITY.Credentials.Password")
        assert not canonical_taxonomy.is_valid_label("INVALID.Label")

        # Test category extraction
        assert canonical_taxonomy.get_category("PII.Contact.Email") == "PII"
        assert canonical_taxonomy.get_subcategory("PII.Contact.Email") == "Contact"
        assert canonical_taxonomy.get_type("PII.Contact.Email") == "Email"

        # Test label listing
        pii_labels = canonical_taxonomy.get_labels_by_category("PII")
        assert len(pii_labels) > 0
        assert any("Email" in label for label in pii_labels)

    def test_taxonomy_versioning(self):
        """Test taxonomy versioning system."""
        # Get initial version
        initial_version = canonical_taxonomy.version_manager.current_version
        assert initial_version is not None

        # Add a new category
        canonical_taxonomy.add_category("TEST", "Test category")

        # Create new version
        new_version = canonical_taxonomy.create_new_version(
            ChangeType.MINOR, ["Added TEST category"], "test-user"
        )

        assert new_version != initial_version
        assert canonical_taxonomy.is_valid_label("TEST")

    def test_schema_evolution_basic_operations(self):
        """Test basic schema evolution operations."""
        # List available schemas
        schemas = schema_evolution_manager.list_schemas()
        assert len(schemas) > 0
        assert "orchestration_request" in schemas

        # Get schema definition
        schema_def = schema_evolution_manager.get_schema_definition(
            "orchestration_request"
        )
        assert schema_def is not None
        assert "properties" in schema_def
        assert "tenant_id" in schema_def["properties"]

        # Validate data
        valid_data = {"tenant_id": "test-tenant", "content": "test content"}
        is_valid, errors = schema_evolution_manager.validate_data(
            valid_data, "orchestration_request"
        )
        assert is_valid
        assert len(errors) == 0

        # Test invalid data
        invalid_data = {"content": "test content"}  # Missing required tenant_id
        is_valid, errors = schema_evolution_manager.validate_data(
            invalid_data, "orchestration_request"
        )
        assert not is_valid
        assert len(errors) > 0

    def test_framework_mappings_basic_operations(self):
        """Test basic framework mapping operations."""
        # List supported frameworks
        frameworks = framework_mapping_registry.get_supported_frameworks()
        assert len(frameworks) > 0
        assert "GDPR" in frameworks
        assert "HIPAA" in frameworks

        # Get framework mappings
        gdpr_mappings = framework_mapping_registry.get_framework_mappings("GDPR")
        assert gdpr_mappings is not None
        assert len(gdpr_mappings) > 0

        # Test mapping functionality
        canonical_labels = ["PII.Contact.Email", "PII.Identity.Name"]
        mapped_labels = framework_mapping_registry.map_to_framework(
            canonical_labels, "GDPR"
        )

        assert len(mapped_labels) == len(canonical_labels)
        for canonical_label in canonical_labels:
            assert canonical_label in mapped_labels

    def test_framework_coverage_stats(self):
        """Test framework coverage statistics."""
        # Get all canonical labels
        all_labels = canonical_taxonomy.valid_labels

        # Test coverage for GDPR
        coverage_stats = framework_mapping_registry.get_coverage_stats(
            "GDPR", all_labels
        )

        assert "coverage" in coverage_stats
        assert "mapped" in coverage_stats
        assert "total" in coverage_stats
        assert coverage_stats["coverage"] >= 0.0
        assert coverage_stats["coverage"] <= 100.0
        assert coverage_stats["mapped"] <= coverage_stats["total"]

    def test_cross_service_integration(self):
        """Test integration across different service components."""
        # Test that mapper service can use canonical taxonomy
        from mapper_service.src.mapper.taxonomy import (
            canonical_taxonomy as mapper_taxonomy,
        )

        # Should be the same instance
        assert mapper_taxonomy is canonical_taxonomy

        # Test that analysis service can use canonical taxonomy
        from analysis_service.src.analysis.taxonomy import (
            canonical_taxonomy as analysis_taxonomy,
        )

        # Should be the same instance
        assert analysis_taxonomy is canonical_taxonomy

    def test_taxonomy_stats_consistency(self):
        """Test that taxonomy statistics are consistent."""
        stats = canonical_taxonomy.get_taxonomy_stats()

        assert "version" in stats
        assert "total_labels" in stats
        assert "active_categories" in stats
        assert "active_subcategories" in stats

        # Verify counts make sense
        assert stats["total_labels"] > 0
        assert stats["active_categories"] > 0
        assert stats["active_subcategories"] > 0
        assert stats["total_labels"] >= stats["active_subcategories"]

    def test_schema_compatibility_checking(self):
        """Test schema compatibility checking."""
        # This test assumes we have multiple versions of a schema
        # For now, we'll test the basic functionality

        schema_name = "orchestration_request"
        versions = schema_evolution_manager.get_schema_versions(schema_name)

        if len(versions) >= 2:
            # Test compatibility between versions
            compatibility = schema_evolution_manager.check_compatibility(
                schema_name, versions[0], versions[-1]
            )

            assert compatibility.from_version == versions[0]
            assert compatibility.to_version == versions[-1]
            assert compatibility.compatibility_level is not None
            assert isinstance(compatibility.compatible, bool)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Validate input using canonical taxonomy
        test_labels = ["PII.Contact.Email", "SECURITY.Credentials.Password"]

        for label in test_labels:
            assert canonical_taxonomy.is_valid_label(label)

        # 2. Map to compliance frameworks
        gdpr_mappings = framework_mapping_registry.map_to_framework(test_labels, "GDPR")
        hipaa_mappings = framework_mapping_registry.map_to_framework(
            test_labels, "HIPAA"
        )

        assert len(gdpr_mappings) == len(test_labels)
        assert len(hipaa_mappings) == len(test_labels)

        # 3. Validate request schema
        request_data = {
            "tenant_id": "test-tenant",
            "content": "test content with email@example.com",
            "detector_types": ["presidio", "deberta"],
            "processing_mode": "standard",
        }

        is_valid, errors = schema_evolution_manager.validate_data(
            request_data, "orchestration_request"
        )
        assert is_valid, f"Validation errors: {errors}"

        # 4. Get taxonomy statistics
        stats = canonical_taxonomy.get_taxonomy_stats()
        assert stats["total_labels"] > len(test_labels)


if __name__ == "__main__":
    pytest.main([__file__])
