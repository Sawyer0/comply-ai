"""
Tests for quality gates functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from src.llama_mapper.api.models import MappingResponse, Provenance
from src.llama_mapper.monitoring.quality_gates import (
    GoldenTestCase,
    QualityGateResult,
    QualityGateValidator,
    create_sample_golden_test_cases,
)


class TestQualityGateValidator:
    """Test quality gate validator functionality."""

    def test_load_golden_test_cases_empty_file(self):
        """Test loading golden test cases from non-existent file."""
        validator = QualityGateValidator(
            golden_test_cases_path=Path("nonexistent.json")
        )
        test_cases = validator.load_golden_test_cases()
        assert test_cases == []

    def test_load_golden_test_cases_valid_file(self):
        """Test loading golden test cases from valid file."""
        test_data = {
            "test_cases": [
                {
                    "detector": "test-detector",
                    "input_output": "test-input",
                    "expected_taxonomy": ["TEST.Category"],
                    "expected_confidence_min": 0.8,
                    "description": "Test case",
                    "category": "TEST",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            validator = QualityGateValidator(golden_test_cases_path=temp_path)
            test_cases = validator.load_golden_test_cases()

            assert len(test_cases) == 1
            assert test_cases[0].detector == "test-detector"
            assert test_cases[0].input_output == "test-input"
            assert test_cases[0].expected_taxonomy == ["TEST.Category"]
            assert test_cases[0].expected_confidence_min == 0.8
            assert test_cases[0].description == "Test case"
            assert test_cases[0].category == "TEST"
        finally:
            temp_path.unlink()

    def test_validate_golden_test_coverage_insufficient(self):
        """Test validation when there are insufficient golden test cases."""
        test_cases = [
            GoldenTestCase(
                detector="test-detector",
                input_output="test",
                expected_taxonomy=["TEST.Category"],
                expected_confidence_min=0.8,
                description="Test",
                category="TEST",
            )
        ]

        validator = QualityGateValidator()
        # Mock detector loader to return our test detector
        validator.thresholds["min_golden_cases_per_detector"] = 100

        results = validator.validate_golden_test_coverage(test_cases)

        # Should have at least one result for our detector
        detector_results = [r for r in results if "test-detector" in r.metric_name]
        assert len(detector_results) >= 1

        # Should fail because we only have 1 case but need 100
        failed_results = [r for r in detector_results if not r.passed]
        assert len(failed_results) >= 1

    @pytest.mark.asyncio
    async def test_run_golden_test_cases_success(self):
        """Test running golden test cases with successful results."""
        # Mock dependencies
        mock_model_server = Mock()
        mock_model_server.generate_mapping = AsyncMock(
            return_value='{"taxonomy": ["TEST.Category"], "scores": {"TEST.Category": 0.95}, "confidence": 0.95}'
        )

        mock_json_validator = Mock()
        mock_json_validator.validate.return_value = (True, [])
        mock_json_validator.parse_output.return_value = MappingResponse(
            taxonomy=["TEST.Category"],
            scores={"TEST.Category": 0.95},
            confidence=0.95,
            provenance=Provenance(),
        )

        test_cases = [
            GoldenTestCase(
                detector="test-detector",
                input_output="test-input",
                expected_taxonomy=["TEST.Category"],
                expected_confidence_min=0.8,
                description="Test case",
                category="TEST",
            )
        ]

        validator = QualityGateValidator(
            model_server=mock_model_server, json_validator=mock_json_validator
        )

        results, metrics = await validator.run_golden_test_cases(test_cases)

        # Check metrics
        assert metrics["total_cases"] == 1
        assert metrics["passed_cases"] == 1
        assert metrics["failed_cases"] == 0
        assert metrics["schema_valid_cases"] == 1
        assert len(metrics["latency_measurements"]) == 1
        assert len(metrics["confidence_scores"]) == 1

        # Check results
        schema_results = [
            r for r in results if r.metric_name == "schema_valid_percentage"
        ]
        assert len(schema_results) == 1
        assert schema_results[0].passed
        assert schema_results[0].actual_value == 100.0

    @pytest.mark.asyncio
    async def test_run_golden_test_cases_schema_failure(self):
        """Test running golden test cases with schema validation failure."""
        # Mock dependencies
        mock_model_server = Mock()
        mock_model_server.generate_mapping = AsyncMock(return_value="invalid json")

        mock_json_validator = Mock()
        mock_json_validator.validate.return_value = (False, ["Invalid JSON"])

        test_cases = [
            GoldenTestCase(
                detector="test-detector",
                input_output="test-input",
                expected_taxonomy=["TEST.Category"],
                expected_confidence_min=0.8,
                description="Test case",
                category="TEST",
            )
        ]

        validator = QualityGateValidator(
            model_server=mock_model_server, json_validator=mock_json_validator
        )

        results, metrics = await validator.run_golden_test_cases(test_cases)

        # Check metrics
        assert metrics["total_cases"] == 1
        assert metrics["passed_cases"] == 0
        assert metrics["failed_cases"] == 1
        assert metrics["schema_valid_cases"] == 0

        # Check results
        schema_results = [
            r for r in results if r.metric_name == "schema_valid_percentage"
        ]
        assert len(schema_results) == 1
        assert not schema_results[0].passed
        assert schema_results[0].actual_value == 0.0

    @pytest.mark.asyncio
    async def test_validate_all_quality_gates(self):
        """Test complete quality gate validation."""
        # Create temporary golden test cases file
        test_data = {
            "test_cases": [
                {
                    "detector": "test-detector",
                    "input_output": "test-input",
                    "expected_taxonomy": ["TEST.Category"],
                    "expected_confidence_min": 0.8,
                    "description": "Test case",
                    "category": "TEST",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            # Mock dependencies
            mock_model_server = Mock()
            mock_model_server.generate_mapping = AsyncMock(
                return_value='{"taxonomy": ["TEST.Category"], "scores": {"TEST.Category": 0.95}, "confidence": 0.95}'
            )

            mock_json_validator = Mock()
            mock_json_validator.validate.return_value = (True, [])
            mock_json_validator.parse_output.return_value = MappingResponse(
                taxonomy=["TEST.Category"],
                scores={"TEST.Category": 0.95},
                confidence=0.95,
                provenance=Provenance(),
            )

            mock_metrics_collector = Mock()
            mock_metrics_collector.check_quality_thresholds.return_value = []

            validator = QualityGateValidator(
                model_server=mock_model_server,
                json_validator=mock_json_validator,
                metrics_collector=mock_metrics_collector,
                golden_test_cases_path=temp_path,
            )

            # Lower thresholds for testing
            validator.thresholds["min_golden_cases_per_detector"] = 1

            passed, results, metrics = await validator.validate_all_quality_gates()

            # Should have some results
            assert len(results) > 0

            # Check that we have coverage and test results
            coverage_results = [r for r in results if "golden_cases" in r.metric_name]
            test_results = [
                r
                for r in results
                if r.metric_name in ["schema_valid_percentage", "golden_test_pass_rate"]
            ]

            assert len(coverage_results) > 0
            assert len(test_results) > 0

        finally:
            temp_path.unlink()

    def test_generate_quality_report(self):
        """Test quality report generation."""
        results = [
            QualityGateResult(
                passed=True,
                metric_name="test_metric_1",
                actual_value=95.0,
                threshold=90.0,
                message="Test passed",
            ),
            QualityGateResult(
                passed=False,
                metric_name="test_metric_2",
                actual_value=85.0,
                threshold=90.0,
                message="Test failed",
                severity="error",
            ),
        ]

        metrics = {"test_data": "value"}

        validator = QualityGateValidator()
        report = validator.generate_quality_report(results, metrics)

        assert report["overall_status"] == "FAILED"  # Because one test failed
        assert report["summary"]["total_checks"] == 2
        assert report["summary"]["passed_checks"] == 1
        assert report["summary"]["failed_checks"] == 1
        assert report["summary"]["critical_failures"] == 1
        assert len(report["results"]) == 2
        assert report["detailed_metrics"] == metrics
        assert isinstance(report["recommendations"], list)

    def test_create_sample_golden_test_cases(self):
        """Test creation of sample golden test cases file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            create_sample_golden_test_cases(temp_path)

            assert temp_path.exists()

            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "version" in data
            assert "test_cases" in data
            assert len(data["test_cases"]) > 0

            # Check first test case structure
            first_case = data["test_cases"][0]
            required_fields = [
                "detector",
                "input_output",
                "expected_taxonomy",
                "expected_confidence_min",
                "description",
                "category",
            ]
            for field in required_fields:
                assert field in first_case

        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestQualityGateResult:
    """Test QualityGateResult dataclass."""

    def test_quality_gate_result_creation(self):
        """Test creating a QualityGateResult."""
        result = QualityGateResult(
            passed=True,
            metric_name="test_metric",
            actual_value=95.0,
            threshold=90.0,
            message="Test passed",
        )

        assert result.passed is True
        assert result.metric_name == "test_metric"
        assert result.actual_value == 95.0
        assert result.threshold == 90.0
        assert result.message == "Test passed"
        assert result.severity == "error"  # Default value


class TestGoldenTestCase:
    """Test GoldenTestCase dataclass."""

    def test_golden_test_case_creation(self):
        """Test creating a GoldenTestCase."""
        case = GoldenTestCase(
            detector="test-detector",
            input_output="test-input",
            expected_taxonomy=["TEST.Category"],
            expected_confidence_min=0.8,
            description="Test case",
            category="TEST",
        )

        assert case.detector == "test-detector"
        assert case.input_output == "test-input"
        assert case.expected_taxonomy == ["TEST.Category"]
        assert case.expected_confidence_min == 0.8
        assert case.description == "Test case"
        assert case.category == "TEST"
