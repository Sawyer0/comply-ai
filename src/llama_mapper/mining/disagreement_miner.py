"""
Disagreement Mining System for Compliance AI Models

Log cases where Orchestrator detectors disagree or Mapper confidence < threshold.
Feed only these back into the next FT round (big quality bang for small data).
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DetectorOutput:
    """Individual detector output."""

    detector_id: str
    detector_type: str
    confidence: float
    taxonomy: List[str]
    raw_output: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OrchestratorResult:
    """Result from detector orchestration."""

    request_id: str
    detector_outputs: List[DetectorOutput]
    mapper_output: Optional[Dict[str, Any]] = None
    mapper_confidence: Optional[float] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DisagreementCase:
    """A case where detectors disagree or confidence is low."""

    case_id: str
    disagreement_type: (
        str  # "detector_disagreement", "low_confidence", "mapper_failure"
    )
    orchestrator_result: OrchestratorResult
    disagreement_details: Dict[str, Any]
    severity: str  # "low", "medium", "high", "critical"
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class DisagreementDetector:
    """Detects disagreements between detectors and low-confidence cases."""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.disagreement_cases = []
        self.logger = logging.getLogger(__name__)

    def analyze_orchestrator_result(
        self, result: OrchestratorResult
    ) -> List[DisagreementCase]:
        """Analyze orchestrator result for disagreements and low confidence."""
        disagreement_cases = []

        # Check for detector disagreements
        detector_disagreements = self._detect_detector_disagreements(result)
        disagreement_cases.extend(detector_disagreements)

        # Check for low mapper confidence
        low_confidence_cases = self._detect_low_confidence(result)
        disagreement_cases.extend(low_confidence_cases)

        # Check for mapper failures
        mapper_failures = self._detect_mapper_failures(result)
        disagreement_cases.extend(mapper_failures)

        # Store cases
        self.disagreement_cases.extend(disagreement_cases)

        return disagreement_cases

    def _detect_detector_disagreements(
        self, result: OrchestratorResult
    ) -> List[DisagreementCase]:
        """Detect disagreements between detectors."""
        disagreement_cases = []

        if len(result.detector_outputs) < 2:
            return disagreement_cases  # Need at least 2 detectors for disagreement

        # Group detectors by type
        detector_groups = defaultdict(list)
        for output in result.detector_outputs:
            detector_groups[output.detector_type].append(output)

        # Check for disagreements within each detector type
        for detector_type, outputs in detector_groups.items():
            if len(outputs) >= 2:
                disagreements = self._find_taxonomy_disagreements(outputs)
                if disagreements:
                    case = DisagreementCase(
                        case_id=f"{result.request_id}_detector_disagreement_{detector_type}",
                        disagreement_type="detector_disagreement",
                        orchestrator_result=result,
                        disagreement_details={
                            "detector_type": detector_type,
                            "disagreements": disagreements,
                            "outputs": [asdict(output) for output in outputs],
                        },
                        severity=self._calculate_disagreement_severity(disagreements),
                    )
                    disagreement_cases.append(case)

        # Check for cross-detector disagreements
        cross_disagreements = self._find_cross_detector_disagreements(
            result.detector_outputs
        )
        if cross_disagreements:
            case = DisagreementCase(
                case_id=f"{result.request_id}_cross_detector_disagreement",
                disagreement_type="cross_detector_disagreement",
                orchestrator_result=result,
                disagreement_details={
                    "cross_disagreements": cross_disagreements,
                    "outputs": [asdict(output) for output in result.detector_outputs],
                },
                severity=self._calculate_cross_disagreement_severity(
                    cross_disagreements
                ),
            )
            disagreement_cases.append(case)

        return disagreement_cases

    def _find_taxonomy_disagreements(
        self, outputs: List[DetectorOutput]
    ) -> List[Dict[str, Any]]:
        """Find taxonomy disagreements within a detector type."""
        disagreements = []

        # Compare all pairs of outputs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                output1, output2 = outputs[i], outputs[j]

                # Check taxonomy disagreement
                taxonomy1 = set(output1.taxonomy)
                taxonomy2 = set(output2.taxonomy)

                if taxonomy1 != taxonomy2:
                    disagreement = {
                        "detector1": output1.detector_id,
                        "detector2": output2.detector_id,
                        "taxonomy1": list(taxonomy1),
                        "taxonomy2": list(taxonomy2),
                        "common": list(taxonomy1 & taxonomy2),
                        "only_in_1": list(taxonomy1 - taxonomy2),
                        "only_in_2": list(taxonomy2 - taxonomy1),
                        "confidence1": output1.confidence,
                        "confidence2": output2.confidence,
                    }
                    disagreements.append(disagreement)

        return disagreements

    def _find_cross_detector_disagreements(
        self, outputs: List[DetectorOutput]
    ) -> List[Dict[str, Any]]:
        """Find disagreements between different detector types."""
        disagreements = []

        # Group by detector type
        detector_groups = defaultdict(list)
        for output in outputs:
            detector_groups[output.detector_type].append(output)

        # Compare outputs from different detector types
        detector_types = list(detector_groups.keys())
        for i in range(len(detector_types)):
            for j in range(i + 1, len(detector_types)):
                type1, type2 = detector_types[i], detector_types[j]
                outputs1 = detector_groups[type1]
                outputs2 = detector_groups[type2]

                # Find best outputs from each type (highest confidence)
                best_output1 = max(outputs1, key=lambda x: x.confidence)
                best_output2 = max(outputs2, key=lambda x: x.confidence)

                # Check for disagreement
                taxonomy1 = set(best_output1.taxonomy)
                taxonomy2 = set(best_output2.taxonomy)

                if taxonomy1 != taxonomy2:
                    disagreement = {
                        "detector_type1": type1,
                        "detector_type2": type2,
                        "detector1": best_output1.detector_id,
                        "detector2": best_output2.detector_id,
                        "taxonomy1": list(taxonomy1),
                        "taxonomy2": list(taxonomy2),
                        "common": list(taxonomy1 & taxonomy2),
                        "only_in_1": list(taxonomy1 - taxonomy2),
                        "only_in_2": list(taxonomy2 - taxonomy1),
                        "confidence1": best_output1.confidence,
                        "confidence2": best_output2.confidence,
                    }
                    disagreements.append(disagreement)

        return disagreements

    def _detect_low_confidence(
        self, result: OrchestratorResult
    ) -> List[DisagreementCase]:
        """Detect cases with low mapper confidence."""
        disagreement_cases = []

        if (
            result.mapper_confidence is not None
            and result.mapper_confidence < self.confidence_threshold
        ):
            case = DisagreementCase(
                case_id=f"{result.request_id}_low_confidence",
                disagreement_type="low_confidence",
                orchestrator_result=result,
                disagreement_details={
                    "mapper_confidence": result.mapper_confidence,
                    "threshold": self.confidence_threshold,
                    "confidence_gap": self.confidence_threshold
                    - result.mapper_confidence,
                },
                severity=self._calculate_confidence_severity(result.mapper_confidence),
            )
            disagreement_cases.append(case)

        return disagreement_cases

    def _detect_mapper_failures(
        self, result: OrchestratorResult
    ) -> List[DisagreementCase]:
        """Detect mapper failures (no output, invalid output, etc.)."""
        disagreement_cases = []

        if result.mapper_output is None:
            case = DisagreementCase(
                case_id=f"{result.request_id}_mapper_failure",
                disagreement_type="mapper_failure",
                orchestrator_result=result,
                disagreement_details={
                    "failure_type": "no_output",
                    "detector_outputs": [
                        asdict(output) for output in result.detector_outputs
                    ],
                },
                severity="high",
            )
            disagreement_cases.append(case)
        elif not self._is_valid_mapper_output(result.mapper_output):
            case = DisagreementCase(
                case_id=f"{result.request_id}_mapper_invalid_output",
                disagreement_type="mapper_failure",
                orchestrator_result=result,
                disagreement_details={
                    "failure_type": "invalid_output",
                    "mapper_output": result.mapper_output,
                    "detector_outputs": [
                        asdict(output) for output in result.detector_outputs
                    ],
                },
                severity="medium",
            )
            disagreement_cases.append(case)

        return disagreement_cases

    def _is_valid_mapper_output(self, mapper_output: Dict[str, Any]) -> bool:
        """Check if mapper output is valid."""
        required_fields = ["taxonomy", "scores", "confidence"]
        return all(field in mapper_output for field in required_fields)

    def _calculate_disagreement_severity(
        self, disagreements: List[Dict[str, Any]]
    ) -> str:
        """Calculate severity of detector disagreements."""
        if not disagreements:
            return "low"

        # Count total disagreements
        total_disagreements = len(disagreements)

        # Check for high-confidence disagreements
        high_confidence_disagreements = 0
        for disagreement in disagreements:
            if disagreement["confidence1"] > 0.8 and disagreement["confidence2"] > 0.8:
                high_confidence_disagreements += 1

        if high_confidence_disagreements > 0:
            return "critical"
        elif total_disagreements > 2:
            return "high"
        elif total_disagreements > 1:
            return "medium"
        else:
            return "low"

    def _calculate_cross_disagreement_severity(
        self, disagreements: List[Dict[str, Any]]
    ) -> str:
        """Calculate severity of cross-detector disagreements."""
        if not disagreements:
            return "low"

        # Cross-detector disagreements are generally more serious
        return "high" if len(disagreements) > 0 else "low"

    def _calculate_confidence_severity(self, confidence: float) -> str:
        """Calculate severity based on confidence level."""
        if confidence < 0.3:
            return "critical"
        elif confidence < 0.5:
            return "high"
        elif confidence < 0.7:
            return "medium"
        else:
            return "low"


class DisagreementMiner:
    """Mines disagreement cases for targeted retraining."""

    def __init__(self, output_dir: str = "disagreement_mining"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.detector = DisagreementDetector()
        self.mining_stats = {
            "total_cases_analyzed": 0,
            "disagreement_cases_found": 0,
            "low_confidence_cases": 0,
            "mapper_failures": 0,
            "severity_distribution": Counter(),
        }

    def mine_disagreements(
        self, orchestrator_results: List[OrchestratorResult]
    ) -> List[DisagreementCase]:
        """Mine disagreement cases from orchestrator results."""
        print(
            f"🔥 Mining disagreements from {len(orchestrator_results)} orchestrator results..."
        )

        all_disagreement_cases = []

        for result in orchestrator_results:
            self.mining_stats["total_cases_analyzed"] += 1

            disagreement_cases = self.detector.analyze_orchestrator_result(result)
            all_disagreement_cases.extend(disagreement_cases)

            # Update stats
            for case in disagreement_cases:
                self.mining_stats["disagreement_cases_found"] += 1
                self.mining_stats["severity_distribution"][case.severity] += 1

                if case.disagreement_type == "low_confidence":
                    self.mining_stats["low_confidence_cases"] += 1
                elif case.disagreement_type == "mapper_failure":
                    self.mining_stats["mapper_failures"] += 1

        # Save disagreement cases
        self._save_disagreement_cases(all_disagreement_cases)

        # Generate mining report
        self._generate_mining_report(all_disagreement_cases)

        print(f"✅ Found {len(all_disagreement_cases)} disagreement cases")
        print(f"  - Low confidence: {self.mining_stats['low_confidence_cases']}")
        print(f"  - Mapper failures: {self.mining_stats['mapper_failures']}")
        print(
            f"  - Detector disagreements: {self.mining_stats['disagreement_cases_found'] - self.mining_stats['low_confidence_cases'] - self.mining_stats['mapper_failures']}"
        )

        return all_disagreement_cases

    def _save_disagreement_cases(self, disagreement_cases: List[DisagreementCase]):
        """Save disagreement cases to files."""
        # Save individual cases
        for case in disagreement_cases:
            case_file = self.output_dir / f"disagreement_case_{case.case_id}.json"
            with open(case_file, "w") as f:
                json.dump(asdict(case), f, indent=2)

        # Save combined file
        combined_file = self.output_dir / "all_disagreement_cases.json"
        with open(combined_file, "w") as f:
            json.dump([asdict(case) for case in disagreement_cases], f, indent=2)

        # Save by severity
        for severity in ["low", "medium", "high", "critical"]:
            severity_cases = [
                case for case in disagreement_cases if case.severity == severity
            ]
            if severity_cases:
                severity_file = self.output_dir / f"disagreement_cases_{severity}.json"
                with open(severity_file, "w") as f:
                    json.dump([asdict(case) for case in severity_cases], f, indent=2)

    def _generate_mining_report(self, disagreement_cases: List[DisagreementCase]):
        """Generate mining report with statistics."""
        report = {
            "mining_timestamp": datetime.now().isoformat(),
            "mining_stats": self.mining_stats,
            "disagreement_analysis": self._analyze_disagreements(disagreement_cases),
            "recommendations": self._generate_recommendations(disagreement_cases),
        }

        report_file = self.output_dir / "mining_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

    def _analyze_disagreements(
        self, disagreement_cases: List[DisagreementCase]
    ) -> Dict[str, Any]:
        """Analyze disagreement patterns."""
        analysis = {
            "total_cases": len(disagreement_cases),
            "by_type": Counter(case.disagreement_type for case in disagreement_cases),
            "by_severity": Counter(case.severity for case in disagreement_cases),
            "common_taxonomy_conflicts": Counter(),
            "detector_type_conflicts": Counter(),
        }

        # Analyze taxonomy conflicts
        for case in disagreement_cases:
            if case.disagreement_type in [
                "detector_disagreement",
                "cross_detector_disagreement",
            ]:
                details = case.disagreement_details
                if "disagreements" in details:
                    for disagreement in details["disagreements"]:
                        for tax in disagreement.get("only_in_1", []):
                            analysis["common_taxonomy_conflicts"][tax] += 1
                        for tax in disagreement.get("only_in_2", []):
                            analysis["common_taxonomy_conflicts"][tax] += 1

                if "detector_type" in details:
                    analysis["detector_type_conflicts"][details["detector_type"]] += 1

        return analysis

    def _generate_recommendations(
        self, disagreement_cases: List[DisagreementCase]
    ) -> List[str]:
        """Generate recommendations based on disagreement analysis."""
        recommendations = []

        # Analyze patterns
        low_confidence_cases = [
            case
            for case in disagreement_cases
            if case.disagreement_type == "low_confidence"
        ]
        mapper_failures = [
            case
            for case in disagreement_cases
            if case.disagreement_type == "mapper_failure"
        ]
        detector_disagreements = [
            case
            for case in disagreement_cases
            if case.disagreement_type == "detector_disagreement"
        ]

        if len(low_confidence_cases) > len(disagreement_cases) * 0.3:
            recommendations.append(
                "High rate of low-confidence cases - consider retraining with more diverse examples"
            )

        if len(mapper_failures) > len(disagreement_cases) * 0.1:
            recommendations.append(
                "Mapper failures detected - investigate model stability and input validation"
            )

        if len(detector_disagreements) > len(disagreement_cases) * 0.4:
            recommendations.append(
                "High detector disagreement rate - consider ensemble methods or confidence weighting"
            )

        # Check for specific taxonomy conflicts
        analysis = self._analyze_disagreements(disagreement_cases)
        common_conflicts = analysis["common_taxonomy_conflicts"]
        if common_conflicts:
            top_conflict = common_conflicts.most_common(1)[0]
            recommendations.append(
                f"Taxonomy conflict in {top_conflict[0]} - consider targeted training for this category"
            )

        return recommendations

    def create_training_dataset(
        self, disagreement_cases: List[DisagreementCase], max_examples: int = 1000
    ) -> List[Dict[str, Any]]:
        """Create training dataset from disagreement cases."""
        print(
            f"🔥 Creating training dataset from {len(disagreement_cases)} disagreement cases..."
        )

        training_examples = []

        # Prioritize by severity
        severity_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        sorted_cases = sorted(
            disagreement_cases,
            key=lambda x: severity_priority.get(x.severity, 0),
            reverse=True,
        )

        for case in sorted_cases[:max_examples]:
            # Create training example from disagreement case
            training_example = self._create_training_example_from_case(case)
            if training_example:
                training_examples.append(training_example)

        # Save training dataset
        dataset_file = self.output_dir / "disagreement_training_dataset.jsonl"
        with open(dataset_file, "w") as f:
            for example in training_examples:
                json.dump(example, f)
                f.write("\n")

        print(f"✅ Created training dataset with {len(training_examples)} examples")
        return training_examples

    def _create_training_example_from_case(
        self, case: DisagreementCase
    ) -> Optional[Dict[str, Any]]:
        """Create training example from disagreement case."""
        try:
            # Extract input from orchestrator result
            detector_outputs = case.orchestrator_result.detector_outputs
            if not detector_outputs:
                return None

            # Combine detector outputs into input
            input_text = self._combine_detector_outputs(detector_outputs)

            # Create expected output (this would be determined by human review or consensus)
            expected_output = self._create_expected_output(case)

            if expected_output:
                return {
                    "instruction": f"Map these detector outputs to canonical taxonomy: {input_text}",
                    "response": json.dumps(expected_output),
                    "metadata": {
                        "case_id": case.case_id,
                        "disagreement_type": case.disagreement_type,
                        "severity": case.severity,
                        "source": "disagreement_mining",
                    },
                }

        except Exception as e:
            print(f"⚠️ Error creating training example from case {case.case_id}: {e}")
            return None

        return None

    def _combine_detector_outputs(self, detector_outputs: List[DetectorOutput]) -> str:
        """Combine detector outputs into input text."""
        combined = []
        for output in detector_outputs:
            combined.append(f"{output.detector_type}: {output.raw_output}")
        return " | ".join(combined)

    def _create_expected_output(
        self, case: DisagreementCase
    ) -> Optional[Dict[str, Any]]:
        """Create expected output for training (would be determined by human review)."""
        # This is a simplified version - in practice, this would involve human review
        # or consensus mechanisms to determine the "correct" output

        if case.disagreement_type == "low_confidence":
            # For low confidence cases, we might want to retrain with higher confidence examples
            return None  # Skip for now

        elif case.disagreement_type == "detector_disagreement":
            # For detector disagreements, we might use consensus or human review
            details = case.disagreement_details
            if "disagreements" in details:
                # Use the most common taxonomy from disagreements
                all_taxonomies = []
                for disagreement in details["disagreements"]:
                    all_taxonomies.extend(disagreement.get("taxonomy1", []))
                    all_taxonomies.extend(disagreement.get("taxonomy2", []))

                if all_taxonomies:
                    # Find most common taxonomy
                    taxonomy_counts = Counter(all_taxonomies)
                    most_common = taxonomy_counts.most_common(1)[0][0]

                    return {
                        "taxonomy": [most_common],
                        "scores": {most_common: 0.8},  # Moderate confidence
                        "confidence": 0.8,
                    }

        return None


# Example usage and testing
if __name__ == "__main__":
    # Create sample orchestrator results
    sample_results = [
        OrchestratorResult(
            request_id="req_001",
            detector_outputs=[
                DetectorOutput(
                    detector_id="pii_detector_1",
                    detector_type="pii",
                    confidence=0.95,
                    taxonomy=["PII.Contact.Email"],
                    raw_output="email address detected: john@company.com",
                    timestamp="2024-01-15T10:30:00Z",
                ),
                DetectorOutput(
                    detector_id="pii_detector_2",
                    detector_type="pii",
                    confidence=0.90,
                    taxonomy=["PII.Contact.Email", "PII.Identifier.SSN"],
                    raw_output="email and SSN detected",
                    timestamp="2024-01-15T10:30:00Z",
                ),
            ],
            mapper_output={
                "taxonomy": ["PII.Contact.Email"],
                "scores": {"PII.Contact.Email": 0.85},
                "confidence": 0.85,
            },
            mapper_confidence=0.85,
        )
    ]

    # Mine disagreements
    miner = DisagreementMiner()
    disagreement_cases = miner.mine_disagreements(sample_results)

    print(f"🎉 Disagreement mining completed!")
    print(f"  Found {len(disagreement_cases)} disagreement cases")
    print(f"  Mining stats: {miner.mining_stats}")
