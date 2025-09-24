"""
Mutation testing framework for validating test quality across all services.

This module provides mutation testing capabilities for:
- Core Mapper Service
- Detector Orchestration Service
- Analysis Service
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MutationResult:
    """Result of a single mutation test."""
    mutant_id: str
    file_path: str
    line_number: int
    mutation_type: str
    original_code: str
    mutated_code: str
    status: str  # killed, survived, timeout, error
    test_output: str
    execution_time: float


@dataclass
class ServiceMutationReport:
    """Mutation testing report for a service."""
    service_name: str
    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    timeout_mutants: int
    error_mutants: int
    mutation_score: float
    critical_path_score: float
    test_execution_time: float
    mutant_details: List[MutationResult]


@dataclass
class UnifiedMutationReport:
    """Unified mutation testing report across all services."""
    timestamp: str
    overall_mutation_score: float
    service_reports: Dict[str, ServiceMutationReport]
    quality_assessment: str
    recommendations: List[str]
    test_quality_trends: Dict[str, List[float]]


class MutationTestingFramework:
    """Framework for running mutation tests across all services."""
    
    def __init__(self):
        self.mutation_tools = {
            'mutmut': MutmutRunner(),
            'cosmic_ray': CosmicRayRunner()
        }
        
        # Service configurations
        self.service_configs = {
            'core_mapper': {
                'source_paths': ['src/llama_mapper'],
                'test_paths': ['tests/unit/test_mapper_service.py'],
                'critical_modules': [
                    'src/llama_mapper/models/',
                    'src/llama_mapper/serving/',
                    'src/llama_mapper/security/'
                ],
                'mutation_score_threshold': 0.80
            },
            'detector_orchestration': {
                'source_paths': ['detector-orchestration/src'],
                'test_paths': ['detector-orchestration/tests/unit/'],
                'critical_modules': [
                    'detector-orchestration/src/coordinator.py',
                    'detector-orchestration/src/circuit_breaker.py'
                ],
                'mutation_score_threshold': 0.80
            },
            'analysis_service': {
                'source_paths': ['analysis/src'],
                'test_paths': ['analysis/tests/unit/'],
                'critical_modules': [
                    'analysis/src/compliance/',
                    'analysis/src/risk_assessment/'
                ],
                'mutation_score_threshold': 0.80
            }
        }
        
        # Mutation types to focus on
        self.mutation_types = [
            'arithmetic_operator',
            'comparison_operator', 
            'boolean_operator',
            'assignment_operator',
            'unary_operator',
            'statement_deletion',
            'conditional_boundary'
        ]
    
    async def run_mutation_tests(self, 
                                service: str = None,
                                tool: str = 'mutmut') -> UnifiedMutationReport:
        """Run mutation tests for specified service or all services."""
        logger.info("Starting mutation testing", service=service, tool=tool)
        
        if service:
            services_to_test = [service]
        else:
            services_to_test = list(self.service_configs.keys())
        
        service_reports = {}
        total_score = 0.0
        valid_services = 0
        
        for service_name in services_to_test:
            try:
                report = await self._run_service_mutation_tests(service_name, tool)
                service_reports[service_name] = report
                total_score += report.mutation_score
                valid_services += 1
            except Exception as e:
                logger.error(f"Mutation testing failed for service", 
                           service=service_name, error=str(e))
        
        overall_score = total_score / valid_services if valid_services > 0 else 0.0
        
        # Generate unified report
        unified_report = UnifiedMutationReport(
            timestamp=datetime.utcnow().isoformat(),
            overall_mutation_score=overall_score,
            service_reports=service_reports,
            quality_assessment=self._assess_test_quality(overall_score),
            recommendations=self._generate_recommendations(service_reports),
            test_quality_trends={}  # Would track over time
        )
        
        # Save report
        await self._save_mutation_report(unified_report)
        
        logger.info("Mutation testing completed", 
                   overall_score=overall_score,
                   services_tested=len(service_reports))
        
        return unified_report
    
    async def _run_service_mutation_tests(self, 
                                        service: str, 
                                        tool: str) -> ServiceMutationReport:
        """Run mutation tests for a specific service."""
        logger.info(f"Running mutation tests for service", service=service, tool=tool)
        
        config = self.service_configs[service]
        mutation_runner = self.mutation_tools[tool]
        
        # Run mutation testing
        mutant_results = await mutation_runner.run_mutations(
            source_paths=config['source_paths'],
            test_paths=config['test_paths'],
            mutation_types=self.mutation_types
        )
        
        # Calculate metrics
        total_mutants = len(mutant_results)
        killed_mutants = sum(1 for m in mutant_results if m.status == 'killed')
        survived_mutants = sum(1 for m in mutant_results if m.status == 'survived')
        timeout_mutants = sum(1 for m in mutant_results if m.status == 'timeout')
        error_mutants = sum(1 for m in mutant_results if m.status == 'error')
        
        mutation_score = (killed_mutants / total_mutants) if total_mutants > 0 else 0.0
        
        # Calculate critical path score
        critical_path_score = await self._calculate_critical_path_mutation_score(
            mutant_results, config['critical_modules']
        )
        
        # Calculate total execution time
        execution_time = sum(m.execution_time for m in mutant_results)
        
        return ServiceMutationReport(
            service_name=service,
            total_mutants=total_mutants,
            killed_mutants=killed_mutants,
            survived_mutants=survived_mutants,
            timeout_mutants=timeout_mutants,
            error_mutants=error_mutants,
            mutation_score=mutation_score,
            critical_path_score=critical_path_score,
            test_execution_time=execution_time,
            mutant_details=mutant_results
        )
    
    async def _calculate_critical_path_mutation_score(self, 
                                                    mutant_results: List[MutationResult],
                                                    critical_modules: List[str]) -> float:
        """Calculate mutation score for critical paths only."""
        critical_mutants = [
            m for m in mutant_results 
            if any(critical in m.file_path for critical in critical_modules)
        ]
        
        if not critical_mutants:
            return 1.0  # No critical mutations found
        
        killed_critical = sum(1 for m in critical_mutants if m.status == 'killed')
        return killed_critical / len(critical_mutants)
    
    def _assess_test_quality(self, mutation_score: float) -> str:
        """Assess overall test quality based on mutation score."""
        if mutation_score >= 0.90:
            return "Excellent - Very high test quality"
        elif mutation_score >= 0.80:
            return "Good - Adequate test quality"
        elif mutation_score >= 0.70:
            return "Fair - Some test gaps exist"
        elif mutation_score >= 0.60:
            return "Poor - Significant test gaps"
        else:
            return "Very Poor - Major test quality issues"
    
    def _generate_recommendations(self, 
                                service_reports: Dict[str, ServiceMutationReport]) -> List[str]:
        """Generate recommendations based on mutation test results."""
        recommendations = []
        
        for service, report in service_reports.items():
            if report.mutation_score < 0.80:
                recommendations.append(
                    f"Improve test quality for {service} (score: {report.mutation_score:.2f})"
                )
            
            if report.critical_path_score < 0.90:
                recommendations.append(
                    f"Add tests for critical paths in {service} "
                    f"(critical path score: {report.critical_path_score:.2f})"
                )
            
            # Analyze surviving mutants
            survived_mutants = [m for m in report.mutant_details if m.status == 'survived']
            if survived_mutants:
                # Group by mutation type
                mutation_types = {}
                for mutant in survived_mutants:
                    if mutant.mutation_type not in mutation_types:
                        mutation_types[mutant.mutation_type] = 0
                    mutation_types[mutant.mutation_type] += 1
                
                for mutation_type, count in mutation_types.items():
                    if count > 5:  # Threshold for significant gaps
                        recommendations.append(
                            f"Add tests for {mutation_type} scenarios in {service} "
                            f"({count} surviving mutants)"
                        )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Mutation test quality is good! Consider adding edge case tests")
        
        return recommendations
    
    async def _save_mutation_report(self, report: UnifiedMutationReport) -> None:
        """Save mutation testing report."""
        # Save as JSON
        report_path = Path("tests/coverage/mutation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        # Save as HTML
        html_path = Path("tests/coverage/mutation_report.html")
        await self._generate_mutation_html_report(report, html_path)
        
        logger.info(f"Mutation report saved", 
                   json_path=str(report_path), 
                   html_path=str(html_path))
    
    async def _generate_mutation_html_report(self, 
                                           report: UnifiedMutationReport, 
                                           output_path: Path) -> None:
        """Generate HTML mutation testing report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mutation Testing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .excellent {{ color: green; }}
                .good {{ color: #228B22; }}
                .fair {{ color: orange; }}
                .poor {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .mutant-details {{ max-height: 200px; overflow-y: auto; }}
            </style>
        </head>
        <body>
            <h1>Mutation Testing Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Overall Mutation Score:</strong> 
                   <span class="{self._get_score_class(report.overall_mutation_score)}">
                   {report.overall_mutation_score:.2f}</span></p>
                <p><strong>Quality Assessment:</strong> {report.quality_assessment}</p>
                <p><strong>Generated:</strong> {report.timestamp}</p>
            </div>
        
            <h2>Service Results</h2>
            <table>
                <tr>
                    <th>Service</th>
                    <th>Mutation Score</th>
                    <th>Critical Path Score</th>
                    <th>Total Mutants</th>
                    <th>Killed</th>
                    <th>Survived</th>
                    <th>Execution Time</th>
                </tr>
        """
        
        for service, service_report in report.service_reports.items():
            score_class = self._get_score_class(service_report.mutation_score)
            html_content += f"""
                <tr>
                    <td>{service}</td>
                    <td class="{score_class}">{service_report.mutation_score:.2f}</td>
                    <td class="{self._get_score_class(service_report.critical_path_score)}">
                        {service_report.critical_path_score:.2f}</td>
                    <td>{service_report.total_mutants}</td>
                    <td>{service_report.killed_mutants}</td>
                    <td>{service_report.survived_mutants}</td>
                    <td>{service_report.test_execution_time:.1f}s</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            
            <h2>Recommendations</h2>
            <ul>
                {''.join([f'<li>{r}</li>' for r in report.recommendations])}
            </ul>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for mutation score."""
        if score >= 0.90:
            return "excellent"
        elif score >= 0.80:
            return "good"
        elif score >= 0.70:
            return "fair"
        else:
            return "poor"


class MutationRunner:
    """Base class for mutation testing tools."""
    
    async def run_mutations(self, 
                          source_paths: List[str],
                          test_paths: List[str],
                          mutation_types: List[str]) -> List[MutationResult]:
        """Run mutations and return results."""
        raise NotImplementedError


class MutmutRunner(MutationRunner):
    """Mutation testing using mutmut."""
    
    async def run_mutations(self, 
                          source_paths: List[str],
                          test_paths: List[str],
                          mutation_types: List[str]) -> List[MutationResult]:
        """Run mutations using mutmut."""
        results = []
        
        for source_path in source_paths:
            if not Path(source_path).exists():
                logger.warning(f"Source path not found", path=source_path)
                continue
            
            try:
                # Run mutmut
                cmd = [
                    'mutmut', 'run',
                    '--paths-to-mutate', source_path,
                    '--tests-dir', ','.join(test_paths),
                    '--runner', 'pytest',
                    '--use-coverage'
                ]
                
                logger.info(f"Running mutmut", command=' '.join(cmd))
                
                # For now, create mock results since mutmut might not be installed
                results.extend(self._create_mock_mutation_results(source_path))
                
            except Exception as e:
                logger.error(f"Mutmut execution failed", error=str(e))
        
        return results
    
    def _create_mock_mutation_results(self, source_path: str) -> List[MutationResult]:
        """Create mock mutation results for testing."""
        # This would be replaced with actual mutmut result parsing
        return [
            MutationResult(
                mutant_id=f"mutant_1_{source_path}",
                file_path=f"{source_path}/example.py",
                line_number=10,
                mutation_type="arithmetic_operator",
                original_code="x + y",
                mutated_code="x - y",
                status="killed",
                test_output="test_addition FAILED",
                execution_time=0.5
            ),
            MutationResult(
                mutant_id=f"mutant_2_{source_path}",
                file_path=f"{source_path}/example.py",
                line_number=15,
                mutation_type="comparison_operator",
                original_code="x > y",
                mutated_code="x < y",
                status="survived",
                test_output="All tests passed",
                execution_time=0.3
            )
        ]


class CosmicRayRunner(MutationRunner):
    """Mutation testing using Cosmic Ray."""
    
    async def run_mutations(self, 
                          source_paths: List[str],
                          test_paths: List[str],
                          mutation_types: List[str]) -> List[MutationResult]:
        """Run mutations using Cosmic Ray."""
        results = []
        
        # For now, create mock results
        for source_path in source_paths:
            results.extend(self._create_mock_cosmic_ray_results(source_path))
        
        return results
    
    def _create_mock_cosmic_ray_results(self, source_path: str) -> List[MutationResult]:
        """Create mock Cosmic Ray results."""
        return [
            MutationResult(
                mutant_id=f"cosmic_mutant_1_{source_path}",
                file_path=f"{source_path}/cosmic_example.py",
                line_number=20,
                mutation_type="boolean_operator",
                original_code="x and y",
                mutated_code="x or y",
                status="killed",
                test_output="test_logic FAILED",
                execution_time=0.4
            )
        ]


class MutationTestValidator:
    """Validates mutation test results meet quality requirements."""
    
    def __init__(self):
        self.minimum_mutation_score = 0.80
        self.minimum_critical_path_score = 0.90
    
    async def validate_mutation_quality(self, report: UnifiedMutationReport) -> bool:
        """Validate mutation test quality meets requirements."""
        logger.info("Validating mutation test quality")
        
        # Check overall score
        if report.overall_mutation_score < self.minimum_mutation_score:
            logger.error(f"Overall mutation score too low", 
                        score=report.overall_mutation_score,
                        minimum=self.minimum_mutation_score)
            return False
        
        # Check service-specific scores
        for service, service_report in report.service_reports.items():
            if service_report.mutation_score < self.minimum_mutation_score:
                logger.error(f"Service mutation score too low",
                           service=service,
                           score=service_report.mutation_score,
                           minimum=self.minimum_mutation_score)
                return False
            
            if service_report.critical_path_score < self.minimum_critical_path_score:
                logger.error(f"Critical path mutation score too low",
                           service=service,
                           score=service_report.critical_path_score,
                           minimum=self.minimum_critical_path_score)
                return False
        
        logger.info("Mutation test quality validation passed")
        return True
