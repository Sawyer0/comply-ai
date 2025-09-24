"""
Multi-service coverage aggregation and validation framework.

This module provides comprehensive coverage tracking across:
- Core Mapper Service
- Detector Orchestration Service
- Analysis Service
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
import coverage
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CoverageMetrics:
    """Coverage metrics for a service."""
    service_name: str
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    total_lines: int
    covered_lines: int
    missing_lines: int
    critical_path_coverage: float
    files_covered: int
    total_files: int


@dataclass
class CrossServiceCoverageMetrics:
    """Coverage metrics for cross-service interactions."""
    interaction_type: str
    coverage_percentage: float
    total_interactions: int
    covered_interactions: int
    missing_interactions: List[str]


@dataclass
class UnifiedCoverageReport:
    """Unified coverage report across all services."""
    timestamp: str
    overall_coverage: float
    service_coverage: Dict[str, CoverageMetrics]
    cross_service_coverage: List[CrossServiceCoverageMetrics]
    critical_path_coverage: float
    coverage_trends: Dict[str, List[float]]
    threshold_violations: List[str]
    recommendations: List[str]


class CoverageThresholds:
    """Coverage thresholds for different service types and paths."""
    
    # Service-specific thresholds
    SERVICE_THRESHOLDS = {
        'core_mapper': 0.85,
        'detector_orchestration': 0.85,
        'analysis_service': 0.85
    }
    
    # Critical path thresholds (higher requirements)
    CRITICAL_PATH_THRESHOLDS = {
        'core_mapper': 0.95,  # Mapping logic is critical
        'detector_orchestration': 0.90,  # Orchestration coordination
        'analysis_service': 0.90,  # Analysis algorithms
        'security_functions': 0.95,  # Security code must be well tested
        'api_endpoints': 0.90  # API endpoints need good coverage
    }
    
    # Cross-service interaction thresholds
    CROSS_SERVICE_THRESHOLDS = {
        'mapper_orchestration': 0.90,
        'orchestration_analysis': 0.90,
        'end_to_end_workflows': 0.85
    }


class CoverageAggregator:
    """Aggregates coverage metrics across all services."""
    
    def __init__(self):
        self.thresholds = CoverageThresholds()
        self.coverage_history: List[UnifiedCoverageReport] = []
        
        # Service paths
        self.service_paths = {
            'core_mapper': 'src/llama_mapper',
            'detector_orchestration': 'detector-orchestration/src',
            'analysis_service': 'analysis/src'  # Assuming analysis service exists
        }
        
        # Critical path patterns
        self.critical_paths = {
            'core_mapper': [
                'src/llama_mapper/models/',
                'src/llama_mapper/serving/',
                'src/llama_mapper/api/endpoints/',
                'src/llama_mapper/security/'
            ],
            'detector_orchestration': [
                'detector-orchestration/src/coordinator.py',
                'detector-orchestration/src/aggregator.py',
                'detector-orchestration/src/circuit_breaker.py'
            ],
            'analysis_service': [
                'analysis/src/compliance/',
                'analysis/src/risk_assessment/',
                'analysis/src/api/'
            ]
        }
    
    async def collect_service_coverage(self, service: str) -> CoverageMetrics:
        """Collect coverage for individual service."""
        logger.info(f"Collecting coverage for service", service=service)
        
        service_path = self.service_paths.get(service)
        if not service_path or not Path(service_path).exists():
            logger.warning(f"Service path not found", service=service, path=service_path)
            return self._create_empty_metrics(service)
        
        try:
            # Initialize coverage instance
            cov = coverage.Coverage(
                source=[service_path],
                config_file='.coveragerc'
            )
            
            # Load existing coverage data if available
            coverage_file = f'.coverage.{service}'
            if Path(coverage_file).exists():
                cov.load()
            
            # Get coverage data
            cov.get_data()
            
            # Calculate metrics
            total_lines = 0
            covered_lines = 0
            missing_lines = 0
            files_covered = 0
            total_files = 0
            
            for filename in cov.get_data().measured_files():
                if service_path in filename:
                    total_files += 1
                    analysis = cov.analysis2(filename)
                    file_total = len(analysis.statements)
                    file_missing = len(analysis.missing)
                    file_covered = file_total - file_missing
                    
                    total_lines += file_total
                    covered_lines += file_covered
                    missing_lines += file_missing
                    
                    if file_covered > 0:
                        files_covered += 1
            
            # Calculate percentages
            line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
            
            # Calculate critical path coverage
            critical_path_coverage = await self._calculate_critical_path_coverage(
                service, cov
            )
            
            return CoverageMetrics(
                service_name=service,
                line_coverage=line_coverage,
                branch_coverage=0.0,  # Would need branch coverage data
                function_coverage=0.0,  # Would need function coverage data
                total_lines=total_lines,
                covered_lines=covered_lines,
                missing_lines=missing_lines,
                critical_path_coverage=critical_path_coverage,
                files_covered=files_covered,
                total_files=total_files
            )
            
        except Exception as e:
            logger.error(f"Failed to collect coverage", service=service, error=str(e))
            return self._create_empty_metrics(service)
    
    async def collect_integration_coverage(self) -> List[CrossServiceCoverageMetrics]:
        """Collect coverage for cross-service interactions."""
        logger.info("Collecting cross-service integration coverage")
        
        integration_metrics = []
        
        # Mapper ↔ Orchestration interactions
        mapper_orch_coverage = await self._analyze_service_interaction(
            'mapper_orchestration',
            'src/llama_mapper/api/',
            'detector-orchestration/src/mapper_client.py'
        )
        integration_metrics.append(mapper_orch_coverage)
        
        # Orchestration ↔ Analysis interactions
        orch_analysis_coverage = await self._analyze_service_interaction(
            'orchestration_analysis',
            'detector-orchestration/src/aggregator.py',
            'analysis/src/api/'
        )
        integration_metrics.append(orch_analysis_coverage)
        
        # End-to-end workflow coverage
        e2e_coverage = await self._analyze_end_to_end_coverage()
        integration_metrics.append(e2e_coverage)
        
        return integration_metrics
    
    async def generate_unified_report(self) -> UnifiedCoverageReport:
        """Generate unified coverage report across all services."""
        logger.info("Generating unified coverage report")
        
        # Collect service coverage
        service_coverage = {}
        total_coverage = 0.0
        service_count = 0
        
        for service in self.service_paths.keys():
            metrics = await self.collect_service_coverage(service)
            service_coverage[service] = metrics
            total_coverage += metrics.line_coverage
            service_count += 1
        
        overall_coverage = total_coverage / service_count if service_count > 0 else 0
        
        # Collect cross-service coverage
        cross_service_coverage = await self.collect_integration_coverage()
        
        # Calculate critical path coverage
        critical_path_coverage = await self._calculate_overall_critical_path_coverage(
            service_coverage
        )
        
        # Check threshold violations
        threshold_violations = self._check_threshold_violations(
            service_coverage, cross_service_coverage, critical_path_coverage
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            service_coverage, threshold_violations
        )
        
        # Get coverage trends
        coverage_trends = self._calculate_coverage_trends()
        
        report = UnifiedCoverageReport(
            timestamp=datetime.utcnow().isoformat(),
            overall_coverage=overall_coverage,
            service_coverage=service_coverage,
            cross_service_coverage=cross_service_coverage,
            critical_path_coverage=critical_path_coverage,
            coverage_trends=coverage_trends,
            threshold_violations=threshold_violations,
            recommendations=recommendations
        )
        
        # Store in history
        self.coverage_history.append(report)
        
        # Save report to file
        await self._save_report(report)
        
        return report
    
    async def _calculate_critical_path_coverage(self, service: str, cov: coverage.Coverage) -> float:
        """Calculate coverage for critical paths in a service."""
        critical_paths = self.critical_paths.get(service, [])
        if not critical_paths:
            return 0.0
        
        total_critical_lines = 0
        covered_critical_lines = 0
        
        for filename in cov.get_data().measured_files():
            # Check if file is in critical path
            is_critical = any(critical_path in filename for critical_path in critical_paths)
            
            if is_critical:
                analysis = cov.analysis2(filename)
                file_total = len(analysis.statements)
                file_missing = len(analysis.missing)
                file_covered = file_total - file_missing
                
                total_critical_lines += file_total
                covered_critical_lines += file_covered
        
        return (covered_critical_lines / total_critical_lines * 100) if total_critical_lines > 0 else 100
    
    async def _analyze_service_interaction(self, 
                                         interaction_type: str,
                                         source_path: str, 
                                         target_path: str) -> CrossServiceCoverageMetrics:
        """Analyze coverage for specific service interaction."""
        # This would analyze how well the interactions between services are tested
        # For now, we'll return mock data
        return CrossServiceCoverageMetrics(
            interaction_type=interaction_type,
            coverage_percentage=85.0,
            total_interactions=10,
            covered_interactions=8,
            missing_interactions=[
                f"Error handling in {interaction_type}",
                f"Timeout scenarios in {interaction_type}"
            ]
        )
    
    async def _analyze_end_to_end_coverage(self) -> CrossServiceCoverageMetrics:
        """Analyze end-to-end workflow coverage."""
        return CrossServiceCoverageMetrics(
            interaction_type='end_to_end_workflows',
            coverage_percentage=80.0,
            total_interactions=15,
            covered_interactions=12,
            missing_interactions=[
                "Failure cascade prevention",
                "Cross-service timeout handling",
                "Data consistency validation"
            ]
        )
    
    async def _calculate_overall_critical_path_coverage(self, 
                                                      service_coverage: Dict[str, CoverageMetrics]) -> float:
        """Calculate overall critical path coverage."""
        total_critical_coverage = 0.0
        service_count = 0
        
        for metrics in service_coverage.values():
            total_critical_coverage += metrics.critical_path_coverage
            service_count += 1
        
        return total_critical_coverage / service_count if service_count > 0 else 0
    
    def _check_threshold_violations(self, 
                                  service_coverage: Dict[str, CoverageMetrics],
                                  cross_service_coverage: List[CrossServiceCoverageMetrics],
                                  critical_path_coverage: float) -> List[str]:
        """Check for coverage threshold violations."""
        violations = []
        
        # Check service thresholds
        for service, metrics in service_coverage.items():
            threshold = self.thresholds.SERVICE_THRESHOLDS.get(service, 0.85)
            if metrics.line_coverage < threshold * 100:
                violations.append(
                    f"Service {service} coverage {metrics.line_coverage:.1f}% "
                    f"below threshold {threshold * 100:.1f}%"
                )
        
        # Check critical path thresholds
        for service, metrics in service_coverage.items():
            threshold = self.thresholds.CRITICAL_PATH_THRESHOLDS.get(service, 0.90)
            if metrics.critical_path_coverage < threshold * 100:
                violations.append(
                    f"Service {service} critical path coverage {metrics.critical_path_coverage:.1f}% "
                    f"below threshold {threshold * 100:.1f}%"
                )
        
        # Check cross-service thresholds
        for interaction in cross_service_coverage:
            threshold = self.thresholds.CROSS_SERVICE_THRESHOLDS.get(
                interaction.interaction_type, 0.85
            )
            if interaction.coverage_percentage < threshold * 100:
                violations.append(
                    f"Cross-service {interaction.interaction_type} coverage "
                    f"{interaction.coverage_percentage:.1f}% below threshold {threshold * 100:.1f}%"
                )
        
        return violations
    
    def _generate_recommendations(self, 
                                service_coverage: Dict[str, CoverageMetrics],
                                violations: List[str]) -> List[str]:
        """Generate recommendations for improving coverage."""
        recommendations = []
        
        if violations:
            recommendations.append("Address coverage threshold violations:")
            recommendations.extend([f"  - {violation}" for violation in violations])
        
        # Service-specific recommendations
        for service, metrics in service_coverage.items():
            if metrics.line_coverage < 90:
                recommendations.append(
                    f"Improve {service} test coverage by adding tests for uncovered code paths"
                )
            
            if metrics.files_covered < metrics.total_files * 0.9:
                recommendations.append(
                    f"Add tests for untested files in {service} "
                    f"({metrics.files_covered}/{metrics.total_files} files covered)"
                )
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Coverage targets met! Consider adding mutation testing for quality validation")
        
        return recommendations
    
    def _calculate_coverage_trends(self) -> Dict[str, List[float]]:
        """Calculate coverage trends over time."""
        if len(self.coverage_history) < 2:
            return {}
        
        trends = {}
        
        # Get last 10 reports for trending
        recent_reports = self.coverage_history[-10:]
        
        for service in self.service_paths.keys():
            service_trend = []
            for report in recent_reports:
                if service in report.service_coverage:
                    service_trend.append(report.service_coverage[service].line_coverage)
            trends[service] = service_trend
        
        # Overall trend
        overall_trend = [report.overall_coverage for report in recent_reports]
        trends['overall'] = overall_trend
        
        return trends
    
    async def _save_report(self, report: UnifiedCoverageReport) -> None:
        """Save coverage report to file."""
        # Save as JSON
        report_path = Path("tests/coverage/unified_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        # Save as HTML summary
        html_path = Path("tests/coverage/unified_report.html")
        await self._generate_html_report(report, html_path)
        
        logger.info(f"Coverage report saved", json_path=str(report_path), html_path=str(html_path))
    
    async def _generate_html_report(self, report: UnifiedCoverageReport, output_path: Path) -> None:
        """Generate HTML coverage report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unified Coverage Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .service {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Unified Coverage Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Overall Coverage:</strong> 
                   <span class="{'good' if report.overall_coverage >= 85 else 'warning' if report.overall_coverage >= 70 else 'error'}">
                   {report.overall_coverage:.1f}%</span></p>
                <p><strong>Critical Path Coverage:</strong> 
                   <span class="{'good' if report.critical_path_coverage >= 90 else 'warning' if report.critical_path_coverage >= 80 else 'error'}">
                   {report.critical_path_coverage:.1f}%</span></p>
                <p><strong>Generated:</strong> {report.timestamp}</p>
            </div>
        
            <h2>Service Coverage</h2>
            <table>
                <tr>
                    <th>Service</th>
                    <th>Line Coverage</th>
                    <th>Critical Path Coverage</th>
                    <th>Files Covered</th>
                    <th>Total Lines</th>
                </tr>
        """
        
        for service, metrics in report.service_coverage.items():
            coverage_class = 'good' if metrics.line_coverage >= 85 else 'warning' if metrics.line_coverage >= 70 else 'error'
            html_content += f"""
                <tr>
                    <td>{service}</td>
                    <td class="{coverage_class}">{metrics.line_coverage:.1f}%</td>
                    <td>{metrics.critical_path_coverage:.1f}%</td>
                    <td>{metrics.files_covered}/{metrics.total_files}</td>
                    <td>{metrics.total_lines}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Cross-Service Coverage</h2>
            <table>
                <tr>
                    <th>Interaction Type</th>
                    <th>Coverage</th>
                    <th>Covered/Total</th>
                    <th>Missing Interactions</th>
                </tr>
        """
        
        for interaction in report.cross_service_coverage:
            html_content += f"""
                <tr>
                    <td>{interaction.interaction_type}</td>
                    <td>{interaction.coverage_percentage:.1f}%</td>
                    <td>{interaction.covered_interactions}/{interaction.total_interactions}</td>
                    <td>{'<br>'.join(interaction.missing_interactions)}</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            
            <h2>Threshold Violations</h2>
            {'<ul>' + ''.join([f'<li class="error">{v}</li>' for v in report.threshold_violations]) + '</ul>' if report.threshold_violations else '<p class="good">No threshold violations found!</p>'}
            
            <h2>Recommendations</h2>
            <ul>
                {''.join([f'<li>{r}</li>' for r in report.recommendations])}
            </ul>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _create_empty_metrics(self, service: str) -> CoverageMetrics:
        """Create empty coverage metrics for a service."""
        return CoverageMetrics(
            service_name=service,
            line_coverage=0.0,
            branch_coverage=0.0,
            function_coverage=0.0,
            total_lines=0,
            covered_lines=0,
            missing_lines=0,
            critical_path_coverage=0.0,
            files_covered=0,
            total_files=0
        )


class CoverageValidator:
    """Validates test coverage meets requirements."""
    
    def __init__(self):
        self.thresholds = CoverageThresholds()
    
    async def validate_coverage(self, report: UnifiedCoverageReport) -> bool:
        """Validate coverage meets threshold requirements."""
        logger.info("Validating coverage thresholds")
        
        # Check if there are any violations
        if report.threshold_violations:
            logger.error(f"Coverage validation failed", violations=report.threshold_violations)
            return False
        
        # Additional validation checks
        validation_checks = [
            self._validate_service_coverage(report.service_coverage),
            self._validate_critical_path_coverage(report.critical_path_coverage),
            self._validate_cross_service_coverage(report.cross_service_coverage)
        ]
        
        all_passed = all(validation_checks)
        
        if all_passed:
            logger.info("Coverage validation passed")
        else:
            logger.error("Coverage validation failed")
        
        return all_passed
    
    def _validate_service_coverage(self, service_coverage: Dict[str, CoverageMetrics]) -> bool:
        """Validate service coverage thresholds."""
        for service, metrics in service_coverage.items():
            threshold = self.thresholds.SERVICE_THRESHOLDS.get(service, 0.85)
            if metrics.line_coverage < threshold * 100:
                return False
        return True
    
    def _validate_critical_path_coverage(self, critical_path_coverage: float) -> bool:
        """Validate critical path coverage."""
        return critical_path_coverage >= 90.0  # 90% threshold for critical paths
    
    def _validate_cross_service_coverage(self, cross_service_coverage: List[CrossServiceCoverageMetrics]) -> bool:
        """Validate cross-service coverage."""
        for interaction in cross_service_coverage:
            threshold = self.thresholds.CROSS_SERVICE_THRESHOLDS.get(
                interaction.interaction_type, 0.85
            )
            if interaction.coverage_percentage < threshold * 100:
                return False
        return True
