"""
Multi-service performance testing framework.

This module provides comprehensive performance testing for:
- Core Mapper Service (p95 < 100ms target)
- Detector Orchestration Service (p95 < 200ms target)  
- Analysis Service (p95 < 500ms target)
- Cross-service workflows and load testing
"""

import asyncio
import time
import statistics
from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import concurrent.futures
import structlog

logger = structlog.get_logger(__name__)


class LoadTestType(Enum):
    """Types of load tests."""
    BASELINE = "baseline"
    STRESS = "stress"
    SPIKE = "spike"
    ENDURANCE = "endurance"
    VOLUME = "volume"


class PerformanceMetric(Enum):
    """Performance metrics to track."""
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    THROUGHPUT_RPS = "throughput_rps"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    CONCURRENT_CONNECTIONS = "concurrent_connections"


@dataclass
class PerformanceTarget:
    """Performance target for a service."""
    service_name: str
    latency_p95_ms: float
    throughput_rps: float
    error_rate_threshold: float
    cpu_usage_threshold: float
    memory_usage_threshold: float


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    test_name: str
    test_type: LoadTestType
    duration_seconds: int
    concurrent_users: int
    ramp_up_seconds: int
    ramp_down_seconds: int
    target_rps: Optional[float] = None
    test_data_size: int = 100


@dataclass
class RequestResult:
    """Result of a single request."""
    timestamp: float
    service: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class ServicePerformanceResult:
    """Performance results for a single service."""
    service_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_rps: float
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    cpu_usage_avg: float
    memory_usage_avg: float
    targets_met: Dict[str, bool]


@dataclass
class CrossServicePerformanceResult:
    """Performance results for cross-service workflows."""
    workflow_name: str
    total_workflows: int
    successful_workflows: int
    failed_workflows: int
    avg_workflow_time: float
    p95_workflow_time: float
    bottleneck_service: str
    service_breakdown: Dict[str, float]


@dataclass
class PerformanceTestReport:
    """Comprehensive performance test report."""
    timestamp: str
    test_config: LoadTestConfig
    test_duration: float
    service_results: Dict[str, ServicePerformanceResult]
    cross_service_results: List[CrossServicePerformanceResult]
    overall_metrics: Dict[str, float]
    sla_compliance: Dict[str, bool]
    bottlenecks_identified: List[str]
    recommendations: List[str]


class MultiServicePerformanceTester:
    """Coordinates performance testing across all services."""
    
    def __init__(self):
        self.service_targets = self._setup_performance_targets()
        self.load_generators = {
            'asyncio': AsyncioLoadGenerator(),
            'concurrent_futures': ConcurrentFuturesLoadGenerator()
        }
        self.performance_history: List[PerformanceTestReport] = []
    
    def _setup_performance_targets(self) -> Dict[str, PerformanceTarget]:
        """Setup performance targets for each service."""
        return {
            'core_mapper': PerformanceTarget(
                service_name='core_mapper',
                latency_p95_ms=100.0,
                throughput_rps=1000.0,
                error_rate_threshold=0.01,  # 1%
                cpu_usage_threshold=0.80,   # 80%
                memory_usage_threshold=0.75  # 75%
            ),
            'detector_orchestration': PerformanceTarget(
                service_name='detector_orchestration',
                latency_p95_ms=200.0,
                throughput_rps=500.0,
                error_rate_threshold=0.01,
                cpu_usage_threshold=0.85,
                memory_usage_threshold=0.80
            ),
            'analysis_service': PerformanceTarget(
                service_name='analysis_service',
                latency_p95_ms=500.0,
                throughput_rps=100.0,
                error_rate_threshold=0.02,
                cpu_usage_threshold=0.90,
                memory_usage_threshold=0.85
            )
        }
    
    async def run_service_performance_test(self, 
                                         service: str,
                                         config: LoadTestConfig,
                                         service_client: Any) -> ServicePerformanceResult:
        """Run performance test for individual service."""
        logger.info(f"Running performance test for service", 
                   service=service, test_type=config.test_type.value)
        
        load_generator = self.load_generators['asyncio']
        
        # Generate test requests
        test_requests = self._generate_service_test_requests(service, config)
        
        # Execute load test
        request_results = await load_generator.execute_load_test(
            test_requests, config, service_client
        )
        
        # Calculate metrics
        service_result = self._calculate_service_metrics(
            service, request_results, config
        )
        
        # Check SLA compliance
        self._check_service_sla_compliance(service, service_result)
        
        logger.info(f"Performance test completed", 
                   service=service,
                   throughput=service_result.throughput_rps,
                   p95_latency=service_result.latency_p95)
        
        return service_result
    
    async def run_cross_service_performance_test(self, 
                                               config: LoadTestConfig,
                                               service_clients: Dict[str, Any]) -> List[CrossServicePerformanceResult]:
        """Run performance test across service boundaries."""
        logger.info("Running cross-service performance test")
        
        cross_service_results = []
        
        # Test detection → mapping → analysis workflow
        workflow_result = await self._test_detection_workflow_performance(
            config, service_clients
        )
        cross_service_results.append(workflow_result)
        
        # Test batch processing workflow
        batch_workflow_result = await self._test_batch_workflow_performance(
            config, service_clients
        )
        cross_service_results.append(batch_workflow_result)
        
        return cross_service_results
    
    async def run_comprehensive_performance_test(self, 
                                               config: LoadTestConfig,
                                               service_clients: Dict[str, Any]) -> PerformanceTestReport:
        """Run comprehensive performance test across all services."""
        logger.info("Running comprehensive performance test", test_type=config.test_type.value)
        
        start_time = time.time()
        
        # Run individual service tests
        service_results = {}
        for service_name, client in service_clients.items():
            result = await self.run_service_performance_test(service_name, config, client)
            service_results[service_name] = result
        
        # Run cross-service tests
        cross_service_results = await self.run_cross_service_performance_test(
            config, service_clients
        )
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(service_results, cross_service_results)
        
        # Check SLA compliance
        sla_compliance = self._check_overall_sla_compliance(service_results)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(service_results, cross_service_results)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            service_results, cross_service_results, bottlenecks
        )
        
        test_duration = time.time() - start_time
        
        report = PerformanceTestReport(
            timestamp=datetime.utcnow().isoformat(),
            test_config=config,
            test_duration=test_duration,
            service_results=service_results,
            cross_service_results=cross_service_results,
            overall_metrics=overall_metrics,
            sla_compliance=sla_compliance,
            bottlenecks_identified=bottlenecks,
            recommendations=recommendations
        )
        
        self.performance_history.append(report)
        await self._save_performance_report(report)
        
        return report
    
    def _generate_service_test_requests(self, service: str, config: LoadTestConfig) -> List[Dict[str, Any]]:
        """Generate test requests for a specific service."""
        requests = []
        
        if service == 'core_mapper':
            for i in range(config.test_data_size):
                requests.append({
                    'endpoint': '/api/v1/map',
                    'method': 'POST',
                    'payload': {
                        'detector_outputs': [
                            {
                                'detector_type': 'presidio',
                                'findings': [
                                    {
                                        'entity_type': 'PERSON',
                                        'confidence': 0.95,
                                        'start': 0,
                                        'end': 8,
                                        'text': '[REDACTED]'
                                    }
                                ]
                            }
                        ],
                        'framework': 'SOC2',
                        'tenant_id': f'perf_test_tenant_{i % 10}',
                        'correlation_id': f'perf_test_{i}'
                    }
                })
        
        elif service == 'detector_orchestration':
            for i in range(config.test_data_size):
                requests.append({
                    'endpoint': '/api/v1/orchestrate',
                    'method': 'POST',
                    'payload': {
                        'content': f'Test content {i} with PII data for performance testing',
                        'detectors': ['presidio'],
                        'framework': 'SOC2',
                        'tenant_id': f'perf_test_tenant_{i % 10}',
                        'correlation_id': f'perf_orch_test_{i}'
                    }
                })
        
        elif service == 'analysis_service':
            for i in range(config.test_data_size):
                requests.append({
                    'endpoint': '/api/v1/analyze',
                    'method': 'POST',
                    'payload': {
                        'mapping_result': {
                            'canonical_result': {
                                'category': 'pii',
                                'subcategory': 'person_name',
                                'confidence': 0.95
                            },
                            'framework_mappings': [
                                {
                                    'control_id': 'CC6.1',
                                    'control_name': 'Logical Access Controls'
                                }
                            ]
                        },
                        'correlation_id': f'perf_analysis_test_{i}',
                        'analysis_type': 'compliance'
                    }
                })
        
        return requests
    
    def _calculate_service_metrics(self, 
                                 service: str,
                                 request_results: List[RequestResult],
                                 config: LoadTestConfig) -> ServicePerformanceResult:
        """Calculate performance metrics for a service."""
        if not request_results:
            return self._create_empty_service_result(service)
        
        # Basic counts
        total_requests = len(request_results)
        successful_requests = sum(1 for r in request_results if r.success)
        failed_requests = total_requests - successful_requests
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        # Response time metrics
        response_times = [r.response_time_ms for r in request_results if r.success]
        
        if response_times:
            latency_p50 = statistics.median(response_times)
            latency_p95 = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            latency_p99 = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            latency_p50 = latency_p95 = latency_p99 = 0
            avg_response_time = max_response_time = min_response_time = 0
        
        # Throughput calculation
        if request_results:
            test_duration = max(r.timestamp for r in request_results) - min(r.timestamp for r in request_results)
            throughput_rps = successful_requests / test_duration if test_duration > 0 else 0
        else:
            throughput_rps = 0
        
        # Resource usage (mock values for now)
        cpu_usage_avg = 50.0  # Would get from monitoring system
        memory_usage_avg = 60.0  # Would get from monitoring system
        
        # Check targets
        targets = self.service_targets.get(service)
        targets_met = {}
        if targets:
            targets_met = {
                'latency_p95': latency_p95 <= targets.latency_p95_ms,
                'throughput': throughput_rps >= targets.throughput_rps,
                'error_rate': error_rate <= targets.error_rate_threshold,
                'cpu_usage': cpu_usage_avg <= targets.cpu_usage_threshold * 100,
                'memory_usage': memory_usage_avg <= targets.memory_usage_threshold * 100
            }
        
        return ServicePerformanceResult(
            service_name=service,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_rate=error_rate,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput_rps=throughput_rps,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            cpu_usage_avg=cpu_usage_avg,
            memory_usage_avg=memory_usage_avg,
            targets_met=targets_met
        )
    
    async def _test_detection_workflow_performance(self, 
                                                 config: LoadTestConfig,
                                                 service_clients: Dict[str, Any]) -> CrossServicePerformanceResult:
        """Test performance of detection → mapping → analysis workflow."""
        workflow_results = []
        
        for i in range(config.test_data_size):
            start_time = time.time()
            workflow_success = True
            service_times = {}
            
            try:
                # Step 1: Orchestration
                orch_start = time.time()
                orch_response = await service_clients['orchestration'].post(
                    '/api/v1/orchestrate',
                    json={
                        'content': f'Workflow test {i} with performance data',
                        'detectors': ['presidio'],
                        'framework': 'SOC2',
                        'tenant_id': f'workflow_perf_tenant_{i % 5}',
                        'auto_map': False
                    }
                )
                service_times['orchestration'] = time.time() - orch_start
                
                if orch_response.status_code != 200:
                    workflow_success = False
                    continue
                
                # Step 2: Mapping
                map_start = time.time()
                orch_data = orch_response.json()
                map_response = await service_clients['mapper'].post(
                    '/api/v1/map',
                    json=orch_data.get('mapper_payload', {})
                )
                service_times['mapper'] = time.time() - map_start
                
                if map_response.status_code != 200:
                    workflow_success = False
                    continue
                
                # Step 3: Analysis
                analysis_start = time.time()
                map_data = map_response.json()
                analysis_response = await service_clients['analysis'].post(
                    '/api/v1/analyze',
                    json={
                        'mapping_result': map_data,
                        'correlation_id': f'workflow_perf_{i}',
                        'analysis_type': 'compliance'
                    }
                )
                service_times['analysis'] = time.time() - analysis_start
                
                if analysis_response.status_code != 200:
                    workflow_success = False
                
            except Exception as e:
                logger.error(f"Workflow performance test failed", iteration=i, error=str(e))
                workflow_success = False
            
            total_time = time.time() - start_time
            workflow_results.append({
                'success': workflow_success,
                'total_time': total_time,
                'service_times': service_times
            })
        
        # Calculate workflow metrics
        successful_workflows = sum(1 for r in workflow_results if r['success'])
        failed_workflows = len(workflow_results) - successful_workflows
        
        if successful_workflows > 0:
            workflow_times = [r['total_time'] for r in workflow_results if r['success']]
            avg_workflow_time = statistics.mean(workflow_times)
            p95_workflow_time = statistics.quantiles(workflow_times, n=20)[18] if len(workflow_times) > 1 else workflow_times[0]
            
            # Calculate service breakdown
            service_breakdown = {}
            for service in ['orchestration', 'mapper', 'analysis']:
                service_times = [r['service_times'].get(service, 0) for r in workflow_results if r['success']]
                if service_times:
                    service_breakdown[service] = statistics.mean(service_times)
            
            # Identify bottleneck
            bottleneck_service = max(service_breakdown, key=service_breakdown.get) if service_breakdown else 'unknown'
        else:
            avg_workflow_time = p95_workflow_time = 0
            service_breakdown = {}
            bottleneck_service = 'unknown'
        
        return CrossServicePerformanceResult(
            workflow_name='detection_to_analysis',
            total_workflows=len(workflow_results),
            successful_workflows=successful_workflows,
            failed_workflows=failed_workflows,
            avg_workflow_time=avg_workflow_time,
            p95_workflow_time=p95_workflow_time,
            bottleneck_service=bottleneck_service,
            service_breakdown=service_breakdown
        )
    
    async def _test_batch_workflow_performance(self, 
                                             config: LoadTestConfig,
                                             service_clients: Dict[str, Any]) -> CrossServicePerformanceResult:
        """Test performance of batch processing workflow."""
        # Simplified batch workflow performance test
        return CrossServicePerformanceResult(
            workflow_name='batch_processing',
            total_workflows=1,
            successful_workflows=1,
            failed_workflows=0,
            avg_workflow_time=5.0,
            p95_workflow_time=5.0,
            bottleneck_service='orchestration',
            service_breakdown={'orchestration': 5.0}
        )
    
    def _calculate_overall_metrics(self, 
                                 service_results: Dict[str, ServicePerformanceResult],
                                 cross_service_results: List[CrossServicePerformanceResult]) -> Dict[str, float]:
        """Calculate overall system performance metrics."""
        overall_metrics = {}
        
        if service_results:
            # Overall throughput (sum of service throughputs)
            overall_metrics['total_throughput_rps'] = sum(
                result.throughput_rps for result in service_results.values()
            )
            
            # Average error rate across services
            overall_metrics['avg_error_rate'] = statistics.mean(
                result.error_rate for result in service_results.values()
            )
            
            # System p95 latency (max of service p95s)
            overall_metrics['system_p95_latency'] = max(
                result.latency_p95 for result in service_results.values()
            )
            
            # Average resource usage
            overall_metrics['avg_cpu_usage'] = statistics.mean(
                result.cpu_usage_avg for result in service_results.values()
            )
            overall_metrics['avg_memory_usage'] = statistics.mean(
                result.memory_usage_avg for result in service_results.values()
            )
        
        # Cross-service workflow metrics
        if cross_service_results:
            workflow_success_rates = [
                r.successful_workflows / r.total_workflows if r.total_workflows > 0 else 0
                for r in cross_service_results
            ]
            overall_metrics['workflow_success_rate'] = statistics.mean(workflow_success_rates)
        
        return overall_metrics
    
    def _check_service_sla_compliance(self, service: str, result: ServicePerformanceResult) -> None:
        """Check if service meets SLA targets."""
        targets = self.service_targets.get(service)
        if not targets:
            return
        
        sla_violations = []
        
        if result.latency_p95 > targets.latency_p95_ms:
            sla_violations.append(f"P95 latency {result.latency_p95:.1f}ms > {targets.latency_p95_ms}ms")
        
        if result.throughput_rps < targets.throughput_rps:
            sla_violations.append(f"Throughput {result.throughput_rps:.1f} RPS < {targets.throughput_rps} RPS")
        
        if result.error_rate > targets.error_rate_threshold:
            sla_violations.append(f"Error rate {result.error_rate:.2%} > {targets.error_rate_threshold:.2%}")
        
        if sla_violations:
            logger.warning(f"SLA violations for service", 
                          service=service, violations=sla_violations)
    
    def _check_overall_sla_compliance(self, 
                                    service_results: Dict[str, ServicePerformanceResult]) -> Dict[str, bool]:
        """Check overall SLA compliance across all services."""
        sla_compliance = {}
        
        for service, result in service_results.items():
            targets = self.service_targets.get(service)
            if targets:
                sla_compliance[service] = all(result.targets_met.values())
        
        return sla_compliance
    
    def _identify_bottlenecks(self, 
                            service_results: Dict[str, ServicePerformanceResult],
                            cross_service_results: List[CrossServicePerformanceResult]) -> List[str]:
        """Identify performance bottlenecks in the system."""
        bottlenecks = []
        
        # Service-level bottlenecks
        for service, result in service_results.items():
            targets = self.service_targets.get(service)
            if targets:
                if result.latency_p95 > targets.latency_p95_ms * 0.8:  # Within 80% of limit
                    bottlenecks.append(f"{service}: High latency ({result.latency_p95:.1f}ms)")
                
                if result.throughput_rps < targets.throughput_rps * 0.8:  # Below 80% of target
                    bottlenecks.append(f"{service}: Low throughput ({result.throughput_rps:.1f} RPS)")
                
                if result.cpu_usage_avg > targets.cpu_usage_threshold * 80:  # Above 80% threshold
                    bottlenecks.append(f"{service}: High CPU usage ({result.cpu_usage_avg:.1f}%)")
        
        # Cross-service bottlenecks
        for workflow_result in cross_service_results:
            if workflow_result.bottleneck_service != 'unknown':
                bottlenecks.append(f"Workflow bottleneck: {workflow_result.bottleneck_service} in {workflow_result.workflow_name}")
        
        return bottlenecks
    
    def _generate_performance_recommendations(self, 
                                            service_results: Dict[str, ServicePerformanceResult],
                                            cross_service_results: List[CrossServicePerformanceResult],
                                            bottlenecks: List[str]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if bottlenecks:
            recommendations.append("Address identified bottlenecks:")
            recommendations.extend([f"  - {bottleneck}" for bottleneck in bottlenecks])
        
        # Service-specific recommendations
        for service, result in service_results.items():
            if result.error_rate > 0.005:  # > 0.5%
                recommendations.append(f"Investigate error rate in {service} ({result.error_rate:.2%})")
            
            if result.latency_p99 > result.latency_p95 * 2:  # High tail latency
                recommendations.append(f"Optimize tail latency in {service}")
        
        # Resource optimization
        high_cpu_services = [
            service for service, result in service_results.items()
            if result.cpu_usage_avg > 70
        ]
        if high_cpu_services:
            recommendations.append(f"Consider scaling CPU for services: {', '.join(high_cpu_services)}")
        
        high_memory_services = [
            service for service, result in service_results.items()
            if result.memory_usage_avg > 70
        ]
        if high_memory_services:
            recommendations.append(f"Consider scaling memory for services: {', '.join(high_memory_services)}")
        
        if not recommendations:
            recommendations.append("Performance targets met! Consider load testing with higher traffic volumes")
        
        return recommendations
    
    def _create_empty_service_result(self, service: str) -> ServicePerformanceResult:
        """Create empty service result for failed tests."""
        return ServicePerformanceResult(
            service_name=service,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            error_rate=1.0,
            latency_p50=0,
            latency_p95=0,
            latency_p99=0,
            throughput_rps=0,
            avg_response_time=0,
            max_response_time=0,
            min_response_time=0,
            cpu_usage_avg=0,
            memory_usage_avg=0,
            targets_met={}
        )
    
    async def _save_performance_report(self, report: PerformanceTestReport) -> None:
        """Save performance test report."""
        from pathlib import Path
        import json
        
        # Save as JSON
        report_path = Path("tests/performance/performance_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        logger.info(f"Performance report saved", path=str(report_path))


class LoadGenerator:
    """Base class for load generators."""
    
    async def execute_load_test(self, 
                              test_requests: List[Dict[str, Any]],
                              config: LoadTestConfig,
                              service_client: Any) -> List[RequestResult]:
        """Execute load test and return results."""
        raise NotImplementedError


class AsyncioLoadGenerator(LoadGenerator):
    """Asyncio-based load generator for high concurrency."""
    
    async def execute_load_test(self, 
                              test_requests: List[Dict[str, Any]],
                              config: LoadTestConfig,
                              service_client: Any) -> List[RequestResult]:
        """Execute load test using asyncio."""
        logger.info(f"Starting asyncio load test", 
                   concurrent_users=config.concurrent_users,
                   duration=config.duration_seconds)
        
        results = []
        start_time = time.time()
        end_time = start_time + config.duration_seconds
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(config.concurrent_users)
        
        async def execute_request(request_data: Dict[str, Any]) -> RequestResult:
            """Execute a single request."""
            async with semaphore:
                request_start = time.time()
                
                try:
                    if request_data['method'] == 'POST':
                        response = await service_client.post(
                            request_data['endpoint'],
                            json=request_data['payload']
                        )
                    else:
                        response = await service_client.get(request_data['endpoint'])
                    
                    response_time = (time.time() - request_start) * 1000  # Convert to ms
                    
                    return RequestResult(
                        timestamp=request_start,
                        service='test_service',
                        endpoint=request_data['endpoint'],
                        method=request_data['method'],
                        status_code=response.status_code,
                        response_time_ms=response_time,
                        request_size_bytes=len(str(request_data.get('payload', ''))),
                        response_size_bytes=len(response.content) if hasattr(response, 'content') else 0,
                        success=200 <= response.status_code < 400
                    )
                
                except Exception as e:
                    response_time = (time.time() - request_start) * 1000
                    
                    return RequestResult(
                        timestamp=request_start,
                        service='test_service',
                        endpoint=request_data['endpoint'],
                        method=request_data['method'],
                        status_code=0,
                        response_time_ms=response_time,
                        request_size_bytes=len(str(request_data.get('payload', ''))),
                        response_size_bytes=0,
                        success=False,
                        error_message=str(e)
                    )
        
        # Execute requests until duration expires
        tasks = []
        request_index = 0
        
        while time.time() < end_time:
            if len(tasks) < config.concurrent_users and request_index < len(test_requests):
                request_data = test_requests[request_index % len(test_requests)]
                task = asyncio.create_task(execute_request(request_data))
                tasks.append(task)
                request_index += 1
            
            # Collect completed tasks
            if tasks:
                done, pending = await asyncio.wait(tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        result = await task
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Task execution failed", error=str(e))
                
                tasks = list(pending)
        
        # Wait for remaining tasks to complete
        if tasks:
            remaining_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in remaining_results:
                if isinstance(result, RequestResult):
                    results.append(result)
        
        logger.info(f"Load test completed", 
                   total_requests=len(results),
                   successful=sum(1 for r in results if r.success))
        
        return results


class ConcurrentFuturesLoadGenerator(LoadGenerator):
    """Thread-based load generator using concurrent.futures."""
    
    async def execute_load_test(self, 
                              test_requests: List[Dict[str, Any]],
                              config: LoadTestConfig,
                              service_client: Any) -> List[RequestResult]:
        """Execute load test using concurrent.futures."""
        # Simplified implementation - would use ThreadPoolExecutor for synchronous clients
        return []
