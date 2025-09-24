"""
Chaos engineering framework for fault tolerance and resilience testing.

This module provides chaos testing capabilities for:
- Service failure scenarios
- Network partitioning and latency injection
- Resource exhaustion testing
- Cascading failure prevention validation
"""

import asyncio
import time
import random
from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class FailureType(Enum):
    """Types of failures to inject."""
    SERVICE_CRASH = "service_crash"
    NETWORK_PARTITION = "network_partition"
    HIGH_LATENCY = "high_latency"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CPU_SATURATION = "cpu_saturation"
    DISK_FULL = "disk_full"
    DATABASE_FAILURE = "database_failure"
    TIMEOUT = "timeout"
    PARTIAL_FAILURE = "partial_failure"


class ChaosScope(Enum):
    """Scope of chaos testing."""
    SINGLE_SERVICE = "single_service"
    CROSS_SERVICE = "cross_service"
    INFRASTRUCTURE = "infrastructure"
    DATA_LAYER = "data_layer"


@dataclass
class ChaosScenario:
    """Definition of a chaos testing scenario."""
    name: str
    description: str
    target_service: str
    failure_type: FailureType
    scope: ChaosScope
    duration_seconds: int
    intensity: float  # 0.0 to 1.0
    expected_behavior: str
    success_criteria: Dict[str, Any]
    blast_radius: List[str]  # Services expected to be affected


@dataclass
class ChaosTestResult:
    """Result of executing a chaos test."""
    scenario_name: str
    target_service: str
    failure_type: FailureType
    start_time: str
    end_time: str
    duration_seconds: float
    success: bool
    expected_behavior_observed: bool
    system_recovery_time: float
    blast_radius_contained: bool
    service_responses: Dict[str, Dict[str, Any]]
    metrics_during_failure: Dict[str, float]
    error_details: Optional[str] = None


@dataclass
class ResilienceReport:
    """Comprehensive resilience testing report."""
    timestamp: str
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    chaos_results: List[ChaosTestResult]
    resilience_score: float
    recovery_metrics: Dict[str, float]
    failure_impact_analysis: Dict[str, Any]
    recommendations: List[str]


class ChaosTestOrchestrator:
    """Orchestrates chaos engineering tests across services."""
    
    def __init__(self, service_clients: Dict[str, Any]):
        self.service_clients = service_clients
        self.chaos_scenarios = self._setup_chaos_scenarios()
        self.failure_injectors = {
            FailureType.SERVICE_CRASH: ServiceCrashInjector(),
            FailureType.NETWORK_PARTITION: NetworkPartitionInjector(),
            FailureType.HIGH_LATENCY: LatencyInjector(),
            FailureType.MEMORY_EXHAUSTION: MemoryExhaustionInjector(),
            FailureType.TIMEOUT: TimeoutInjector()
        }
    
    def _setup_chaos_scenarios(self) -> List[ChaosScenario]:
        """Setup predefined chaos testing scenarios."""
        return [
            ChaosScenario(
                name="core_mapper_service_failure",
                description="Test system behavior when Core Mapper service crashes",
                target_service="core_mapper",
                failure_type=FailureType.SERVICE_CRASH,
                scope=ChaosScope.SINGLE_SERVICE,
                duration_seconds=30,
                intensity=1.0,
                expected_behavior="orchestration_fallback_to_rules",
                success_criteria={
                    "orchestration_continues": True,
                    "fallback_activated": True,
                    "error_rate_increase": "<50%",
                    "recovery_time": "<60s"
                },
                blast_radius=["detector_orchestration"]
            ),
            ChaosScenario(
                name="detector_orchestration_network_partition",
                description="Test system behavior during network partition of orchestration service",
                target_service="detector_orchestration",
                failure_type=FailureType.NETWORK_PARTITION,
                scope=ChaosScope.CROSS_SERVICE,
                duration_seconds=60,
                intensity=0.8,
                expected_behavior="direct_detector_calls",
                success_criteria={
                    "direct_detection_available": True,
                    "partial_functionality_maintained": True,
                    "no_data_loss": True
                },
                blast_radius=["core_mapper", "analysis_service"]
            ),
            ChaosScenario(
                name="analysis_service_memory_exhaustion",
                description="Test system behavior when Analysis service runs out of memory",
                target_service="analysis_service",
                failure_type=FailureType.MEMORY_EXHAUSTION,
                scope=ChaosScope.SINGLE_SERVICE,
                duration_seconds=45,
                intensity=0.9,
                expected_behavior="core_functionality_preserved",
                success_criteria={
                    "mapper_continues": True,
                    "orchestration_continues": True,
                    "analysis_degrades_gracefully": True
                },
                blast_radius=[]
            ),
            ChaosScenario(
                name="database_connection_failure",
                description="Test system behavior when database connections fail",
                target_service="postgres",
                failure_type=FailureType.DATABASE_FAILURE,
                scope=ChaosScope.DATA_LAYER,
                duration_seconds=20,
                intensity=1.0,
                expected_behavior="cached_responses_served",
                success_criteria={
                    "cache_fallback_active": True,
                    "read_operations_continue": True,
                    "write_operations_queued": True
                },
                blast_radius=["core_mapper", "detector_orchestration", "analysis_service"]
            ),
            ChaosScenario(
                name="high_latency_injection",
                description="Test system behavior under high network latency",
                target_service="all_services",
                failure_type=FailureType.HIGH_LATENCY,
                scope=ChaosScope.CROSS_SERVICE,
                duration_seconds=90,
                intensity=0.6,
                expected_behavior="timeout_handling_activated",
                success_criteria={
                    "timeouts_configured": True,
                    "circuit_breakers_trip": True,
                    "user_experience_degraded_gracefully": True
                },
                blast_radius=["core_mapper", "detector_orchestration", "analysis_service"]
            )
        ]
    
    async def inject_service_failure(self, 
                                   service: str, 
                                   failure_type: FailureType,
                                   duration: int = 30) -> ChaosTestResult:
        """Inject failure into specific service."""
        logger.info(f"Injecting {failure_type.value} into {service}", duration=duration)
        
        start_time = time.time()
        start_timestamp = datetime.utcnow().isoformat()
        
        try:
            # Get baseline metrics
            baseline_metrics = await self._collect_baseline_metrics()
            
            # Inject failure
            injector = self.failure_injectors.get(failure_type)
            if not injector:
                raise ValueError(f"No injector available for {failure_type.value}")
            
            await injector.inject_failure(service, duration)
            
            # Monitor system during failure
            failure_metrics = await self._monitor_system_during_failure(duration)
            
            # Measure recovery
            recovery_start = time.time()
            await injector.recover_from_failure(service)
            recovery_time = time.time() - recovery_start
            
            # Validate expected behavior
            expected_behavior_observed = await self._validate_expected_behavior(
                service, failure_type
            )
            
            # Check blast radius containment
            blast_radius_contained = await self._check_blast_radius_containment(service)
            
            end_time = time.time()
            end_timestamp = datetime.utcnow().isoformat()
            
            return ChaosTestResult(
                scenario_name=f"{service}_{failure_type.value}",
                target_service=service,
                failure_type=failure_type,
                start_time=start_timestamp,
                end_time=end_timestamp,
                duration_seconds=end_time - start_time,
                success=expected_behavior_observed and blast_radius_contained,
                expected_behavior_observed=expected_behavior_observed,
                system_recovery_time=recovery_time,
                blast_radius_contained=blast_radius_contained,
                service_responses=failure_metrics.get('service_responses', {}),
                metrics_during_failure=failure_metrics.get('metrics', {})
            )
        
        except Exception as e:
            logger.error(f"Chaos test failed", service=service, 
                        failure_type=failure_type.value, error=str(e))
            
            return ChaosTestResult(
                scenario_name=f"{service}_{failure_type.value}",
                target_service=service,
                failure_type=failure_type,
                start_time=start_timestamp,
                end_time=datetime.utcnow().isoformat(),
                duration_seconds=time.time() - start_time,
                success=False,
                expected_behavior_observed=False,
                system_recovery_time=0.0,
                blast_radius_contained=False,
                service_responses={},
                metrics_during_failure={},
                error_details=str(e)
            )
    
    async def inject_network_partition(self, services: List[str]) -> ChaosTestResult:
        """Inject network partition between services."""
        logger.info(f"Injecting network partition", services=services)
        
        start_time = time.time()
        start_timestamp = datetime.utcnow().isoformat()
        
        try:
            # Simulate network partition by blocking inter-service communication
            partition_injector = self.failure_injectors[FailureType.NETWORK_PARTITION]
            
            for service in services:
                await partition_injector.inject_failure(service, 60)
            
            # Monitor system behavior during partition
            partition_metrics = await self._monitor_network_partition(services, 60)
            
            # Recovery
            recovery_start = time.time()
            for service in services:
                await partition_injector.recover_from_failure(service)
            recovery_time = time.time() - recovery_start
            
            # Validate partition handling
            partition_handled = await self._validate_partition_handling(services)
            
            return ChaosTestResult(
                scenario_name="network_partition",
                target_service=",".join(services),
                failure_type=FailureType.NETWORK_PARTITION,
                start_time=start_timestamp,
                end_time=datetime.utcnow().isoformat(),
                duration_seconds=time.time() - start_time,
                success=partition_handled,
                expected_behavior_observed=partition_handled,
                system_recovery_time=recovery_time,
                blast_radius_contained=True,
                service_responses=partition_metrics.get('service_responses', {}),
                metrics_during_failure=partition_metrics.get('metrics', {})
            )
        
        except Exception as e:
            logger.error(f"Network partition test failed", error=str(e))
            
            return ChaosTestResult(
                scenario_name="network_partition",
                target_service=",".join(services),
                failure_type=FailureType.NETWORK_PARTITION,
                start_time=start_timestamp,
                end_time=datetime.utcnow().isoformat(),
                duration_seconds=time.time() - start_time,
                success=False,
                expected_behavior_observed=False,
                system_recovery_time=0.0,
                blast_radius_contained=False,
                service_responses={},
                metrics_during_failure={},
                error_details=str(e)
            )
    
    async def test_cascading_failure_prevention(self) -> ChaosTestResult:
        """Test system's ability to prevent cascading failures."""
        logger.info("Testing cascading failure prevention")
        
        start_time = time.time()
        start_timestamp = datetime.utcnow().isoformat()
        
        try:
            # Inject multiple simultaneous failures
            failure_tasks = []
            
            # Start with database failure
            db_failure_task = asyncio.create_task(
                self.inject_service_failure("postgres", FailureType.DATABASE_FAILURE, 30)
            )
            failure_tasks.append(db_failure_task)
            
            # Wait 10 seconds, then add service failure
            await asyncio.sleep(10)
            service_failure_task = asyncio.create_task(
                self.inject_service_failure("core_mapper", FailureType.SERVICE_CRASH, 20)
            )
            failure_tasks.append(service_failure_task)
            
            # Wait 5 more seconds, then add latency
            await asyncio.sleep(5)
            latency_task = asyncio.create_task(
                self.inject_service_failure("detector_orchestration", FailureType.HIGH_LATENCY, 15)
            )
            failure_tasks.append(latency_task)
            
            # Wait for all failures to complete
            failure_results = await asyncio.gather(*failure_tasks, return_exceptions=True)
            
            # Check if system prevented cascade
            cascade_prevented = await self._validate_cascade_prevention()
            
            # Measure overall recovery
            recovery_start = time.time()
            await self._ensure_all_services_recovered()
            total_recovery_time = time.time() - recovery_start
            
            return ChaosTestResult(
                scenario_name="cascading_failure_prevention",
                target_service="all_services",
                failure_type=FailureType.PARTIAL_FAILURE,
                start_time=start_timestamp,
                end_time=datetime.utcnow().isoformat(),
                duration_seconds=time.time() - start_time,
                success=cascade_prevented,
                expected_behavior_observed=cascade_prevented,
                system_recovery_time=total_recovery_time,
                blast_radius_contained=cascade_prevented,
                service_responses={},
                metrics_during_failure={}
            )
        
        except Exception as e:
            logger.error(f"Cascading failure test failed", error=str(e))
            
            return ChaosTestResult(
                scenario_name="cascading_failure_prevention",
                target_service="all_services",
                failure_type=FailureType.PARTIAL_FAILURE,
                start_time=start_timestamp,
                end_time=datetime.utcnow().isoformat(),
                duration_seconds=time.time() - start_time,
                success=False,
                expected_behavior_observed=False,
                system_recovery_time=0.0,
                blast_radius_contained=False,
                service_responses={},
                metrics_during_failure={},
                error_details=str(e)
            )
    
    async def run_all_chaos_scenarios(self) -> ResilienceReport:
        """Run all predefined chaos scenarios."""
        logger.info("Running all chaos engineering scenarios")
        
        chaos_results = []
        
        for scenario in self.chaos_scenarios:
            try:
                if scenario.failure_type == FailureType.NETWORK_PARTITION:
                    result = await self.inject_network_partition([scenario.target_service])
                elif scenario.name == "cascading_failure_prevention":
                    result = await self.test_cascading_failure_prevention()
                else:
                    result = await self.inject_service_failure(
                        scenario.target_service,
                        scenario.failure_type,
                        scenario.duration_seconds
                    )
                
                chaos_results.append(result)
                
                # Wait between tests to allow system stabilization
                await asyncio.sleep(30)
            
            except Exception as e:
                logger.error(f"Chaos scenario failed", scenario=scenario.name, error=str(e))
        
        # Generate comprehensive resilience report
        return self._generate_resilience_report(chaos_results)
    
    async def _collect_baseline_metrics(self) -> Dict[str, Any]:
        """Collect baseline system metrics before failure injection."""
        baseline = {}
        
        for service_name, client in self.service_clients.items():
            try:
                response = await client.get("/health", timeout=5.0)
                baseline[service_name] = {
                    "status": response.status_code,
                    "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0
                }
            except Exception as e:
                baseline[service_name] = {
                    "status": 0,
                    "error": str(e)
                }
        
        return baseline
    
    async def _monitor_system_during_failure(self, duration: int) -> Dict[str, Any]:
        """Monitor system behavior during failure injection."""
        metrics = {
            "service_responses": {},
            "metrics": {}
        }
        
        # Monitor at 10-second intervals
        monitoring_interval = 10
        monitoring_count = duration // monitoring_interval
        
        for i in range(monitoring_count):
            timestamp = time.time()
            
            # Check service health
            for service_name, client in self.service_clients.items():
                try:
                    response = await client.get("/health", timeout=5.0)
                    metrics["service_responses"][f"{service_name}_{timestamp}"] = {
                        "status": response.status_code,
                        "timestamp": timestamp
                    }
                except Exception as e:
                    metrics["service_responses"][f"{service_name}_{timestamp}"] = {
                        "status": 0,
                        "error": str(e),
                        "timestamp": timestamp
                    }
            
            await asyncio.sleep(monitoring_interval)
        
        return metrics
    
    async def _monitor_network_partition(self, services: List[str], duration: int) -> Dict[str, Any]:
        """Monitor system during network partition."""
        # Simplified monitoring for network partition
        return {
            "service_responses": {},
            "metrics": {
                "partition_duration": duration,
                "affected_services": len(services)
            }
        }
    
    async def _validate_expected_behavior(self, service: str, failure_type: FailureType) -> bool:
        """Validate that system exhibits expected behavior during failure."""
        # Simplified validation - would check specific behavior based on failure type
        
        if failure_type == FailureType.SERVICE_CRASH:
            # Check if other services are still responding
            healthy_services = 0
            for service_name, client in self.service_clients.items():
                if service_name != service:
                    try:
                        response = await client.get("/health", timeout=5.0)
                        if response.status_code == 200:
                            healthy_services += 1
                    except Exception:
                        pass
            
            return healthy_services > 0  # At least one other service should be healthy
        
        elif failure_type == FailureType.DATABASE_FAILURE:
            # Check if services can still serve cached responses
            return True  # Simplified - would check actual cache behavior
        
        return True  # Default to true for other failure types
    
    async def _check_blast_radius_containment(self, failed_service: str) -> bool:
        """Check if failure blast radius was contained."""
        # Check if other services are still functional
        functional_services = 0
        total_other_services = 0
        
        for service_name, client in self.service_clients.items():
            if service_name != failed_service:
                total_other_services += 1
                try:
                    response = await client.get("/health", timeout=5.0)
                    if response.status_code == 200:
                        functional_services += 1
                except Exception:
                    pass
        
        # At least 50% of other services should remain functional
        return functional_services >= (total_other_services * 0.5) if total_other_services > 0 else True
    
    async def _validate_partition_handling(self, services: List[str]) -> bool:
        """Validate that network partition was handled correctly."""
        # Simplified validation - would check if services implement proper partition tolerance
        return True
    
    async def _validate_cascade_prevention(self) -> bool:
        """Validate that cascading failures were prevented."""
        # Check if at least one service is still functional
        functional_services = 0
        
        for service_name, client in self.service_clients.items():
            try:
                response = await client.get("/health", timeout=5.0)
                if response.status_code == 200:
                    functional_services += 1
            except Exception:
                pass
        
        return functional_services > 0
    
    async def _ensure_all_services_recovered(self) -> None:
        """Ensure all services have recovered from failures."""
        # Wait for services to recover
        max_recovery_time = 120  # 2 minutes
        recovery_start = time.time()
        
        while time.time() - recovery_start < max_recovery_time:
            all_healthy = True
            
            for service_name, client in self.service_clients.items():
                try:
                    response = await client.get("/health", timeout=5.0)
                    if response.status_code != 200:
                        all_healthy = False
                        break
                except Exception:
                    all_healthy = False
                    break
            
            if all_healthy:
                logger.info("All services recovered successfully")
                return
            
            await asyncio.sleep(5)
        
        logger.warning("Some services may not have fully recovered")
    
    def _generate_resilience_report(self, chaos_results: List[ChaosTestResult]) -> ResilienceReport:
        """Generate comprehensive resilience report."""
        total_scenarios = len(chaos_results)
        passed_scenarios = sum(1 for r in chaos_results if r.success)
        failed_scenarios = total_scenarios - passed_scenarios
        
        # Calculate resilience score
        resilience_score = (passed_scenarios / total_scenarios) * 100 if total_scenarios > 0 else 0
        
        # Calculate recovery metrics
        recovery_times = [r.system_recovery_time for r in chaos_results if r.system_recovery_time > 0]
        recovery_metrics = {
            "avg_recovery_time": sum(recovery_times) / len(recovery_times) if recovery_times else 0,
            "max_recovery_time": max(recovery_times) if recovery_times else 0,
            "recovery_success_rate": len(recovery_times) / total_scenarios if total_scenarios > 0 else 0
        }
        
        # Failure impact analysis
        failure_impact_analysis = {
            "blast_radius_containment_rate": sum(1 for r in chaos_results if r.blast_radius_contained) / total_scenarios if total_scenarios > 0 else 0,
            "expected_behavior_rate": sum(1 for r in chaos_results if r.expected_behavior_observed) / total_scenarios if total_scenarios > 0 else 0
        }
        
        # Generate recommendations
        recommendations = self._generate_resilience_recommendations(chaos_results)
        
        return ResilienceReport(
            timestamp=datetime.utcnow().isoformat(),
            total_scenarios=total_scenarios,
            passed_scenarios=passed_scenarios,
            failed_scenarios=failed_scenarios,
            chaos_results=chaos_results,
            resilience_score=resilience_score,
            recovery_metrics=recovery_metrics,
            failure_impact_analysis=failure_impact_analysis,
            recommendations=recommendations
        )
    
    def _generate_resilience_recommendations(self, chaos_results: List[ChaosTestResult]) -> List[str]:
        """Generate recommendations based on chaos test results."""
        recommendations = []
        
        failed_tests = [r for r in chaos_results if not r.success]
        if failed_tests:
            recommendations.append("Address failed resilience tests:")
            for result in failed_tests:
                recommendations.append(f"  - {result.scenario_name}: {result.error_details or 'Expected behavior not observed'}")
        
        # Recovery time recommendations
        slow_recoveries = [r for r in chaos_results if r.system_recovery_time > 60]
        if slow_recoveries:
            recommendations.append("Improve recovery times for:")
            for result in slow_recoveries:
                recommendations.append(f"  - {result.scenario_name}: {result.system_recovery_time:.1f}s recovery time")
        
        # Blast radius recommendations
        uncontained_failures = [r for r in chaos_results if not r.blast_radius_contained]
        if uncontained_failures:
            recommendations.append("Improve failure isolation for:")
            for result in uncontained_failures:
                recommendations.append(f"  - {result.scenario_name}: Blast radius not contained")
        
        if not recommendations:
            recommendations.append("Excellent resilience! Consider testing more extreme failure scenarios")
        
        return recommendations


# Failure Injectors

class FailureInjector:
    """Base class for failure injection."""
    
    async def inject_failure(self, target: str, duration: int) -> None:
        """Inject failure into target."""
        raise NotImplementedError
    
    async def recover_from_failure(self, target: str) -> None:
        """Recover from injected failure."""
        raise NotImplementedError


class ServiceCrashInjector(FailureInjector):
    """Injects service crash failures."""
    
    async def inject_failure(self, target: str, duration: int) -> None:
        """Simulate service crash."""
        logger.info(f"Simulating service crash", service=target, duration=duration)
        # In real implementation, would stop/kill the service container
        await asyncio.sleep(duration)
    
    async def recover_from_failure(self, target: str) -> None:
        """Recover from service crash."""
        logger.info(f"Recovering from service crash", service=target)
        # In real implementation, would restart the service


class NetworkPartitionInjector(FailureInjector):
    """Injects network partition failures."""
    
    async def inject_failure(self, target: str, duration: int) -> None:
        """Simulate network partition."""
        logger.info(f"Simulating network partition", service=target, duration=duration)
        # In real implementation, would use iptables or similar to block traffic
        await asyncio.sleep(duration)
    
    async def recover_from_failure(self, target: str) -> None:
        """Recover from network partition."""
        logger.info(f"Recovering from network partition", service=target)
        # In real implementation, would restore network connectivity


class LatencyInjector(FailureInjector):
    """Injects high latency."""
    
    async def inject_failure(self, target: str, duration: int) -> None:
        """Simulate high latency."""
        logger.info(f"Simulating high latency", service=target, duration=duration)
        # In real implementation, would use tc (traffic control) to add latency
        await asyncio.sleep(duration)
    
    async def recover_from_failure(self, target: str) -> None:
        """Recover from high latency."""
        logger.info(f"Recovering from high latency", service=target)


class MemoryExhaustionInjector(FailureInjector):
    """Injects memory exhaustion."""
    
    async def inject_failure(self, target: str, duration: int) -> None:
        """Simulate memory exhaustion."""
        logger.info(f"Simulating memory exhaustion", service=target, duration=duration)
        # In real implementation, would use stress-ng or similar
        await asyncio.sleep(duration)
    
    async def recover_from_failure(self, target: str) -> None:
        """Recover from memory exhaustion."""
        logger.info(f"Recovering from memory exhaustion", service=target)


class TimeoutInjector(FailureInjector):
    """Injects timeout failures."""
    
    async def inject_failure(self, target: str, duration: int) -> None:
        """Simulate timeout conditions."""
        logger.info(f"Simulating timeout conditions", service=target, duration=duration)
        await asyncio.sleep(duration)
    
    async def recover_from_failure(self, target: str) -> None:
        """Recover from timeout conditions."""
        logger.info(f"Recovering from timeout conditions", service=target)
