"""
Performance tests for quality alerting system.

This module contains performance and load tests to ensure the quality
alerting system can handle production workloads efficiently.
"""

import asyncio
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from src.llama_mapper.analysis.quality import (
    AlertSeverity,
    QualityAlertingSystem,
    QualityMetric,
    QualityMetricType,
    QualityThreshold,
)


class TestQualityAlertingPerformance:
    """Performance tests for quality alerting system."""

    @pytest.fixture
    def performance_system(self):
        """Create a system optimized for performance testing."""
        return QualityAlertingSystem(
            monitoring_interval_seconds=5,  # Less frequent monitoring
            max_metrics_per_type=50000,  # Large capacity
            alert_retention_days=1,
        )

    def test_metric_recording_performance(self, performance_system):
        """Test metric recording performance under load."""
        num_metrics = 1000

        start_time = time.time()

        # Record metrics sequentially
        for i in range(num_metrics):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + (i % 10) * 0.01,
                timestamp=datetime.now(),
                labels={"service": "test", "iteration": str(i)},
            )
            performance_system.process_metric(metric)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert (
            total_time < 2.0
        ), f"Recording {num_metrics} metrics took {total_time:.2f}s"

        metrics_per_second = num_metrics / total_time
        assert (
            metrics_per_second > 500
        ), f"Only achieved {metrics_per_second:.0f} metrics/sec"

        # Verify all metrics were recorded
        current_metrics = performance_system.quality_monitor.get_current_metrics()
        assert QualityMetricType.SCHEMA_VALIDATION_RATE in current_metrics

        # Check performance statistics
        perf_stats = performance_system.quality_monitor.get_performance_statistics()
        assert perf_stats["metrics_recorded"] >= num_metrics

    def test_concurrent_metric_recording(self, performance_system):
        """Test concurrent metric recording performance."""
        num_threads = 10
        metrics_per_thread = 100
        total_metrics = num_threads * metrics_per_thread

        def record_metrics(thread_id: int) -> List[QualityMetric]:
            """Record metrics for a single thread."""
            metrics = []
            for i in range(metrics_per_thread):
                metric = QualityMetric(
                    metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                    value=0.9 + (i % 10) * 0.01,
                    timestamp=datetime.now(),
                    labels={
                        "service": "test",
                        "thread": str(thread_id),
                        "iteration": str(i),
                    },
                )
                performance_system.process_metric(metric)
                metrics.append(metric)
            return metrics

        start_time = time.time()

        # Record metrics concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_metrics, i) for i in range(num_threads)]

            # Wait for all threads to complete
            for future in as_completed(futures):
                metrics = future.result()
                assert len(metrics) == metrics_per_thread

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 3.0, f"Concurrent recording took {total_time:.2f}s"

        metrics_per_second = total_metrics / total_time
        assert (
            metrics_per_second > 300
        ), f"Only achieved {metrics_per_second:.0f} metrics/sec"

        # Verify all metrics were recorded
        perf_stats = performance_system.quality_monitor.get_performance_statistics()
        assert perf_stats["metrics_recorded"] >= total_metrics

    def test_metric_retrieval_performance(self, performance_system):
        """Test metric retrieval performance."""
        # First, populate with many metrics
        num_metrics = 5000
        base_time = datetime.now()

        for i in range(num_metrics):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + (i % 10) * 0.01,
                timestamp=base_time + timedelta(seconds=i),
                labels={"service": "test", "iteration": str(i)},
            )
            performance_system.process_metric(metric)

        # Test retrieval performance
        start_time = time.time()

        # Retrieve metrics for different time ranges
        for i in range(100):
            start_range = base_time + timedelta(seconds=i * 10)
            end_range = start_range + timedelta(minutes=1)

            metrics = performance_system.quality_monitor.get_metrics(
                QualityMetricType.SCHEMA_VALIDATION_RATE, start_range, end_range
            )

            # Verify we got some metrics
            assert len(metrics) > 0

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 1.0, f"Retrieval took {total_time:.2f}s"

        retrievals_per_second = 100 / total_time
        assert (
            retrievals_per_second > 100
        ), f"Only achieved {retrievals_per_second:.0f} retrievals/sec"

    def test_statistics_calculation_performance(self, performance_system):
        """Test statistics calculation performance."""
        # Populate with metrics
        num_metrics = 2000
        base_time = datetime.now()

        for i in range(num_metrics):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + (i % 10) * 0.01,
                timestamp=base_time + timedelta(seconds=i),
                labels={"service": "test"},
            )
            performance_system.process_metric(metric)

        # Test statistics calculation performance
        start_time = time.time()

        # Calculate statistics multiple times
        for i in range(50):
            stats = performance_system.quality_monitor.get_metric_statistics(
                QualityMetricType.SCHEMA_VALIDATION_RATE, time_window_minutes=60
            )

            # Verify stats are calculated
            assert "count" in stats
            assert "mean" in stats
            assert "std" in stats

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 0.5, f"Statistics calculation took {total_time:.2f}s"

        calculations_per_second = 50 / total_time
        assert (
            calculations_per_second > 100
        ), f"Only achieved {calculations_per_second:.0f} calculations/sec"

    def test_trend_analysis_performance(self, performance_system):
        """Test trend analysis performance."""
        # Populate with metrics showing a trend
        num_metrics = 1000
        base_time = datetime.now()

        for i in range(num_metrics):
            # Create a declining trend
            value = 0.95 - (i * 0.0001)
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=value,
                timestamp=base_time + timedelta(minutes=i),
                labels={"service": "test"},
            )
            performance_system.process_metric(metric)

        # Test trend analysis performance
        start_time = time.time()

        # Calculate trends multiple times
        for i in range(20):
            trends = performance_system.quality_monitor.get_metric_trends(
                QualityMetricType.SCHEMA_VALIDATION_RATE, time_window_minutes=60
            )

            # Verify trends are calculated
            assert "trend" in trends
            assert "slope" in trends
            assert "r_squared" in trends

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 0.3, f"Trend analysis took {total_time:.2f}s"

        analyses_per_second = 20 / total_time
        assert (
            analyses_per_second > 50
        ), f"Only achieved {analyses_per_second:.0f} analyses/sec"

    def test_degradation_detection_performance(self, performance_system):
        """Test degradation detection performance."""
        # Populate with metrics
        num_metrics = 1000
        base_time = datetime.now()

        for i in range(num_metrics):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + (i % 10) * 0.01,
                timestamp=base_time + timedelta(seconds=i),
                labels={"service": "test"},
            )
            performance_system.process_metric(metric)

        # Get metrics for detection
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=10)
        metrics = performance_system.quality_monitor.get_metrics(
            QualityMetricType.SCHEMA_VALIDATION_RATE, start_time, end_time
        )

        # Create threshold
        threshold = QualityThreshold(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            warning_threshold=0.95,
            critical_threshold=0.90,
            min_samples=10,
            time_window_minutes=10,
        )

        # Test detection performance
        start_time = time.time()

        # Run detection multiple times
        for i in range(50):
            degradation = performance_system.degradation_detector.detect_degradation(
                QualityMetricType.SCHEMA_VALIDATION_RATE, metrics, threshold
            )
            # Degradation may or may not be detected depending on data

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 0.2, f"Degradation detection took {total_time:.2f}s"

        detections_per_second = 50 / total_time
        assert (
            detections_per_second > 200
        ), f"Only achieved {detections_per_second:.0f} detections/sec"

    def test_alert_processing_performance(self, performance_system):
        """Test alert processing performance."""
        # Create many alerts
        num_alerts = 1000

        start_time = time.time()

        for i in range(num_alerts):
            alert = performance_system.alert_manager.create_alert(
                title=f"Test Alert {i}",
                description=f"Test alert description {i}",
                severity=AlertSeverity.MEDIUM,
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            )

            # Send alert (to logging handler)
            performance_system.alert_manager.send_alert(alert)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 2.0, f"Alert processing took {total_time:.2f}s"

        alerts_per_second = num_alerts / total_time
        assert (
            alerts_per_second > 500
        ), f"Only achieved {alerts_per_second:.0f} alerts/sec"

        # Verify alerts were created
        stats = performance_system.alert_manager.get_alert_statistics()
        assert stats["total_created"] >= num_alerts

    def test_memory_usage_under_load(self, performance_system):
        """Test memory usage under load."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Add many metrics
        num_metrics = 10000
        for i in range(num_metrics):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + (i % 10) * 0.01,
                timestamp=datetime.now(),
                labels={"service": "test", "iteration": str(i)},
            )
            performance_system.process_metric(metric)

        # Get memory usage after adding metrics
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage should be reasonable
        # Allow for some overhead, but shouldn't be excessive
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"

        # Check storage statistics
        storage_stats = performance_system.quality_monitor.get_storage_statistics()
        assert storage_stats["total_metrics"] >= num_metrics

    def test_cleanup_performance(self, performance_system):
        """Test cleanup performance."""
        # Add many old metrics
        old_time = datetime.now() - timedelta(hours=2)
        num_old_metrics = 5000

        for i in range(num_old_metrics):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + (i % 10) * 0.01,
                timestamp=old_time + timedelta(seconds=i),
                labels={"service": "test"},
            )
            performance_system.process_metric(metric)

        # Add some recent metrics
        recent_time = datetime.now()
        num_recent_metrics = 1000

        for i in range(num_recent_metrics):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + (i % 10) * 0.01,
                timestamp=recent_time + timedelta(seconds=i),
                labels={"service": "test"},
            )
            performance_system.process_metric(metric)

        # Test cleanup performance
        start_time = time.time()

        # Force cleanup
        performance_system.quality_monitor._cleanup_old_metrics()

        end_time = time.time()
        cleanup_time = end_time - start_time

        # Cleanup should be fast
        assert cleanup_time < 0.5, f"Cleanup took {cleanup_time:.2f}s"

        # Check that old metrics were cleaned up
        storage_stats = performance_system.quality_monitor.get_storage_statistics()
        # Should have fewer metrics after cleanup
        assert storage_stats["total_metrics"] < num_old_metrics + num_recent_metrics

    def test_system_responsiveness_under_load(self, performance_system):
        """Test system responsiveness under continuous load."""
        # Start monitoring
        performance_system.start_monitoring()

        try:
            # Create a thread that continuously adds metrics
            stop_flag = threading.Event()
            metrics_added = 0

            def add_metrics_continuously():
                nonlocal metrics_added
                while not stop_flag.is_set():
                    metric = QualityMetric(
                        metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                        value=0.9 + (metrics_added % 10) * 0.01,
                        timestamp=datetime.now(),
                        labels={"service": "test", "continuous": "true"},
                    )
                    performance_system.process_metric(metric)
                    metrics_added += 1
                    time.sleep(0.001)  # 1ms delay

            # Start the continuous metric addition
            metric_thread = threading.Thread(target=add_metrics_continuously)
            metric_thread.start()

            # Let it run for a bit
            time.sleep(2)

            # Test system responsiveness during load
            start_time = time.time()

            # Try to get system status (should be responsive)
            status = performance_system.get_system_status()
            dashboard_data = performance_system.get_quality_dashboard_data()

            end_time = time.time()
            response_time = end_time - start_time

            # System should remain responsive
            assert response_time < 0.1, f"System response time was {response_time:.3f}s"

            # Stop the continuous thread
            stop_flag.set()
            metric_thread.join(timeout=1)

            # Verify metrics were added
            assert metrics_added > 100, f"Only added {metrics_added} metrics"

        finally:
            performance_system.stop_monitoring()


class TestQualityAlertingScalability:
    """Scalability tests for quality alerting system."""

    def test_large_metric_volume(self):
        """Test system with very large metric volumes."""
        system = QualityAlertingSystem(
            max_metrics_per_type=100000, alert_retention_days=1  # Very large capacity
        )

        # Add a large number of metrics
        num_metrics = 50000
        start_time = time.time()

        for i in range(num_metrics):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + (i % 100) * 0.001,
                timestamp=datetime.now() + timedelta(seconds=i),
                labels={"service": "test", "batch": str(i // 1000)},
            )
            system.process_metric(metric)

        end_time = time.time()
        total_time = end_time - start_time

        # Should handle large volumes efficiently
        assert (
            total_time < 10.0
        ), f"Processing {num_metrics} metrics took {total_time:.2f}s"

        # Verify metrics were stored
        storage_stats = system.quality_monitor.get_storage_statistics()
        assert storage_stats["total_metrics"] >= num_metrics

    def test_multiple_metric_types(self):
        """Test system with multiple metric types."""
        system = QualityAlertingSystem(max_metrics_per_type=10000)

        metric_types = [
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            QualityMetricType.TEMPLATE_FALLBACK_RATE,
            QualityMetricType.OPA_COMPILATION_SUCCESS_RATE,
            QualityMetricType.CONFIDENCE_SCORE,
            QualityMetricType.RESPONSE_TIME,
            QualityMetricType.ERROR_RATE,
            QualityMetricType.THROUGHPUT,
        ]

        metrics_per_type = 1000

        start_time = time.time()

        for metric_type in metric_types:
            for i in range(metrics_per_type):
                # Use appropriate value ranges for different metric types
                if metric_type in [
                    QualityMetricType.SCHEMA_VALIDATION_RATE,
                    QualityMetricType.TEMPLATE_FALLBACK_RATE,
                    QualityMetricType.OPA_COMPILATION_SUCCESS_RATE,
                    QualityMetricType.ERROR_RATE,
                ]:
                    value = 0.9 + (i % 10) * 0.01
                elif metric_type == QualityMetricType.CONFIDENCE_SCORE:
                    value = 0.7 + (i % 30) * 0.01
                elif metric_type == QualityMetricType.RESPONSE_TIME:
                    value = 0.5 + (i % 50) * 0.1
                else:  # THROUGHPUT
                    value = 10.0 + (i % 100) * 0.1

                metric = QualityMetric(
                    metric_type=metric_type,
                    value=value,
                    timestamp=datetime.now() + timedelta(seconds=i),
                    labels={"service": "test", "type": metric_type.value},
                )
                system.process_metric(metric)

        end_time = time.time()
        total_time = end_time - start_time

        # Should handle multiple types efficiently
        assert total_time < 5.0, f"Processing multiple types took {total_time:.2f}s"

        # Verify all types were stored
        current_metrics = system.quality_monitor.get_current_metrics()
        for metric_type in metric_types:
            assert metric_type in current_metrics


if __name__ == "__main__":
    pytest.main([__file__])
