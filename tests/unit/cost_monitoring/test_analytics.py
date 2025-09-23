"""Unit tests for cost analytics system."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock

from src.llama_mapper.cost_monitoring.analytics.cost_analytics import (
    CostAnalytics,
    CostAnalyticsConfig,
    CostTrend,
    CostOptimizationRecommendation,
    CostAnomaly,
    CostForecast,
)


class TestCostAnalyticsConfig:
    """Test cost analytics configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CostAnalyticsConfig()
        
        assert config.enabled is True
        assert config.analysis_interval_hours == 24
        assert config.anomaly_detection_enabled is True
        assert config.forecasting_enabled is True
        assert config.recommendation_engine_enabled is True
        assert config.anomaly_threshold == 2.0
        assert config.forecast_horizon_days == 30
        assert config.min_data_points == 7
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CostAnalyticsConfig(
            enabled=False,
            analysis_interval_hours=12,
            anomaly_detection_enabled=False,
            forecasting_enabled=False,
            recommendation_engine_enabled=False,
            anomaly_threshold=1.5,
            forecast_horizon_days=14,
            min_data_points=5,
        )
        
        assert config.enabled is False
        assert config.analysis_interval_hours == 12
        assert config.anomaly_detection_enabled is False
        assert config.forecasting_enabled is False
        assert config.recommendation_engine_enabled is False
        assert config.anomaly_threshold == 1.5
        assert config.forecast_horizon_days == 14
        assert config.min_data_points == 5


class TestCostTrend:
    """Test cost trend data structure."""
    
    def test_cost_trend_creation(self):
        """Test cost trend creation."""
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        end_time = datetime.now(timezone.utc)
        
        trend = CostTrend(
            period_start=start_time,
            period_end=end_time,
            total_cost=1000.0,
            average_daily_cost=142.86,
            cost_growth_rate=15.5,
            peak_cost=200.0,
            lowest_cost=100.0,
            cost_volatility=25.0,
            currency="USD",
        )
        
        assert trend.period_start == start_time
        assert trend.period_end == end_time
        assert trend.total_cost == 1000.0
        assert trend.average_daily_cost == 142.86
        assert trend.cost_growth_rate == 15.5
        assert trend.peak_cost == 200.0
        assert trend.lowest_cost == 100.0
        assert trend.cost_volatility == 25.0
        assert trend.currency == "USD"


class TestCostOptimizationRecommendation:
    """Test cost optimization recommendation data structure."""
    
    def test_recommendation_creation(self):
        """Test recommendation creation."""
        timestamp = datetime.now(timezone.utc)
        recommendation = CostOptimizationRecommendation(
            recommendation_id="test_recommendation",
            category="compute",
            title="Optimize Compute Resources",
            description="Consider right-sizing instances",
            potential_savings=100.0,
            confidence=0.8,
            effort_level="medium",
            impact_level="high",
            priority=8,
            currency="USD",
            created_at=timestamp,
            tenant_id="test-tenant",
        )
        
        assert recommendation.recommendation_id == "test_recommendation"
        assert recommendation.category == "compute"
        assert recommendation.title == "Optimize Compute Resources"
        assert recommendation.description == "Consider right-sizing instances"
        assert recommendation.potential_savings == 100.0
        assert recommendation.confidence == 0.8
        assert recommendation.effort_level == "medium"
        assert recommendation.impact_level == "high"
        assert recommendation.priority == 8
        assert recommendation.currency == "USD"
        assert recommendation.created_at == timestamp
        assert recommendation.tenant_id == "test-tenant"


class TestCostAnomaly:
    """Test cost anomaly data structure."""
    
    def test_anomaly_creation(self):
        """Test anomaly creation."""
        timestamp = datetime.now(timezone.utc)
        anomaly = CostAnomaly(
            anomaly_id="test_anomaly",
            anomaly_type="spike",
            detected_at=timestamp,
            cost_value=200.0,
            expected_value=100.0,
            deviation_percent=100.0,
            severity="high",
            description="Cost spike detected",
            currency="USD",
            tenant_id="test-tenant",
        )
        
        assert anomaly.anomaly_id == "test_anomaly"
        assert anomaly.anomaly_type == "spike"
        assert anomaly.detected_at == timestamp
        assert anomaly.cost_value == 200.0
        assert anomaly.expected_value == 100.0
        assert anomaly.deviation_percent == 100.0
        assert anomaly.severity == "high"
        assert anomaly.description == "Cost spike detected"
        assert anomaly.currency == "USD"
        assert anomaly.tenant_id == "test-tenant"


class TestCostForecast:
    """Test cost forecast data structure."""
    
    def test_forecast_creation(self):
        """Test forecast creation."""
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(days=30)
        created_at = datetime.now(timezone.utc)
        
        forecast = CostForecast(
            forecast_id="test_forecast",
            forecast_period_start=start_time,
            forecast_period_end=end_time,
            predicted_cost=1500.0,
            confidence_interval_lower=1200.0,
            confidence_interval_upper=1800.0,
            confidence_level=0.95,
            model_type="linear_regression",
            currency="USD",
            created_at=created_at,
            tenant_id="test-tenant",
        )
        
        assert forecast.forecast_id == "test_forecast"
        assert forecast.forecast_period_start == start_time
        assert forecast.forecast_period_end == end_time
        assert forecast.predicted_cost == 1500.0
        assert forecast.confidence_interval_lower == 1200.0
        assert forecast.confidence_interval_upper == 1800.0
        assert forecast.confidence_level == 0.95
        assert forecast.model_type == "linear_regression"
        assert forecast.currency == "USD"
        assert forecast.created_at == created_at
        assert forecast.tenant_id == "test-tenant"


class TestCostAnalytics:
    """Test cost analytics system."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CostAnalyticsConfig(
            analysis_interval_hours=1,  # Fast for testing
            min_data_points=3,  # Lower for testing
        )
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        collector._metrics_history = []
        return collector
    
    @pytest.fixture
    def analytics(self, config, mock_metrics_collector):
        """Create test analytics system."""
        return CostAnalytics(config, mock_metrics_collector)
    
    def test_analytics_initialization(self, analytics):
        """Test analytics initialization."""
        assert analytics.config is not None
        assert analytics.metrics_collector is not None
        assert analytics._recommendations == []
        assert analytics._anomalies == []
        assert analytics._forecasts == []
        assert analytics._running is False
        assert analytics._analysis_task is None
    
    def test_determine_anomaly_severity(self, analytics):
        """Test anomaly severity determination."""
        # Test different z-scores
        assert analytics._determine_anomaly_severity(1.0) == "low"
        assert analytics._determine_anomaly_severity(2.6) == "medium"
        assert analytics._determine_anomaly_severity(3.5) == "high"
        assert analytics._determine_anomaly_severity(4.5) == "critical"
    
    @pytest.mark.asyncio
    async def test_detect_anomalies(self, analytics):
        """Test anomaly detection."""
        # Mock cost trends with some anomalies
        analytics.metrics_collector.get_cost_trends = Mock(return_value={
            "costs": [100.0, 105.0, 110.0, 200.0, 115.0, 120.0, 125.0],  # 200.0 is an anomaly
        })
        
        await analytics._detect_anomalies()
        
        # Should detect one anomaly
        assert len(analytics._anomalies) == 1
        anomaly = analytics._anomalies[0]
        assert anomaly.anomaly_type == "spike"
        assert anomaly.cost_value == 200.0
        assert anomaly.severity in ["medium", "high", "critical"]
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_insufficient_data(self, analytics):
        """Test anomaly detection with insufficient data."""
        # Mock cost trends with insufficient data
        analytics.metrics_collector.get_cost_trends = Mock(return_value={
            "costs": [100.0, 105.0],  # Less than min_data_points
        })
        
        await analytics._detect_anomalies()
        
        # Should not detect any anomalies
        assert len(analytics._anomalies) == 0
    
    @pytest.mark.asyncio
    async def test_generate_forecasts(self, analytics):
        """Test forecast generation."""
        # Mock cost trends
        analytics.metrics_collector.get_cost_trends = Mock(return_value={
            "costs": [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0],  # Upward trend
        })
        
        await analytics._generate_forecasts()
        
        # Should generate one forecast
        assert len(analytics._forecasts) == 1
        forecast = analytics._forecasts[0]
        assert forecast.predicted_cost > 0
        assert forecast.confidence_level > 0
        assert forecast.model_type == "linear_regression"
    
    @pytest.mark.asyncio
    async def test_generate_forecasts_insufficient_data(self, analytics):
        """Test forecast generation with insufficient data."""
        # Mock cost trends with insufficient data
        analytics.metrics_collector.get_cost_trends = Mock(return_value={
            "costs": [100.0, 105.0],  # Less than min_data_points
        })
        
        await analytics._generate_forecasts()
        
        # Should not generate any forecasts
        assert len(analytics._forecasts) == 0
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, analytics):
        """Test recommendation generation."""
        # Mock cost breakdown
        from src.llama_mapper.cost_monitoring.core.metrics_collector import CostBreakdown
        
        breakdown = CostBreakdown(
            compute_cost=600.0,  # > 60% of total
            memory_cost=200.0,
            storage_cost=100.0,
            network_cost=50.0,
            api_cost=50.0,
            total_cost=1000.0,
            currency="USD",
            period_start=datetime.now(timezone.utc) - timedelta(days=7),
            period_end=datetime.now(timezone.utc),
        )
        
        analytics.metrics_collector.get_cost_breakdown = Mock(return_value=breakdown)
        
        await analytics._generate_recommendations()
        
        # Should generate recommendations
        assert len(analytics._recommendations) > 0
        
        # Check for compute optimization recommendation
        compute_recs = [r for r in analytics._recommendations if r.category == "compute"]
        assert len(compute_recs) > 0
        assert compute_recs[0].potential_savings > 0
    
    def test_get_cost_trend_analysis(self, analytics):
        """Test cost trend analysis."""
        # Mock cost trends
        analytics.metrics_collector.get_cost_trends = Mock(return_value={
            "costs": [100.0, 105.0, 110.0, 115.0, 120.0],
            "total_cost": 550.0,
            "average_daily_cost": 110.0,
        })
        
        trend = analytics.get_cost_trend_analysis(days=5)
        
        assert isinstance(trend, CostTrend)
        assert trend.total_cost == 550.0
        assert trend.average_daily_cost == 110.0
        assert trend.peak_cost == 120.0
        assert trend.lowest_cost == 100.0
        assert trend.cost_volatility >= 0
    
    def test_get_cost_trend_analysis_no_data(self, analytics):
        """Test cost trend analysis with no data."""
        # Mock empty cost trends
        analytics.metrics_collector.get_cost_trends = Mock(return_value={
            "costs": [],
            "total_cost": 0.0,
            "average_daily_cost": 0.0,
        })
        
        trend = analytics.get_cost_trend_analysis(days=5)
        
        assert isinstance(trend, CostTrend)
        assert trend.total_cost == 0.0
        assert trend.average_daily_cost == 0.0
        assert trend.cost_growth_rate == 0.0
        assert trend.peak_cost == 0.0
        assert trend.lowest_cost == 0.0
        assert trend.cost_volatility == 0.0
    
    def test_get_optimization_recommendations(self, analytics):
        """Test getting optimization recommendations."""
        # Add test recommendations
        recommendations = [
            CostOptimizationRecommendation(
                recommendation_id="rec_1",
                category="compute",
                title="Compute Optimization",
                description="Optimize compute resources",
                potential_savings=100.0,
                confidence=0.8,
                effort_level="medium",
                impact_level="high",
                priority=8,
                created_at=datetime.now(timezone.utc),
            ),
            CostOptimizationRecommendation(
                recommendation_id="rec_2",
                category="storage",
                title="Storage Optimization",
                description="Optimize storage usage",
                potential_savings=50.0,
                confidence=0.6,
                effort_level="low",
                impact_level="medium",
                priority=5,
                created_at=datetime.now(timezone.utc),
            ),
        ]
        
        analytics._recommendations = recommendations
        
        # Get all recommendations
        all_recs = analytics.get_optimization_recommendations()
        assert len(all_recs) == 2
        
        # Filter by category
        compute_recs = analytics.get_optimization_recommendations(category="compute")
        assert len(compute_recs) == 1
        assert compute_recs[0].category == "compute"
        
        # Filter by priority
        high_priority_recs = analytics.get_optimization_recommendations(priority_min=7)
        assert len(high_priority_recs) == 1
        assert high_priority_recs[0].priority >= 7
        
        # Check sorting by priority
        assert all_recs[0].priority >= all_recs[1].priority
    
    def test_get_cost_anomalies(self, analytics):
        """Test getting cost anomalies."""
        now = datetime.now(timezone.utc)
        
        # Add test anomalies
        anomalies = [
            CostAnomaly(
                anomaly_id="anomaly_1",
                anomaly_type="spike",
                detected_at=now - timedelta(hours=1),
                cost_value=200.0,
                expected_value=100.0,
                deviation_percent=100.0,
                severity="high",
                description="Cost spike detected",
            ),
            CostAnomaly(
                anomaly_id="anomaly_2",
                anomaly_type="drop",
                detected_at=now - timedelta(hours=2),
                cost_value=50.0,
                expected_value=100.0,
                deviation_percent=-50.0,
                severity="medium",
                description="Cost drop detected",
            ),
        ]
        
        analytics._anomalies = anomalies
        
        # Get all anomalies
        all_anomalies = analytics.get_cost_anomalies(days=1)
        assert len(all_anomalies) == 2
        
        # Filter by severity
        high_anomalies = analytics.get_cost_anomalies(severity="high", days=1)
        assert len(high_anomalies) == 1
        assert high_anomalies[0].severity == "high"
        
        # Check sorting by detection time
        assert all_anomalies[0].detected_at >= all_anomalies[1].detected_at
    
    def test_get_latest_forecast(self, analytics):
        """Test getting latest forecast."""
        now = datetime.now(timezone.utc)
        
        # Add test forecasts
        forecast1 = CostForecast(
            forecast_id="forecast_1",
            forecast_period_start=now,
            forecast_period_end=now + timedelta(days=30),
            predicted_cost=1000.0,
            confidence_interval_lower=800.0,
            confidence_interval_upper=1200.0,
            confidence_level=0.95,
            model_type="linear_regression",
            created_at=now - timedelta(hours=2),
        )
        
        forecast2 = CostForecast(
            forecast_id="forecast_2",
            forecast_period_start=now,
            forecast_period_end=now + timedelta(days=30),
            predicted_cost=1100.0,
            confidence_interval_lower=900.0,
            confidence_interval_upper=1300.0,
            confidence_level=0.95,
            model_type="linear_regression",
            created_at=now - timedelta(hours=1),  # More recent
        )
        
        analytics._forecasts = [forecast1, forecast2]
        
        # Get latest forecast
        latest = analytics.get_latest_forecast()
        assert latest is not None
        assert latest.forecast_id == "forecast_2"  # More recent
    
    def test_get_latest_forecast_none(self, analytics):
        """Test getting latest forecast when none exist."""
        latest = analytics.get_latest_forecast()
        assert latest is None
    
    def test_get_analytics_summary(self, analytics):
        """Test getting analytics summary."""
        # Add test data
        analytics._recommendations = [
            CostOptimizationRecommendation(
                recommendation_id="rec_1",
                category="compute",
                title="Compute Optimization",
                description="Optimize compute resources",
                potential_savings=100.0,
                confidence=0.8,
                effort_level="medium",
                impact_level="high",
                priority=8,
                created_at=datetime.now(timezone.utc),
            ),
        ]
        
        analytics._anomalies = [
            CostAnomaly(
                anomaly_id="anomaly_1",
                anomaly_type="spike",
                detected_at=datetime.now(timezone.utc),
                cost_value=200.0,
                expected_value=100.0,
                deviation_percent=100.0,
                severity="critical",
                description="Cost spike detected",
            ),
        ]
        
        analytics._forecasts = [
            CostForecast(
                forecast_id="forecast_1",
                forecast_period_start=datetime.now(timezone.utc),
                forecast_period_end=datetime.now(timezone.utc) + timedelta(days=30),
                predicted_cost=1000.0,
                confidence_interval_lower=800.0,
                confidence_interval_upper=1200.0,
                confidence_level=0.95,
                model_type="linear_regression",
                created_at=datetime.now(timezone.utc),
            ),
        ]
        
        # Mock cost trend analysis
        analytics.get_cost_trend_analysis = Mock(return_value=CostTrend(
            period_start=datetime.now(timezone.utc) - timedelta(days=30),
            period_end=datetime.now(timezone.utc),
            total_cost=1000.0,
            average_daily_cost=33.33,
            cost_growth_rate=10.0,
            peak_cost=50.0,
            lowest_cost=20.0,
            cost_volatility=10.0,
        ))
        
        summary = analytics.get_analytics_summary(days=30)
        
        assert "cost_trend" in summary
        assert "recommendations" in summary
        assert "anomalies" in summary
        assert "forecast" in summary
        
        assert summary["recommendations"]["total"] == 1
        assert summary["recommendations"]["total_potential_savings"] == 100.0
        assert summary["recommendations"]["by_category"]["compute"] == 1
        
        assert summary["anomalies"]["total"] == 1
        assert summary["anomalies"]["critical"] == 1
        assert summary["anomalies"]["by_type"]["spike"] == 1
        
        assert summary["forecast"] is not None
