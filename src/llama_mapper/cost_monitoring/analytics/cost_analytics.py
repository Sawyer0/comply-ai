"""Cost analytics and reporting system for insights and optimization recommendations."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ...logging import get_logger
from ..core.metrics_collector import CostMetricsCollector, CostBreakdown
from ..guardrails.cost_guardrails import GuardrailViolation
from ..autoscaling.cost_aware_scaler import ScalingDecision


class CostTrend(BaseModel):
    """Cost trend analysis over time."""
    
    period_start: datetime = Field(description="Start of the analysis period")
    period_end: datetime = Field(description="End of the analysis period")
    total_cost: float = Field(description="Total cost for the period")
    average_daily_cost: float = Field(description="Average daily cost")
    cost_growth_rate: float = Field(description="Percentage growth rate")
    peak_cost: float = Field(description="Peak cost in the period")
    lowest_cost: float = Field(description="Lowest cost in the period")
    cost_volatility: float = Field(description="Cost volatility (standard deviation)")
    currency: str = Field(default="USD", description="Currency code")


class CostOptimizationRecommendation(BaseModel):
    """Recommendation for cost optimization."""
    
    recommendation_id: str = Field(description="Unique identifier for the recommendation")
    category: str = Field(description="Category of optimization (compute, storage, network, etc.)")
    title: str = Field(description="Title of the recommendation")
    description: str = Field(description="Detailed description")
    potential_savings: float = Field(description="Potential cost savings")
    confidence: float = Field(description="Confidence level (0-1)")
    effort_level: str = Field(description="Implementation effort (low, medium, high)")
    impact_level: str = Field(description="Impact level (low, medium, high)")
    priority: int = Field(description="Priority score (1-10)")
    currency: str = Field(default="USD", description="Currency code")
    created_at: datetime = Field(description="When the recommendation was created")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")


class CostAnomaly(BaseModel):
    """Detected cost anomaly."""
    
    anomaly_id: str = Field(description="Unique identifier for the anomaly")
    anomaly_type: str = Field(description="Type of anomaly (spike, drop, unusual_pattern)")
    detected_at: datetime = Field(description="When the anomaly was detected")
    cost_value: float = Field(description="Anomalous cost value")
    expected_value: float = Field(description="Expected cost value")
    deviation_percent: float = Field(description="Percentage deviation from expected")
    severity: str = Field(description="Severity level (low, medium, high, critical)")
    description: str = Field(description="Description of the anomaly")
    currency: str = Field(default="USD", description="Currency code")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")


class CostForecast(BaseModel):
    """Cost forecast for future periods."""
    
    forecast_id: str = Field(description="Unique identifier for the forecast")
    forecast_period_start: datetime = Field(description="Start of forecast period")
    forecast_period_end: datetime = Field(description="End of forecast period")
    predicted_cost: float = Field(description="Predicted total cost")
    confidence_interval_lower: float = Field(description="Lower bound of confidence interval")
    confidence_interval_upper: float = Field(description="Upper bound of confidence interval")
    confidence_level: float = Field(description="Confidence level (0-1)")
    model_type: str = Field(description="Type of forecasting model used")
    currency: str = Field(default="USD", description="Currency code")
    created_at: datetime = Field(description="When the forecast was created")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")


class CostAnalyticsConfig(BaseModel):
    """Configuration for cost analytics."""
    
    enabled: bool = Field(default=True, description="Enable cost analytics")
    analysis_interval_hours: int = Field(default=24, description="How often to run analysis")
    anomaly_detection_enabled: bool = Field(default=True, description="Enable anomaly detection")
    forecasting_enabled: bool = Field(default=True, description="Enable cost forecasting")
    recommendation_engine_enabled: bool = Field(default=True, description="Enable optimization recommendations")
    anomaly_threshold: float = Field(default=2.0, description="Standard deviations for anomaly detection")
    forecast_horizon_days: int = Field(default=30, description="Days to forecast ahead")
    min_data_points: int = Field(default=7, description="Minimum data points for analysis")


class CostAnalytics:
    """Cost analytics and reporting system."""
    
    def __init__(
        self,
        config: CostAnalyticsConfig,
        metrics_collector: CostMetricsCollector,
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = get_logger(__name__)
        self._recommendations: List[CostOptimizationRecommendation] = []
        self._anomalies: List[CostAnomaly] = []
        self._forecasts: List[CostForecast] = []
        self._running = False
        self._analysis_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the cost analytics system."""
        if self._running:
            return
        
        self._running = True
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        self.logger.info("Cost analytics system started")
    
    async def stop(self) -> None:
        """Stop the cost analytics system."""
        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Cost analytics system stopped")
    
    async def _analysis_loop(self) -> None:
        """Main analysis loop."""
        while self._running:
            try:
                await self._run_analysis()
                await asyncio.sleep(self.config.analysis_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in cost analysis", error=str(e))
                await asyncio.sleep(300)  # 5 minute delay before retry
    
    async def _run_analysis(self) -> None:
        """Run comprehensive cost analysis."""
        try:
            # Detect anomalies
            if self.config.anomaly_detection_enabled:
                await self._detect_anomalies()
            
            # Generate forecasts
            if self.config.forecasting_enabled:
                await self._generate_forecasts()
            
            # Generate recommendations
            if self.config.recommendation_engine_enabled:
                await self._generate_recommendations()
            
            self.logger.info("Cost analysis completed")
            
        except Exception as e:
            self.logger.error("Failed to run cost analysis", error=str(e))
    
    async def _detect_anomalies(self) -> None:
        """Detect cost anomalies."""
        # Get recent cost data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=30)
        
        cost_trends = self.metrics_collector.get_cost_trends(days=30)
        
        if len(cost_trends["costs"]) < self.config.min_data_points:
            return
        
        # Calculate statistics
        costs = cost_trends["costs"]
        mean_cost = sum(costs) / len(costs)
        
        # Calculate standard deviation
        variance = sum((cost - mean_cost) ** 2 for cost in costs) / len(costs)
        std_dev = variance ** 0.5
        
        # Check for anomalies in recent data
        recent_costs = costs[-7:]  # Last 7 days
        for i, cost in enumerate(recent_costs):
            if std_dev > 0:
                z_score = abs(cost - mean_cost) / std_dev
                
                if z_score > self.config.anomaly_threshold:
                    anomaly_type = "spike" if cost > mean_cost else "drop"
                    severity = self._determine_anomaly_severity(z_score)
                    
                    anomaly = CostAnomaly(
                        anomaly_id=f"anomaly_{int(datetime.now().timestamp())}_{i}",
                        anomaly_type=anomaly_type,
                        detected_at=datetime.now(timezone.utc),
                        cost_value=cost,
                        expected_value=mean_cost,
                        deviation_percent=(cost - mean_cost) / mean_cost * 100,
                        severity=severity,
                        description=f"Cost {anomaly_type} detected: ${cost:.2f} vs expected ${mean_cost:.2f}",
                    )
                    
                    self._anomalies.append(anomaly)
                    self.logger.warning(
                        "Cost anomaly detected",
                        anomaly_id=anomaly.anomaly_id,
                        type=anomaly_type,
                        cost=cost,
                        expected=mean_cost,
                        severity=severity,
                    )
    
    def _determine_anomaly_severity(self, z_score: float) -> str:
        """Determine anomaly severity based on z-score."""
        if z_score > 4.0:
            return "critical"
        elif z_score > 3.0:
            return "high"
        elif z_score > 2.5:
            return "medium"
        else:
            return "low"
    
    async def _generate_forecasts(self) -> None:
        """Generate cost forecasts."""
        # Get historical data
        cost_trends = self.metrics_collector.get_cost_trends(days=30)
        
        if len(cost_trends["costs"]) < self.config.min_data_points:
            return
        
        # Simple linear regression forecast (in practice, you'd use more sophisticated models)
        costs = cost_trends["costs"]
        n = len(costs)
        
        # Calculate trend
        x_values = list(range(n))
        y_values = costs
        
        # Simple linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        # Forecast future costs
        forecast_start = datetime.now(timezone.utc)
        forecast_end = forecast_start + timedelta(days=self.config.forecast_horizon_days)
        
        # Predict cost for the forecast period
        future_x = n + self.config.forecast_horizon_days
        predicted_cost = slope * future_x + intercept
        
        # Calculate confidence interval (simplified)
        residuals = [y - (slope * x + intercept) for x, y in zip(x_values, y_values)]
        residual_std = (sum(r ** 2 for r in residuals) / (n - 2)) ** 0.5 if n > 2 else 0
        
        confidence_interval = 1.96 * residual_std  # 95% confidence interval
        
        forecast = CostForecast(
            forecast_id=f"forecast_{int(datetime.now().timestamp())}",
            forecast_period_start=forecast_start,
            forecast_period_end=forecast_end,
            predicted_cost=max(0, predicted_cost),  # Ensure non-negative
            confidence_interval_lower=max(0, predicted_cost - confidence_interval),
            confidence_interval_upper=predicted_cost + confidence_interval,
            confidence_level=0.95,
            model_type="linear_regression",
            created_at=datetime.now(timezone.utc),
        )
        
        self._forecasts.append(forecast)
        self.logger.info(
            "Generated cost forecast",
            forecast_id=forecast.forecast_id,
            predicted_cost=predicted_cost,
            confidence_interval=confidence_interval,
        )
    
    async def _generate_recommendations(self) -> None:
        """Generate cost optimization recommendations."""
        # Get current cost breakdown
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=7)
        
        cost_breakdown = self.metrics_collector.get_cost_breakdown(start_time, end_time)
        
        # Generate recommendations based on cost breakdown
        recommendations = []
        
        # Compute optimization recommendations
        if cost_breakdown.compute_cost > cost_breakdown.total_cost * 0.6:  # > 60% of total cost
            recommendations.append(CostOptimizationRecommendation(
                recommendation_id=f"compute_opt_{int(datetime.now().timestamp())}",
                category="compute",
                title="Optimize Compute Resources",
                description="Compute costs represent over 60% of total costs. Consider right-sizing instances or implementing auto-scaling.",
                potential_savings=cost_breakdown.compute_cost * 0.2,  # 20% savings
                confidence=0.8,
                effort_level="medium",
                impact_level="high",
                priority=8,
                created_at=datetime.now(timezone.utc),
            ))
        
        # Storage optimization recommendations
        if cost_breakdown.storage_cost > cost_breakdown.total_cost * 0.3:  # > 30% of total cost
            recommendations.append(CostOptimizationRecommendation(
                recommendation_id=f"storage_opt_{int(datetime.now().timestamp())}",
                category="storage",
                title="Optimize Storage Usage",
                description="Storage costs are high. Consider data lifecycle policies and compression.",
                potential_savings=cost_breakdown.storage_cost * 0.3,  # 30% savings
                confidence=0.7,
                effort_level="low",
                impact_level="medium",
                priority=6,
                created_at=datetime.now(timezone.utc),
            ))
        
        # Network optimization recommendations
        if cost_breakdown.network_cost > cost_breakdown.total_cost * 0.2:  # > 20% of total cost
            recommendations.append(CostOptimizationRecommendation(
                recommendation_id=f"network_opt_{int(datetime.now().timestamp())}",
                category="network",
                title="Optimize Network Usage",
                description="Network costs are significant. Consider data compression and CDN usage.",
                potential_savings=cost_breakdown.network_cost * 0.25,  # 25% savings
                confidence=0.6,
                effort_level="medium",
                impact_level="medium",
                priority=5,
                created_at=datetime.now(timezone.utc),
            ))
        
        # API optimization recommendations
        if cost_breakdown.api_cost > cost_breakdown.total_cost * 0.1:  # > 10% of total cost
            recommendations.append(CostOptimizationRecommendation(
                recommendation_id=f"api_opt_{int(datetime.now().timestamp())}",
                category="api",
                title="Optimize API Usage",
                description="API costs are notable. Consider request batching and caching strategies.",
                potential_savings=cost_breakdown.api_cost * 0.4,  # 40% savings
                confidence=0.9,
                effort_level="low",
                impact_level="high",
                priority=7,
                created_at=datetime.now(timezone.utc),
            ))
        
        # Add recommendations to the list
        self._recommendations.extend(recommendations)
        
        self.logger.info(
            "Generated optimization recommendations",
            count=len(recommendations),
            total_potential_savings=sum(r.potential_savings for r in recommendations),
        )
    
    def get_cost_trend_analysis(self, days: int = 30) -> CostTrend:
        """Get cost trend analysis for the specified period."""
        cost_trends = self.metrics_collector.get_cost_trends(days=days)
        
        if not cost_trends["costs"]:
            return CostTrend(
                period_start=datetime.now(timezone.utc) - timedelta(days=days),
                period_end=datetime.now(timezone.utc),
                total_cost=0.0,
                average_daily_cost=0.0,
                cost_growth_rate=0.0,
                peak_cost=0.0,
                lowest_cost=0.0,
                cost_volatility=0.0,
            )
        
        costs = cost_trends["costs"]
        total_cost = sum(costs)
        average_daily_cost = total_cost / len(costs) if costs else 0.0
        
        # Calculate growth rate
        if len(costs) >= 2:
            first_half = costs[:len(costs)//2]
            second_half = costs[len(costs)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            cost_growth_rate = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0.0
        else:
            cost_growth_rate = 0.0
        
        # Calculate volatility (standard deviation)
        mean_cost = average_daily_cost
        variance = sum((cost - mean_cost) ** 2 for cost in costs) / len(costs)
        cost_volatility = variance ** 0.5
        
        return CostTrend(
            period_start=datetime.now(timezone.utc) - timedelta(days=days),
            period_end=datetime.now(timezone.utc),
            total_cost=total_cost,
            average_daily_cost=average_daily_cost,
            cost_growth_rate=cost_growth_rate,
            peak_cost=max(costs),
            lowest_cost=min(costs),
            cost_volatility=cost_volatility,
        )
    
    def get_optimization_recommendations(
        self,
        category: Optional[str] = None,
        priority_min: int = 1,
        tenant_id: Optional[str] = None,
    ) -> List[CostOptimizationRecommendation]:
        """Get optimization recommendations with optional filtering."""
        recommendations = self._recommendations
        
        if category:
            recommendations = [r for r in recommendations if r.category == category]
        
        if priority_min > 1:
            recommendations = [r for r in recommendations if r.priority >= priority_min]
        
        if tenant_id:
            recommendations = [r for r in recommendations if r.tenant_id == tenant_id]
        
        # Sort by priority (highest first)
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        return recommendations
    
    def get_cost_anomalies(
        self,
        severity: Optional[str] = None,
        days: int = 30,
        tenant_id: Optional[str] = None,
    ) -> List[CostAnomaly]:
        """Get cost anomalies with optional filtering."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        anomalies = [a for a in self._anomalies if a.detected_at >= cutoff_date]
        
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        
        if tenant_id:
            anomalies = [a for a in anomalies if a.tenant_id == tenant_id]
        
        # Sort by detection time (most recent first)
        anomalies.sort(key=lambda a: a.detected_at, reverse=True)
        
        return anomalies
    
    def get_latest_forecast(self, tenant_id: Optional[str] = None) -> Optional[CostForecast]:
        """Get the most recent cost forecast."""
        forecasts = self._forecasts
        
        if tenant_id:
            forecasts = [f for f in forecasts if f.tenant_id == tenant_id]
        
        if not forecasts:
            return None
        
        # Return the most recent forecast
        return max(forecasts, key=lambda f: f.created_at)
    
    def get_analytics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get a comprehensive analytics summary."""
        cost_trend = self.get_cost_trend_analysis(days)
        recommendations = self.get_optimization_recommendations()
        anomalies = self.get_cost_anomalies(days=days)
        latest_forecast = self.get_latest_forecast()
        
        summary = {
            "cost_trend": cost_trend.model_dump(),
            "recommendations": {
                "total": len(recommendations),
                "high_priority": len([r for r in recommendations if r.priority >= 7]),
                "total_potential_savings": sum(r.potential_savings for r in recommendations),
                "by_category": {},
            },
            "anomalies": {
                "total": len(anomalies),
                "critical": len([a for a in anomalies if a.severity == "critical"]),
                "high": len([a for a in anomalies if a.severity == "high"]),
                "by_type": {},
            },
            "forecast": latest_forecast.model_dump() if latest_forecast else None,
        }
        
        # Count recommendations by category
        for rec in recommendations:
            if rec.category not in summary["recommendations"]["by_category"]:
                summary["recommendations"]["by_category"][rec.category] = 0
            summary["recommendations"]["by_category"][rec.category] += 1
        
        # Count anomalies by type
        for anomaly in anomalies:
            if anomaly.anomaly_type not in summary["anomalies"]["by_type"]:
                summary["anomalies"]["by_type"][anomaly.anomaly_type] = 0
            summary["anomalies"]["by_type"][anomaly.anomaly_type] += 1
        
        return summary
