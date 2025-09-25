"""
Quality Manager for comprehensive quality orchestration.

This module provides a centralized quality manager that coordinates
all quality monitoring, alerting, and degradation detection.
"""

from typing import Optional, Any, Dict
import structlog

from .monitoring import QualityMonitor
from .alerting_system import QualityAlertingSystem
from .degradation import QualityDegradationDetector
from .alert_manager import AlertManager

logger = structlog.get_logger(__name__)


class QualityManager:
    """
    Centralized quality manager that orchestrates all quality components.
    
    This class provides a unified interface for quality operations including
    monitoring, alerting, and degradation detection.
    """
    
    def __init__(
        self,
        database_manager: Optional[Any] = None,
        connection_pool: Optional[Any] = None,
    ):
        """
        Initialize the quality manager.
        
        Args:
            database_manager: Database manager instance
            connection_pool: Connection pool manager instance
        """
        self.database_manager = database_manager
        self.connection_pool = connection_pool
        
        # Initialize quality monitor
        self.quality_monitor = QualityMonitor()
        
        # Initialize alerting system
        self.alerting_system = QualityAlertingSystem()
        
        # Initialize degradation detector
        self.degradation_detector = QualityDegradationDetector()
        
        # Initialize alert manager
        self.alert_manager = AlertManager()
        
        logger.info("QualityManager initialized")
    
    async def start_request_monitoring(self, request: Any) -> None:
        """
        Start monitoring a request.
        
        Args:
            request: The request to monitor
        """
        try:
            await self.quality_monitor.start_request_monitoring(request)
        except Exception as e:
            logger.warning("Failed to start request monitoring", error=str(e))
    
    async def end_request_monitoring(self, request: Any, response: Any) -> None:
        """
        End monitoring a request.
        
        Args:
            request: The request that was monitored
            response: The response from the request
        """
        try:
            await self.quality_monitor.end_request_monitoring(request, response)
            
            # Check for quality degradation
            degradation = await self.degradation_detector.check_degradation(
                request, response
            )
            
            if degradation:
                # Generate alert if degradation detected
                await self.alerting_system.generate_alert(degradation)
                
        except Exception as e:
            logger.warning("Failed to end request monitoring", error=str(e))
    
    async def get_quality_status(self) -> Dict[str, Any]:
        """
        Get the current quality status.
        
        Returns:
            Dictionary containing quality status information
        """
        try:
            status = {
                "monitoring": {
                    "active": self.quality_monitor is not None,
                    "metrics_count": len(await self.quality_monitor.get_metrics()),
                },
                "alerting": {
                    "active": self.alerting_system is not None,
                    "alerts_count": len(await self.alert_manager.get_active_alerts()),
                },
                "degradation": {
                    "detector_active": self.degradation_detector is not None,
                },
            }
            
            return status
            
        except Exception as e:
            logger.error("Failed to get quality status", error=str(e))
            return {"error": str(e)}
    
    async def update_quality_config(self, config: Dict[str, Any]) -> bool:
        """
        Update quality configuration.
        
        Args:
            config: Quality configuration dictionary
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Update monitoring config
            if "monitoring" in config:
                await self.quality_monitor.update_config(config["monitoring"])
            
            # Update alerting config
            if "alerting" in config:
                await self.alerting_system.update_config(config["alerting"])
            
            # Update degradation detection config
            if "degradation" in config:
                await self.degradation_detector.update_config(config["degradation"])
            
            logger.info("Quality configuration updated", config_keys=list(config.keys()))
            return True
            
        except Exception as e:
            logger.error("Failed to update quality configuration", error=str(e))
            return False
    
    async def get_quality_metrics(self) -> Dict[str, Any]:
        """
        Get current quality metrics.
        
        Returns:
            Dictionary containing quality metrics
        """
        try:
            return await self.quality_monitor.get_metrics()
        except Exception as e:
            logger.error("Failed to get quality metrics", error=str(e))
            return {}
    
    async def get_active_alerts(self) -> Dict[str, Any]:
        """
        Get active quality alerts.
        
        Returns:
            Dictionary containing active alerts
        """
        try:
            return await self.alert_manager.get_active_alerts()
        except Exception as e:
            logger.error("Failed to get active alerts", error=str(e))
            return {}
