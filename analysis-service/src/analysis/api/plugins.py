"""
Plugin API endpoints for Analysis Service.

This module provides REST API endpoints for plugin management, execution,
and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Dict, Any, Optional
import logging

from ..plugins import (
    PluginManager,
    PluginRegistry,
    AnalysisRequest,
    AnalysisResult,
    PluginType,
    PluginStatus,
)
from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plugins", tags=["plugins"])


# Import dependency injection
from ..dependencies import get_plugin_manager


@router.get("/", response_model=Dict[str, Any])
async def list_plugins(
    plugin_type: Optional[str] = Query(None, description="Filter by plugin type"),
    status: Optional[str] = Query(None, description="Filter by plugin status"),
    plugin_manager: PluginManager = Depends(get_plugin_manager),
):
    """
    List all registered plugins with their status and metadata.

    Args:
        plugin_type: Optional plugin type filter
        status: Optional status filter
        plugin_manager: Plugin manager instance

    Returns:
        List of plugins with metadata
    """
    try:
        logger.info("Listing plugins", correlation_id=get_correlation_id())

        registry = plugin_manager.get_registry()
        plugins_info = registry.list_plugins()

        # Apply filters
        filtered_plugins = {}
        for plugin_name, plugin_info in plugins_info.items():
            # Filter by plugin type
            if plugin_type:
                try:
                    filter_type = PluginType(plugin_type)
                    if plugin_info["metadata"]["plugin_type"] != filter_type.value:
                        continue
                except ValueError:
                    raise HTTPException(
                        status_code=400, detail=f"Invalid plugin type: {plugin_type}"
                    )

            # Filter by status
            if status:
                try:
                    filter_status = PluginStatus(status)
                    if plugin_info["status"] != filter_status.value:
                        continue
                except ValueError:
                    raise HTTPException(
                        status_code=400, detail=f"Invalid status: {status}"
                    )

            filtered_plugins[plugin_name] = plugin_info

        return {
            "plugins": filtered_plugins,
            "total_count": len(filtered_plugins),
            "filters": {"plugin_type": plugin_type, "status": status},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list plugins", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list plugins: {str(e)}")


@router.get("/{plugin_name}", response_model=Dict[str, Any])
async def get_plugin_info(
    plugin_name: str, plugin_manager: PluginManager = Depends(get_plugin_manager)
):
    """
    Get detailed information about a specific plugin.

    Args:
        plugin_name: Plugin name
        plugin_manager: Plugin manager instance

    Returns:
        Plugin information
    """
    try:
        logger.info(
            "Getting plugin info",
            plugin_name=plugin_name,
            correlation_id=get_correlation_id(),
        )

        registry = plugin_manager.get_registry()
        plugin = registry.get_plugin(plugin_name)

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        metadata = registry.plugin_metadata.get(plugin_name)
        status = registry.plugin_status.get(plugin_name)
        errors = registry.plugin_errors.get(plugin_name, [])

        # Get health status
        health_info = await plugin_manager.get_plugin_health(plugin_name)

        return {
            "plugin_name": plugin_name,
            "metadata": metadata.dict() if metadata else {},
            "status": status.value if status else "unknown",
            "health": health_info,
            "errors": errors,
            "capabilities": metadata.capabilities if metadata else [],
            "supported_frameworks": metadata.supported_frameworks if metadata else [],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get plugin info", plugin_name=plugin_name, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to get plugin info: {str(e)}"
        )


@router.post("/{plugin_name}/initialize", response_model=Dict[str, Any])
async def initialize_plugin(
    plugin_name: str,
    config: Dict[str, Any] = Body(default={}),
    plugin_manager: PluginManager = Depends(get_plugin_manager),
):
    """
    Initialize a specific plugin with configuration.

    Args:
        plugin_name: Plugin name
        config: Plugin configuration
        plugin_manager: Plugin manager instance

    Returns:
        Initialization result
    """
    try:
        logger.info(
            "Initializing plugin",
            plugin_name=plugin_name,
            correlation_id=get_correlation_id(),
        )

        success = await plugin_manager.initialize_plugin(plugin_name, config)

        if not success:
            registry = plugin_manager.get_registry()
            errors = registry.plugin_errors.get(plugin_name, [])
            raise HTTPException(
                status_code=500,
                detail=f"Plugin initialization failed: {'; '.join(errors) if errors else 'Unknown error'}",
            )

        logger.info("Plugin initialized successfully", plugin_name=plugin_name)

        return {
            "plugin_name": plugin_name,
            "status": "initialized",
            "message": "Plugin initialized successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to initialize plugin", plugin_name=plugin_name, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize plugin: {str(e)}"
        )


@router.post("/{plugin_name}/analyze", response_model=Dict[str, Any])
async def execute_plugin_analysis(
    plugin_name: str,
    request: AnalysisRequest,
    plugin_manager: PluginManager = Depends(get_plugin_manager),
):
    """
    Execute analysis using a specific plugin.

    Args:
        plugin_name: Plugin name
        request: Analysis request
        plugin_manager: Plugin manager instance

    Returns:
        Analysis result
    """
    try:
        logger.info(
            "Executing plugin analysis",
            plugin_name=plugin_name,
            request_id=request.request_id,
            correlation_id=get_correlation_id(),
        )

        result = await plugin_manager.execute_analysis(plugin_name, request)

        if not result:
            raise HTTPException(status_code=500, detail="Analysis execution failed")

        logger.info(
            "Plugin analysis completed",
            plugin_name=plugin_name,
            request_id=request.request_id,
            confidence=result.confidence,
        )

        return {
            "request_id": result.request_id,
            "plugin_name": result.plugin_name,
            "plugin_version": result.plugin_version,
            "confidence": result.confidence,
            "result_data": result.result_data,
            "processing_time_ms": result.processing_time_ms,
            "metadata": result.metadata,
            "errors": result.errors,
            "warnings": result.warnings,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to execute plugin analysis",
            plugin_name=plugin_name,
            request_id=request.request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to execute analysis: {str(e)}"
        )


@router.post("/{plugin_name}/batch-analyze", response_model=Dict[str, Any])
async def execute_plugin_batch_analysis(
    plugin_name: str,
    requests: List[AnalysisRequest],
    plugin_manager: PluginManager = Depends(get_plugin_manager),
):
    """
    Execute batch analysis using a specific plugin.

    Args:
        plugin_name: Plugin name
        requests: List of analysis requests
        plugin_manager: Plugin manager instance

    Returns:
        Batch analysis results
    """
    try:
        logger.info(
            "Executing plugin batch analysis",
            plugin_name=plugin_name,
            request_count=len(requests),
            correlation_id=get_correlation_id(),
        )

        results = await plugin_manager.execute_batch_analysis(plugin_name, requests)

        # Calculate summary statistics
        successful_results = [r for r in results if not r.errors]
        failed_results = [r for r in results if r.errors]

        avg_confidence = 0.0
        if successful_results:
            avg_confidence = sum(r.confidence for r in successful_results) / len(
                successful_results
            )

        avg_processing_time = 0.0
        if results:
            avg_processing_time = sum(r.processing_time_ms for r in results) / len(
                results
            )

        logger.info(
            "Plugin batch analysis completed",
            plugin_name=plugin_name,
            total_requests=len(requests),
            successful_results=len(successful_results),
            failed_results=len(failed_results),
        )

        return {
            "plugin_name": plugin_name,
            "total_requests": len(requests),
            "results": [
                {
                    "request_id": result.request_id,
                    "confidence": result.confidence,
                    "result_data": result.result_data,
                    "processing_time_ms": result.processing_time_ms,
                    "metadata": result.metadata,
                    "errors": result.errors,
                    "warnings": result.warnings,
                }
                for result in results
            ],
            "summary": {
                "successful_count": len(successful_results),
                "failed_count": len(failed_results),
                "success_rate": (
                    len(successful_results) / len(requests) * 100 if requests else 0
                ),
                "avg_confidence": avg_confidence,
                "avg_processing_time_ms": avg_processing_time,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to execute plugin batch analysis",
            plugin_name=plugin_name,
            request_count=len(requests),
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to execute batch analysis: {str(e)}"
        )


@router.get("/{plugin_name}/health", response_model=Dict[str, Any])
async def get_plugin_health(
    plugin_name: str, plugin_manager: PluginManager = Depends(get_plugin_manager)
):
    """
    Get health status of a specific plugin.

    Args:
        plugin_name: Plugin name
        plugin_manager: Plugin manager instance

    Returns:
        Plugin health status
    """
    try:
        logger.info(
            "Getting plugin health",
            plugin_name=plugin_name,
            correlation_id=get_correlation_id(),
        )

        health_info = await plugin_manager.get_plugin_health(plugin_name)

        return {
            "plugin_name": plugin_name,
            "health": health_info,
            "timestamp": health_info.get("last_check", "unknown"),
        }

    except Exception as e:
        logger.error(
            "Failed to get plugin health", plugin_name=plugin_name, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get plugin health: {str(e)}"
        )


@router.post("/{plugin_name}/reload", response_model=Dict[str, Any])
async def reload_plugin(
    plugin_name: str, plugin_manager: PluginManager = Depends(get_plugin_manager)
):
    """
    Reload a specific plugin.

    Args:
        plugin_name: Plugin name
        plugin_manager: Plugin manager instance

    Returns:
        Reload result
    """
    try:
        logger.info(
            "Reloading plugin",
            plugin_name=plugin_name,
            correlation_id=get_correlation_id(),
        )

        success = await plugin_manager.reload_plugin(plugin_name)

        if not success:
            raise HTTPException(status_code=500, detail="Plugin reload failed")

        logger.info("Plugin reloaded successfully", plugin_name=plugin_name)

        return {
            "plugin_name": plugin_name,
            "status": "reloaded",
            "message": "Plugin reloaded successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reload plugin", plugin_name=plugin_name, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to reload plugin: {str(e)}"
        )


@router.get("/types/{plugin_type}/plugins", response_model=List[Dict[str, Any]])
async def get_plugins_by_type(
    plugin_type: str, plugin_manager: PluginManager = Depends(get_plugin_manager)
):
    """
    Get all plugins of a specific type.

    Args:
        plugin_type: Plugin type
        plugin_manager: Plugin manager instance

    Returns:
        List of plugins of the specified type
    """
    try:
        logger.info(
            "Getting plugins by type",
            plugin_type=plugin_type,
            correlation_id=get_correlation_id(),
        )

        # Validate plugin type
        try:
            pt = PluginType(plugin_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid plugin type: {plugin_type}"
            )

        registry = plugin_manager.get_registry()
        plugins = registry.get_plugins_by_type(pt)

        result = []
        for plugin in plugins:
            # Find plugin name
            plugin_name = None
            for name, p in registry.plugins.items():
                if p == plugin:
                    plugin_name = name
                    break

            if plugin_name:
                metadata = registry.plugin_metadata.get(plugin_name)
                status = registry.plugin_status.get(plugin_name)

                result.append(
                    {
                        "plugin_name": plugin_name,
                        "metadata": metadata.dict() if metadata else {},
                        "status": status.value if status else "unknown",
                    }
                )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get plugins by type", plugin_type=plugin_type, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get plugins by type: {str(e)}"
        )


@router.post("/discover", response_model=Dict[str, Any])
async def discover_plugins(plugin_manager: PluginManager = Depends(get_plugin_manager)):
    """
    Discover and load new plugins from configured directories.

    Args:
        plugin_manager: Plugin manager instance

    Returns:
        Discovery result
    """
    try:
        logger.info("Discovering plugins", correlation_id=get_correlation_id())

        discovered_plugins = await plugin_manager.discover_plugins()

        logger.info("Plugin discovery completed", count=len(discovered_plugins))

        return {
            "discovered_plugins": discovered_plugins,
            "count": len(discovered_plugins),
            "message": "Plugin discovery completed",
        }

    except Exception as e:
        logger.error("Failed to discover plugins", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to discover plugins: {str(e)}"
        )


@router.post("/initialize-all", response_model=Dict[str, Any])
async def initialize_all_plugins(
    plugin_manager: PluginManager = Depends(get_plugin_manager),
):
    """
    Initialize all registered plugins.

    Args:
        plugin_manager: Plugin manager instance

    Returns:
        Initialization results
    """
    try:
        logger.info("Initializing all plugins", correlation_id=get_correlation_id())

        results = await plugin_manager.initialize_all_plugins()

        successful_count = sum(1 for success in results.values() if success)
        failed_count = len(results) - successful_count

        logger.info(
            "Plugin initialization completed",
            successful=successful_count,
            failed=failed_count,
        )

        return {
            "results": results,
            "summary": {
                "total_plugins": len(results),
                "successful": successful_count,
                "failed": failed_count,
                "success_rate": successful_count / len(results) * 100 if results else 0,
            },
            "message": "Plugin initialization completed",
        }

    except Exception as e:
        logger.error("Failed to initialize all plugins", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize all plugins: {str(e)}"
        )
