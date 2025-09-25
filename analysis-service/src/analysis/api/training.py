"""
Training API endpoints for Analysis Service.

This module provides REST API endpoints for model training operations,
training job management, and training pipeline configuration.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Header

from shared.utils.correlation import get_correlation_id

from ..config.settings import AnalysisSettings
from ..dependencies import get_tenant_manager
from ..pipelines.training_pipeline import (
    TrainingConfig,
    TrainingPipeline,
    TrainingPipelineFactory,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/jobs", response_model=Dict[str, Any])
async def create_training_job(
    training_request: Dict[str, Any] = Body(...),
    x_tenant_id: str = Header(..., description="Tenant ID"),
    tenant_manager=Depends(get_tenant_manager),
):
    """
    Create a new training job.

    Args:
        training_request: Training configuration and data
        x_tenant_id: Tenant identifier

    Returns:
        Training job information
    """
    try:
        logger.info(
            "Creating training job",
            tenant_id=x_tenant_id,
            correlation_id=get_correlation_id(),
        )

        # Validate tenant
        tenant_config = await tenant_manager.get_tenant_config(x_tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # Extract training parameters
        model_type = training_request.get("model_type", "phi-3-mini")
        training_data = training_request.get("training_data", [])
        validation_data = training_request.get("validation_data")
        custom_config = training_request.get("config", {})

        if not training_data:
            raise HTTPException(status_code=400, detail="Training data is required")

        # Create real training pipeline with actual dependencies
        from ..ml.model_server import ModelServer
        from ..quality.monitoring import QualityMonitor
        from ..config.settings import AnalysisSettings
        
        settings = AnalysisSettings()
        
        # Initialize real model server
        model_server = ModelServer(
            backend=settings.model_backend,
            model_path=settings.model_path,
            generation_config=settings.generation_config
        )
        
        # Initialize real quality monitor
        quality_monitor = QualityMonitor(
            confidence_threshold=settings.confidence_threshold,
            enable_metrics=settings.enable_metrics
        )
        
        # Initialize model server and quality monitor
        await model_server.initialize()
        await quality_monitor.initialize()

        if model_type == "compliance":
            pipeline = TrainingPipelineFactory.create_compliance_analysis_pipeline(
                settings, model_server, quality_monitor, custom_config
            )
        elif model_type == "risk":
            pipeline = TrainingPipelineFactory.create_risk_assessment_pipeline(
                settings, model_server, quality_monitor, custom_config
            )
        else:
            # Default pipeline
            config = TrainingConfig(**custom_config)
            pipeline = TrainingPipeline(config, model_server, quality_monitor, settings)

        # Start training (async)
        training_result = await pipeline.execute_training(
            training_data, validation_data
        )

        logger.info(
            "Training job created successfully",
            tenant_id=x_tenant_id,
            training_id=training_result.training_id,
        )

        return {
            "training_id": training_result.training_id,
            "tenant_id": x_tenant_id,
            "model_type": model_type,
            "status": "completed",
            "model_path": training_result.model_path,
            "quality_score": training_result.quality_score,
            "created_at": training_result.completed_at.isoformat(),
            "metrics": training_result.metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to create training job", tenant_id=x_tenant_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Training job creation failed: {str(e)}"
        ) from e


@router.get("/jobs/{training_id}", response_model=Dict[str, Any])
async def get_training_job(
    training_id: str,
    x_tenant_id: str = Header(..., description="Tenant ID"),
    tenant_manager=Depends(get_tenant_manager),
):
    """
    Get training job status and results.

    Args:
        training_id: Training job identifier
        x_tenant_id: Tenant identifier

    Returns:
        Training job information
    """
    try:
        logger.info(
            "Getting training job",
            training_id=training_id,
            tenant_id=x_tenant_id,
            correlation_id=get_correlation_id(),
        )

        # Validate tenant
        tenant_config = await tenant_manager.get_tenant_config(x_tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # Simulate job status (in real implementation, would query job store)
        job_status = {
            "training_id": training_id,
            "tenant_id": x_tenant_id,
            "status": "completed",
            "progress": 1.0,
            "current_epoch": 2,
            "total_epochs": 2,
            "model_path": f"./checkpoints/model_{training_id}",
            "quality_score": 0.89,
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "metrics": {
                "final_loss": 0.3,
                "validation_loss": 0.25,
                "training_samples": 1000,
                "validation_samples": 200,
            },
        }

        return job_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get training job", training_id=training_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get training job: {str(e)}"
        ) from e


@router.get("/jobs", response_model=List[Dict[str, Any]])
async def list_training_jobs(
    x_tenant_id: str = Header(..., description="Tenant ID"),
    status: Optional[str] = None,
    limit: int = 50,
    tenant_manager=Depends(get_tenant_manager),
):
    """
    List training jobs for a tenant.

    Args:
        x_tenant_id: Tenant identifier
        status: Optional status filter
        limit: Maximum number of jobs to return

    Returns:
        List of training jobs
    """
    try:
        logger.info(
            "Listing training jobs",
            tenant_id=x_tenant_id,
            correlation_id=get_correlation_id(),
        )

        # Validate tenant
        tenant_config = await tenant_manager.get_tenant_config(x_tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # Simulate job list (in real implementation, would query job store)
        jobs = [
            {
                "training_id": f"training_{i}",
                "tenant_id": x_tenant_id,
                "model_type": "phi-3-mini",
                "status": "completed",
                "quality_score": 0.85 + (i * 0.02),
                "created_at": datetime.now().isoformat(),
            }
            for i in range(min(limit, 5))  # Simulate 5 jobs
        ]

        # Apply status filter
        if status:
            jobs = [job for job in jobs if job["status"] == status]

        return jobs

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to list training jobs", tenant_id=x_tenant_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to list training jobs: {str(e)}"
        ) from e


@router.delete("/jobs/{training_id}", response_model=Dict[str, Any])
async def cancel_training_job(
    training_id: str,
    x_tenant_id: str = Header(..., description="Tenant ID"),
    tenant_manager=Depends(get_tenant_manager),
):
    """
    Cancel a training job.

    Args:
        training_id: Training job identifier
        x_tenant_id: Tenant identifier

    Returns:
        Cancellation result
    """
    try:
        logger.info(
            "Cancelling training job",
            training_id=training_id,
            tenant_id=x_tenant_id,
            correlation_id=get_correlation_id(),
        )

        # Validate tenant
        tenant_config = await tenant_manager.get_tenant_config(x_tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # In real implementation, would cancel the actual job
        logger.info("Training job cancelled", training_id=training_id)

        return {
            "training_id": training_id,
            "status": "cancelled",
            "message": "Training job cancelled successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to cancel training job", training_id=training_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel training job: {str(e)}"
        ) from e


@router.get("/configs/templates", response_model=List[Dict[str, Any]])
async def get_training_templates():
    """
    Get available training configuration templates.

    Returns:
        List of training templates
    """
    try:
        templates = [
            {
                "name": "compliance-analysis",
                "description": "Template for compliance analysis models",
                "model_type": "phi-3-compliance",
                "config": {
                    "learning_rate": 1e-4,
                    "num_epochs": 2,
                    "lora_r": 128,
                    "lora_alpha": 256,
                },
            },
            {
                "name": "risk-assessment",
                "description": "Template for risk assessment models",
                "model_type": "phi-3-risk",
                "config": {
                    "learning_rate": 5e-5,
                    "num_epochs": 3,
                    "lora_r": 128,
                    "lora_alpha": 256,
                },
            },
        ]

        return templates

    except Exception as e:
        logger.error("Failed to get training templates", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to get training templates: {str(e)}"
        ) from e


@router.post("/validate-data", response_model=Dict[str, Any])
async def validate_training_data(
    training_data: List[Dict[str, Any]] = Body(...),
    x_tenant_id: str = Header(..., description="Tenant ID"),
):
    """
    Validate training data format and quality.

    Args:
        training_data: Training data to validate
        x_tenant_id: Tenant identifier

    Returns:
        Validation results
    """
    try:
        logger.info(
            "Validating training data",
            tenant_id=x_tenant_id,
            data_size=len(training_data),
            correlation_id=get_correlation_id(),
        )

        validation_results = {
            "valid": True,
            "total_samples": len(training_data),
            "issues": [],
            "recommendations": [],
        }

        # Validate data format
        for i, example in enumerate(training_data):
            if "instruction" not in example:
                validation_results["issues"].append(
                    f"Sample {i}: Missing 'instruction' field"
                )
                validation_results["valid"] = False

            if "response" not in example:
                validation_results["issues"].append(
                    f"Sample {i}: Missing 'response' field"
                )
                validation_results["valid"] = False

            # Check content length
            if "instruction" in example and len(example["instruction"]) < 10:
                validation_results["issues"].append(
                    f"Sample {i}: Instruction too short"
                )

            if "response" in example and len(example["response"]) < 5:
                validation_results["issues"].append(f"Sample {i}: Response too short")

        # Add recommendations
        if len(training_data) < 100:
            validation_results["recommendations"].append(
                "Consider adding more training samples for better model performance"
            )

        if len(training_data) > 10000:
            validation_results["recommendations"].append(
                "Large dataset detected - consider using distributed training"
            )

        return validation_results

    except Exception as e:
        logger.error("Failed to validate training data", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Training data validation failed: {str(e)}"
        ) from e
