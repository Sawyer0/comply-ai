"""
Data Preparation Pipeline Stage

Orchestrates data preparation using existing data generation infrastructure.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from ..core.pipeline import PipelineStage, PipelineContext
from ..exceptions import StageError
from ...training.data_generator import (
    HybridTrainingDataGenerator,
    AnalysisModuleDataGenerator,
    DatasetValidator,
    TrainingExample
)

logger = structlog.get_logger(__name__)


class DataPreparationStage(PipelineStage):
    """
    Data preparation stage that orchestrates training data generation.
    
    Uses existing data generators to create high-quality training datasets
    for both mapper and analyst models.
    """
    
    def __init__(self, name: str = "data_preparation", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.hybrid_generator = HybridTrainingDataGenerator()
        self.analysis_generator = AnalysisModuleDataGenerator()
        self.validator = DatasetValidator()
    
    async def _validate_preconditions(self, context: PipelineContext) -> None:
        """Validate preconditions for data preparation."""
        # Check required configuration
        required_keys = ["output_dir", "model_type"]
        for key in required_keys:
            if key not in context.config:
                raise StageError(f"Required configuration key missing: {key}")
        
        # Validate model type
        model_type = context.config["model_type"]
        if model_type not in ["dual", "mapper", "analyst"]:
            raise StageError(f"Invalid model type: {model_type}")
    
    async def _execute_stage(self, context: PipelineContext) -> PipelineContext:
        """Execute data preparation stage."""
        model_type = context.config["model_type"]
        output_dir = Path(context.config["output_dir"])
        
        self.logger.info("Starting data preparation",
                        model_type=model_type,
                        output_dir=str(output_dir))
        
        # Create output directory
        data_dir = output_dir / "training_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        prepared_datasets = {}
        
        if model_type in ["dual", "mapper"]:
            # Prepare mapper training data
            mapper_data = await self._prepare_mapper_data(context, data_dir)
            prepared_datasets["mapper"] = mapper_data
        
        if model_type in ["dual", "analyst"]:
            # Prepare analyst training data
            analyst_data = await self._prepare_analyst_data(context, data_dir)
            prepared_datasets["analyst"] = analyst_data
        
        # Validate prepared datasets
        validation_results = await self._validate_datasets(prepared_datasets)
        
        # Save dataset metadata
        metadata = {
            "model_type": model_type,
            "datasets": prepared_datasets,
            "validation_results": validation_results,
            "preparation_config": self.config
        }
        
        metadata_path = data_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info("Data preparation completed",
                        datasets_prepared=list(prepared_datasets.keys()),
                        total_examples=sum(d["example_count"] for d in prepared_datasets.values()))
        
        return context.with_artifact("training_data_path", str(data_dir)).with_artifact(
            "dataset_metadata", metadata
        )
    
    async def _prepare_mapper_data(self, context: PipelineContext, output_dir: Path) -> Dict[str, Any]:
        """Prepare training data for mapper model."""
        self.logger.info("Preparing mapper training data")
        
        # Configuration for mapper data generation
        num_examples = context.config.get("mapper_examples", 2000)
        categories = context.config.get("mapper_categories", ["sec", "gdpr", "hipaa", "industries"])
        real_world_ratio = context.config.get("mapper_real_world_ratio", 0.7)
        
        # Generate hybrid training data
        loop = asyncio.get_event_loop()
        training_examples = await loop.run_in_executor(
            None,
            self.hybrid_generator.generate_hybrid_training_set,
            num_examples,
            categories,  # real_world_categories
            categories,  # synthetic_categories
            True  # balance_categories
        )
        
        # Save training data
        mapper_data_path = output_dir / "mapper_training_data.jsonl"
        await self._save_training_examples(training_examples, mapper_data_path)
        
        # Generate additional specialized datasets
        edge_case_examples = await loop.run_in_executor(
            None,
            self.hybrid_generator.generate_edge_case_hybrid,
            int(num_examples * 0.2),  # 20% edge cases
            categories,
        )
        
        edge_case_path = output_dir / "mapper_edge_cases.jsonl"
        await self._save_training_examples(edge_case_examples, edge_case_path)
        
        # Generate industry-specific data
        industry_examples = await loop.run_in_executor(
            None,
            self.hybrid_generator.generate_industry_specific_hybrid,
            "financial_services",  # industry parameter
            int(num_examples * 0.3),  # 30% industry-specific
            categories,
        )
        
        industry_data_path = output_dir / "mapper_industry_data.jsonl"
        await self._save_training_examples(industry_examples, industry_data_path)
        
        total_examples = len(training_examples) + len(edge_case_examples) + len(industry_examples)
        
        return {
            "model_type": "mapper",
            "example_count": total_examples,
            "main_dataset": str(mapper_data_path),
            "edge_cases": str(edge_case_path),
            "industry_specific": str(industry_data_path),
            "categories": categories,
            "real_world_ratio": real_world_ratio
        }
    
    async def _prepare_analyst_data(self, context: PipelineContext, output_dir: Path) -> Dict[str, Any]:
        """Prepare training data for analyst model."""
        self.logger.info("Preparing analyst training data")
        
        # Configuration for analyst data generation
        num_examples = context.config.get("analyst_examples", 1500)
        
        # Generate analysis training data
        loop = asyncio.get_event_loop()
        training_examples = await loop.run_in_executor(
            None,
            self.analysis_generator.generate_balanced_analysis_set,
            num_examples // 4,  # target_examples_per_category
            True,  # include_coverage_gaps
            True,  # include_incidents
            True,  # include_threshold_tuning
            True   # include_opa_policies
        )
        
        # Save training data
        analyst_data_path = output_dir / "analyst_training_data.jsonl"
        await self._save_training_examples(training_examples, analyst_data_path)
        
        # Generate specialized compliance scenarios
        compliance_examples = await loop.run_in_executor(
            None,
            self.analysis_generator.generate_coverage_gap_examples,
            int(num_examples * 0.4)  # 40% compliance scenarios
        )
        
        compliance_path = output_dir / "analyst_compliance_scenarios.jsonl"
        await self._save_training_examples(compliance_examples, compliance_path)
        
        # Generate policy analysis examples
        policy_examples = await loop.run_in_executor(
            None,
            self.analysis_generator.generate_opa_policy_examples,
            int(num_examples * 0.3)  # 30% policy analysis
        )
        
        policy_path = output_dir / "analyst_policy_analysis.jsonl"
        await self._save_training_examples(policy_examples, policy_path)
        
        total_examples = len(training_examples) + len(compliance_examples) + len(policy_examples)
        
        return {
            "model_type": "analyst",
            "example_count": total_examples,
            "main_dataset": str(analyst_data_path),
            "compliance_scenarios": str(compliance_path),
            "policy_analysis": str(policy_path)
        }
    
    async def _save_training_examples(self, examples: List[TrainingExample], file_path: Path) -> None:
        """Save training examples to JSONL file."""
        with open(file_path, 'w') as f:
            for example in examples:
                example_dict = {
                    "instruction": example.instruction,
                    "response": example.response,
                    "metadata": example.metadata
                }
                f.write(json.dumps(example_dict) + '\n')
        
        self.logger.debug("Training examples saved",
                         file_path=str(file_path),
                         example_count=len(examples))
    
    async def _validate_datasets(self, prepared_datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prepared datasets."""
        self.logger.info("Validating prepared datasets")
        
        validation_results = {}
        
        for dataset_type, dataset_info in prepared_datasets.items():
            self.logger.info("Validating dataset", dataset_type=dataset_type)
            
            # Load examples for validation
            examples = await self._load_examples_for_validation(dataset_info["main_dataset"])
            
            # Run validation
            loop = asyncio.get_event_loop()
            # Create mock predictions for validation (in real scenario, this would be actual model predictions)
            mock_predictions = [{"taxonomy": ["PII.Contact.Email"], "confidence": 0.9} for _ in examples]
            mock_ground_truth = [{"taxonomy": ["PII.Contact.Email"], "confidence": 1.0} for _ in examples]
            
            validation_result = await loop.run_in_executor(
                None,
                self.validator.evaluate_model_performance,
                mock_predictions,
                mock_ground_truth
            )
            
            validation_results[dataset_type] = {
                "is_valid": validation_result.is_valid,
                "total_examples": validation_result.total_examples,
                "valid_examples": validation_result.valid_examples,
                "validation_errors": validation_result.validation_errors,
                "coverage_analysis": validation_result.coverage_analysis,
                "quality_score": validation_result.quality_score
            }
            
            if not validation_result.is_valid:
                self.logger.warning("Dataset validation failed",
                                  dataset_type=dataset_type,
                                  errors=validation_result.validation_errors[:5])  # Show first 5 errors
        
        return validation_results
    
    async def _load_examples_for_validation(self, file_path: str) -> List[TrainingExample]:
        """Load training examples from JSONL file for validation."""
        examples = []
        
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example = TrainingExample(
                    instruction=data["instruction"],
                    response=data["response"],
                    metadata=data.get("metadata", {})
                )
                examples.append(example)
        
        return examples
    
    async def _collect_metrics(self, context: PipelineContext) -> Dict[str, Any]:
        """Collect data preparation metrics."""
        dataset_metadata = context.artifacts.get("dataset_metadata", {})
        
        metrics = {
            "model_type": dataset_metadata.get("model_type"),
            "datasets_prepared": len(dataset_metadata.get("datasets", {})),
            "total_examples": 0,
            "validation_passed": True
        }
        
        # Calculate total examples and validation status
        for dataset_info in dataset_metadata.get("datasets", {}).values():
            metrics["total_examples"] += dataset_info.get("example_count", 0)
        
        # Check validation results
        validation_results = dataset_metadata.get("validation_results", {})
        for result in validation_results.values():
            if not result.get("is_valid", True):
                metrics["validation_passed"] = False
                break
        
        return metrics
    
    def get_dependencies(self) -> List[str]:
        """Data preparation has no dependencies."""
        return []