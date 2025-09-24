"""
Light Hyperparameter Sweeps for Compliance AI Models

Mapper: try r ∈ {64, 128}, α ∈ {128, 256}, lora_dropout ∈ {0.05, 0.1}, lr ∈ {2e-5, 5e-5}
Analyst: keep it small (r=8–16), focus on prompt templates and instruction packing; tune only lr and lora_dropout
"""

import itertools
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class LoRAConfig:
    """LoRA configuration for hyperparameter sweeps."""
    r: int
    alpha: int
    dropout: float
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """Training configuration for hyperparameter sweeps."""
    learning_rate: float
    num_train_epochs: int = 3
    max_steps: int = 1000
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

@dataclass
class SweepConfig:
    """Complete sweep configuration."""
    lora: LoRAConfig
    training: TrainingConfig
    model_type: str  # "mapper" or "analyst"
    sweep_id: str
    description: str = ""

class HyperparameterSweepGenerator:
    """Generates hyperparameter sweep configurations for both models."""
    
    def __init__(self):
        self.mapper_sweeps = []
        self.analyst_sweeps = []
    
    def generate_mapper_sweeps(self) -> List[SweepConfig]:
        """Generate hyperparameter sweeps for Llama-3-8B Mapper."""
        print("🔥 Generating Mapper hyperparameter sweeps...")
        
        # Mapper sweep parameters
        lora_r_values = [64, 128]  # Reduced from 256 for latency optimization
        lora_alpha_values = [128, 256]  # 2x rank rule
        lora_dropout_values = [0.05, 0.1]
        learning_rate_values = [2e-5, 5e-5]
        
        # Generate all combinations
        sweep_id = 0
        for r, alpha, dropout, lr in itertools.product(
            lora_r_values, lora_alpha_values, lora_dropout_values, learning_rate_values
        ):
            lora_config = LoRAConfig(
                r=r,
                alpha=alpha,
                dropout=dropout
            )
            
            training_config = TrainingConfig(
                learning_rate=lr,
                num_train_epochs=3,
                max_steps=1000,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=16,
                warmup_steps=100
            )
            
            sweep_config = SweepConfig(
                lora=lora_config,
                training=training_config,
                model_type="mapper",
                sweep_id=f"mapper_sweep_{sweep_id:03d}",
                description=f"Mapper: r={r}, α={alpha}, dropout={dropout}, lr={lr}"
            )
            
            self.mapper_sweeps.append(sweep_config)
            sweep_id += 1
        
        print(f"✅ Generated {len(self.mapper_sweeps)} Mapper sweep configurations")
        return self.mapper_sweeps
    
    def generate_analyst_sweeps(self) -> List[SweepConfig]:
        """Generate hyperparameter sweeps for Phi-3 Analyst."""
        print("🔥 Generating Analyst hyperparameter sweeps...")
        
        # Analyst sweep parameters (keep it small, focus on templates)
        lora_r_values = [8, 16]  # Small ranks for efficiency
        lora_alpha_values = [16, 32]  # 2x rank rule
        lora_dropout_values = [0.05, 0.1]
        learning_rate_values = [2e-5, 5e-5]
        
        # Generate all combinations
        sweep_id = 0
        for r, alpha, dropout, lr in itertools.product(
            lora_r_values, lora_alpha_values, lora_dropout_values, learning_rate_values
        ):
            lora_config = LoRAConfig(
                r=r,
                alpha=alpha,
                dropout=dropout
            )
            
            training_config = TrainingConfig(
                learning_rate=lr,
                num_train_epochs=4,  # More epochs for Analyst
                max_steps=1000,
                per_device_train_batch_size=4,  # Larger batch for Phi-3
                gradient_accumulation_steps=8,  # Effective batch size = 32
                warmup_steps=100
            )
            
            sweep_config = SweepConfig(
                lora=lora_config,
                training=training_config,
                model_type="analyst",
                sweep_id=f"analyst_sweep_{sweep_id:03d}",
                description=f"Analyst: r={r}, α={alpha}, dropout={dropout}, lr={lr}"
            )
            
            self.analyst_sweeps.append(sweep_config)
            sweep_id += 1
        
        print(f"✅ Generated {len(self.analyst_sweeps)} Analyst sweep configurations")
        return self.analyst_sweeps
    
    def generate_all_sweeps(self) -> Dict[str, Any]:
        """Generate all hyperparameter sweeps."""
        mapper_sweeps = self.generate_mapper_sweeps()
        analyst_sweeps = self.generate_analyst_sweeps()
        
        return {
            "mapper_sweeps": mapper_sweeps,
            "analyst_sweeps": analyst_sweeps,
            "total_sweeps": len(mapper_sweeps) + len(analyst_sweeps)
        }
    
    def save_sweeps(self, output_dir: str = "config/sweeps"):
        """Save sweep configurations to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_sweeps = self.generate_all_sweeps()
        
        # Save individual sweep files
        for sweep in all_sweeps["mapper_sweeps"]:
            sweep_file = output_path / f"{sweep.sweep_id}.json"
            with open(sweep_file, 'w') as f:
                json.dump(asdict(sweep), f, indent=2)
        
        for sweep in all_sweeps["analyst_sweeps"]:
            sweep_file = output_path / f"{sweep.sweep_id}.json"
            with open(sweep_file, 'w') as f:
                json.dump(asdict(sweep), f, indent=2)
        
        # Save combined files
        mapper_file = output_path / "mapper_sweeps.json"
        with open(mapper_file, 'w') as f:
            json.dump([asdict(sweep) for sweep in all_sweeps["mapper_sweeps"]], f, indent=2)
        
        analyst_file = output_path / "analyst_sweeps.json"
        with open(analyst_file, 'w') as f:
            json.dump([asdict(sweep) for sweep in all_sweeps["analyst_sweeps"]], f, indent=2)
        
        # Save summary
        summary_file = output_path / "sweep_summary.json"
        summary = {
            "total_sweeps": all_sweeps["total_sweeps"],
            "mapper_sweeps": len(all_sweeps["mapper_sweeps"]),
            "analyst_sweeps": len(all_sweeps["analyst_sweeps"]),
            "mapper_parameters": {
                "lora_r": [64, 128],
                "lora_alpha": [128, 256],
                "lora_dropout": [0.05, 0.1],
                "learning_rate": [2e-5, 5e-5]
            },
            "analyst_parameters": {
                "lora_r": [8, 16],
                "lora_alpha": [16, 32],
                "lora_dropout": [0.05, 0.1],
                "learning_rate": [2e-5, 5e-5]
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Saved {all_sweeps['total_sweeps']} sweep configurations to {output_path}")
        print(f"  - Mapper sweeps: {len(all_sweeps['mapper_sweeps'])}")
        print(f"  - Analyst sweeps: {len(all_sweeps['analyst_sweeps'])}")
        print(f"  - Individual files: {all_sweeps['total_sweeps']}")
        print(f"  - Combined files: mapper_sweeps.json, analyst_sweeps.json")
        print(f"  - Summary: sweep_summary.json")

class SweepRunner:
    """Runs hyperparameter sweeps with Weights & Biases integration."""
    
    def __init__(self, project_name: str = "comply-ai-hyperparameter-sweeps"):
        self.project_name = project_name
        self.sweep_configs = {}
    
    def load_sweeps(self, sweep_dir: str = "config/sweeps"):
        """Load sweep configurations from files."""
        sweep_path = Path(sweep_dir)
        
        if not sweep_path.exists():
            print(f"⚠️ Sweep directory {sweep_path} does not exist")
            return
        
        # Load mapper sweeps
        mapper_file = sweep_path / "mapper_sweeps.json"
        if mapper_file.exists():
            with open(mapper_file, 'r') as f:
                self.sweep_configs["mapper"] = json.load(f)
            print(f"✅ Loaded {len(self.sweep_configs['mapper'])} Mapper sweeps")
        
        # Load analyst sweeps
        analyst_file = sweep_path / "analyst_sweeps.json"
        if analyst_file.exists():
            with open(analyst_file, 'r') as f:
                self.sweep_configs["analyst"] = json.load(f)
            print(f"✅ Loaded {len(self.sweep_configs['analyst'])} Analyst sweeps")
    
    def create_wandb_sweep_config(self, model_type: str) -> Dict[str, Any]:
        """Create Weights & Biases sweep configuration."""
        if model_type not in self.sweep_configs:
            raise ValueError(f"No sweeps loaded for model type: {model_type}")
        
        sweeps = self.sweep_configs[model_type]
        
        # Extract parameter ranges
        lora_r_values = list(set(sweep["lora"]["r"] for sweep in sweeps))
        lora_alpha_values = list(set(sweep["lora"]["alpha"] for sweep in sweeps))
        lora_dropout_values = list(set(sweep["lora"]["dropout"] for sweep in sweeps))
        learning_rate_values = list(set(sweep["training"]["learning_rate"] for sweep in sweeps))
        
        wandb_config = {
            "method": "grid",
            "name": f"{model_type}_hyperparameter_sweep",
            "metric": {
                "name": "eval/macro_f1" if model_type == "mapper" else "eval/rouge_l",
                "goal": "maximize"
            },
            "parameters": {
                "lora_r": {
                    "values": lora_r_values
                },
                "lora_alpha": {
                    "values": lora_alpha_values
                },
                "lora_dropout": {
                    "values": lora_dropout_values
                },
                "learning_rate": {
                    "values": learning_rate_values
                },
                "model_type": {
                    "value": model_type
                }
            }
        }
        
        return wandb_config
    
    def run_sweep(self, model_type: str, sweep_count: int = 10):
        """Run a hyperparameter sweep."""
        if model_type not in self.sweep_configs:
            raise ValueError(f"No sweeps loaded for model type: {model_type}")
        
        print(f"🚀 Running {model_type} hyperparameter sweep...")
        print(f"  Total configurations: {len(self.sweep_configs[model_type])}")
        print(f"  Sweep count: {sweep_count}")
        
        # This would integrate with actual training pipeline
        # For now, just print the configuration
        wandb_config = self.create_wandb_sweep_config(model_type)
        print(f"  W&B config: {json.dumps(wandb_config, indent=2)}")
        
        return wandb_config

def main():
    """Main function to generate and save hyperparameter sweeps."""
    print("🔥 Generating hyperparameter sweeps for Compliance AI models...")
    
    # Generate sweeps
    generator = HyperparameterSweepGenerator()
    all_sweeps = generator.generate_all_sweeps()
    
    # Save sweeps
    generator.save_sweeps()
    
    # Create sweep runner
    runner = SweepRunner()
    runner.load_sweeps()
    
    # Create W&B configurations
    mapper_wandb_config = runner.create_wandb_sweep_config("mapper")
    analyst_wandb_config = runner.create_wandb_sweep_config("analyst")
    
    print(f"\n🎉 Hyperparameter sweeps generated successfully!")
    print(f"  Total sweeps: {all_sweeps['total_sweeps']}")
    print(f"  Mapper sweeps: {len(all_sweeps['mapper_sweeps'])}")
    print(f"  Analyst sweeps: {len(all_sweeps['analyst_sweeps'])}")
    print(f"  W&B integration: Ready")
    
    # Print sample configurations
    print(f"\n📋 Sample Mapper Configuration:")
    sample_mapper = all_sweeps["mapper_sweeps"][0]
    print(f"  {sample_mapper.description}")
    
    print(f"\n📋 Sample Analyst Configuration:")
    sample_analyst = all_sweeps["analyst_sweeps"][0]
    print(f"  {sample_analyst.description}")

if __name__ == "__main__":
    main()
