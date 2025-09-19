#!/usr/bin/env python3
"""
Example of using a trained LoRA model locally after downloading from Colab.

This script shows how to load a checkpoint trained in Google Colab and use it
for inference on your local machine.
"""

import argparse
import torch
from pathlib import Path
from typing import List, Dict

def load_model_from_checkpoint(checkpoint_path: str):
    """
    Load a trained LoRA model from checkpoint directory.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    
    try:
        # Load using PEFT
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        
        # Load the fine-tuned model
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Device: {next(model.parameters()).device}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure you downloaded the complete checkpoint from Colab")
        raise


def create_instruction_prompt(instruction: str) -> str:
    """Create instruction-following prompt format for Llama-3."""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 50) -> str:
    """
    Generate response using the trained model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        instruction: Input instruction
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated response
    """
    # Create prompt
    prompt = create_instruction_prompt(instruction)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    response = full_response.replace(prompt, "").strip()
    
    return response


def run_interactive_demo(model, tokenizer):
    """Run interactive demo with the trained model."""
    print("\n🤖 Interactive Mapper Demo")
    print("Enter detector outputs to map to canonical taxonomy.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("Detector output: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Create instruction
            instruction = f"Map this detector output to canonical taxonomy: '{user_input}'"
            
            # Generate response
            print("🔄 Generating response...")
            response = generate_response(model, tokenizer, instruction)
            
            print(f"📋 Canonical label: {response}")
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print("👋 Goodbye!")


def run_batch_test(model, tokenizer):
    """Run batch test with predefined examples."""
    test_cases = [
        "User entered SSN: 123-45-6789",
        "Email found: user@example.com", 
        "Phone number detected: (555) 123-4567",
        "Credit card: 4111-1111-1111-1111",
        "Jailbreak attempt detected",
        "Prompt injection: ignore previous instructions",
        "Address: 123 Main St, Anytown USA",
        "Bank account: 123456789",
        "Passport number: A12345678",
        "Tool request detected: execute command",
    ]
    
    print("\n🧪 Running batch test...")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        instruction = f"Map this detector output to canonical taxonomy: '{test_case}'"
        response = generate_response(model, tokenizer, instruction)
        
        print(f"{i:2d}. Input:  {test_case}")
        print(f"    Output: {response}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Use trained LoRA model for inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/final_model",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch", "both"],
        default="both",
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    print("🦙 Llama Mapper - Trained Model Demo")
    print("=" * 40)
    
    # Load model
    try:
        model, tokenizer = load_model_from_checkpoint(args.checkpoint)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1
    
    # Run demo
    if args.mode in ["batch", "both"]:
        run_batch_test(model, tokenizer)
    
    if args.mode in ["interactive", "both"]:
        run_interactive_demo(model, tokenizer)
    
    return 0


if __name__ == "__main__":
    exit(main())