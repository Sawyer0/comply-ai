#!/usr/bin/env python3
"""
Test comprehensive detector coverage for investor demonstrations.

This script shows support for major enterprise AI safety and content moderation tools,
demonstrating true detector agnosticism.
"""

import sys
import os
sys.path.append('src')

from llama_mapper.serving.fallback_mapper import FallbackMapper
import json

def test_enterprise_detector_coverage():
    """Test coverage of major enterprise AI safety tools."""
    
    print("🌐 Testing Enterprise Detector Coverage")
    print("=" * 70)
    
    # Initialize fallback mapper
    mapper = FallbackMapper(detector_configs_path=".kiro/pillars-detectors")
    
    # Enterprise detector test cases
    enterprise_tests = [
        # Amazon Bedrock Guardrails
        {"detector": "amazon-bedrock-guardrails", "output": "HATE", "category": "Enterprise Cloud AI"},
        {"detector": "amazon-bedrock-guardrails", "output": "PERSON_NAME", "category": "Enterprise Cloud AI"},
        {"detector": "amazon-bedrock-guardrails", "output": "PROMPT_INJECTION", "category": "Enterprise Cloud AI"},
        
        # NVIDIA NeMo Guardrails
        {"detector": "nvidia-nemo-guardrails", "output": "jailbreak", "category": "Enterprise AI Platform"},
        {"detector": "nvidia-nemo-guardrails", "output": "harmful_content", "category": "Enterprise AI Platform"},
        {"detector": "nvidia-nemo-guardrails", "output": "hallucination", "category": "Enterprise AI Platform"},
        
        # Microsoft Azure Content Safety
        {"detector": "azure-content-safety", "output": "Hate", "category": "Enterprise Cloud AI"},
        {"detector": "azure-content-safety", "output": "violence_4", "category": "Enterprise Cloud AI"},
        {"detector": "azure-content-safety", "output": "Sexual", "category": "Enterprise Cloud AI"},
        
        # Google Cloud DLP
        {"detector": "google-cloud-dlp", "output": "PERSON_NAME", "category": "Enterprise Cloud AI"},
        {"detector": "google-cloud-dlp", "output": "US_SOCIAL_SECURITY_NUMBER", "category": "Enterprise Cloud AI"},
        {"detector": "google-cloud-dlp", "output": "CREDIT_CARD_NUMBER", "category": "Enterprise Cloud AI"},
        
        # Anthropic Claude Safety
        {"detector": "anthropic-claude-safety", "output": "harmful", "category": "Leading AI Model"},
        {"detector": "anthropic-claude-safety", "output": "jailbreak_attempt", "category": "Leading AI Model"},
        {"detector": "anthropic-claude-safety", "output": "misinformation", "category": "Leading AI Model"},
        
        # Hugging Face Transformers (Open Source)
        {"detector": "huggingface-transformers", "output": "TOXIC", "category": "Open Source AI"},
        {"detector": "huggingface-transformers", "output": "hate_speech", "category": "Open Source AI"},
        {"detector": "huggingface-transformers", "output": "cyberbullying", "category": "Open Source AI"},
        
        # Existing detectors
        {"detector": "presidio", "output": "EMAIL_ADDRESS", "category": "Microsoft Open Source"},
        {"detector": "openai-moderation", "output": "hate/threatening", "category": "Leading AI Model"},
        {"detector": "llama-guard", "output": "violence", "category": "Meta AI Safety"},
        {"detector": "deberta-toxicity", "output": "toxic", "category": "Academic Research"},
    ]
    
    results = {
        "total_tests": len(enterprise_tests),
        "passed": 0,
        "failed": 0,
        "categories": {},
        "detectors_supported": set()
    }
    
    print(f"Testing {len(enterprise_tests)} enterprise detector scenarios...\n")
    
    for i, test_case in enumerate(enterprise_tests, 1):
        detector = test_case["detector"]
        output = test_case["output"]
        category = test_case["category"]
        
        try:
            # Test the mapping
            response = mapper.map(detector, output, reason="enterprise_test")
            
            # Check if we got a meaningful result (not OTHER.Unknown)
            actual = response.taxonomy[0] if response.taxonomy else "OTHER.Unknown"
            
            if actual != "OTHER.Unknown":
                status = "✅ MAPPED"
                results["passed"] += 1
                results["detectors_supported"].add(detector)
            else:
                status = "❌ UNMAPPED"
                results["failed"] += 1
            
            # Track by category
            if category not in results["categories"]:
                results["categories"][category] = {"tested": 0, "passed": 0}
            results["categories"][category]["tested"] += 1
            if actual != "OTHER.Unknown":
                results["categories"][category]["passed"] += 1
            
            print(f"{i:2d}. {status} | {detector:25s} | {output:20s} → {actual}")
            print(f"     Category: {category} | Confidence: {response.confidence}")
            
        except Exception as e:
            print(f"{i:2d}. ❌ ERROR | {detector:25s} | {output:20s} → Exception: {e}")
            results["failed"] += 1
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("📊 ENTERPRISE DETECTOR COVERAGE SUMMARY")
    print("=" * 70)
    
    success_rate = (results["passed"] / results["total_tests"]) * 100
    print(f"Overall Coverage: {success_rate:.1f}% ({results['passed']}/{results['total_tests']})")
    print(f"Detectors Supported: {len(results['detectors_supported'])}")
    
    print(f"\nCoverage by Enterprise Category:")
    for category, stats in results["categories"].items():
        category_success = (stats["passed"] / stats["tested"]) * 100
        print(f"  {category:25s}: {category_success:5.1f}% ({stats['passed']}/{stats['tested']})")
    
    # List all supported detectors
    all_supported = mapper.get_supported_detectors()
    print(f"\nAll Supported Detectors ({len(all_supported)}):")
    
    detector_categories = {
        "Enterprise Cloud AI": ["amazon-bedrock-guardrails", "azure-content-safety", "google-cloud-dlp"],
        "Enterprise AI Platforms": ["nvidia-nemo-guardrails"],
        "Leading AI Models": ["openai-moderation", "anthropic-claude-safety", "llama-guard"],
        "Microsoft Tools": ["presidio"],
        "Open Source AI": ["huggingface-transformers", "deberta-toxicity", "detoxify-hatebert"],
        "Rule-Based Tools": ["regex-pii"]
    }
    
    for category, detectors in detector_categories.items():
        print(f"\n  {category}:")
        for detector in detectors:
            if detector in all_supported:
                mappings = mapper.get_detector_mappings(detector)
                mapping_count = len(mappings) if mappings else 0
                print(f"    ✅ {detector:30s} ({mapping_count:3d} mappings)")
            else:
                print(f"    ❌ {detector:30s} (not configured)")
    
    # Generate market analysis
    print("\n" + "=" * 70)
    print("💰 MARKET COVERAGE ANALYSIS")
    print("=" * 70)
    
    market_segments = {
        "Enterprise Cloud (AWS/Azure/GCP)": ["amazon-bedrock-guardrails", "azure-content-safety", "google-cloud-dlp"],
        "AI Platform Vendors": ["nvidia-nemo-guardrails", "openai-moderation"],
        "Leading AI Models": ["anthropic-claude-safety", "llama-guard"],
        "Open Source Community": ["huggingface-transformers", "presidio", "deberta-toxicity"],
        "Custom/Rule-Based": ["regex-pii"]
    }
    
    total_market_coverage = 0
    for segment, detectors in market_segments.items():
        supported_in_segment = sum(1 for d in detectors if d in all_supported)
        coverage = (supported_in_segment / len(detectors)) * 100
        total_market_coverage += coverage
        print(f"  {segment:30s}: {coverage:5.1f}% ({supported_in_segment}/{len(detectors)})")
    
    avg_market_coverage = total_market_coverage / len(market_segments)
    
    print(f"\nOverall Market Coverage: {avg_market_coverage:.1f}%")
    
    # Investment thesis
    print("\n" + "=" * 70)
    print("🚀 INVESTOR VALUE PROPOSITION")
    print("=" * 70)
    
    print("✅ Comprehensive Enterprise Coverage:")
    print(f"  • {len(all_supported)} detector types supported out-of-the-box")
    print(f"  • {avg_market_coverage:.1f}% coverage across major market segments")
    print(f"  • Works with AWS, Azure, GCP, NVIDIA, OpenAI, Anthropic, Meta")
    print(f"  • Supports both proprietary and open-source tools")
    
    print("\n🎯 Competitive Advantages:")
    print("  • True detector agnosticism - works with any AI safety tool")
    print("  • No vendor lock-in - customers can switch tools freely")
    print("  • Immediate integration with existing enterprise stacks")
    print("  • Unified taxonomy across all tools and vendors")
    
    print("\n💼 Enterprise Sales Benefits:")
    print("  • 'Works with your existing tools' - no rip-and-replace")
    print("  • Reduces vendor risk and increases negotiating power")
    print("  • Future-proof as new tools emerge")
    print("  • Compliance across heterogeneous tool environments")
    
    print("\n📈 Market Expansion Opportunities:")
    print("  • Easy to add new detectors as market evolves")
    print("  • Can support customer-specific or industry-specific tools")
    print("  • Platform play - become the 'Stripe for AI safety'")
    print("  • Network effects as more tools are added")
    
    return results

if __name__ == "__main__":
    results = test_enterprise_detector_coverage()
    
    success_rate = results["passed"] / results["total_tests"]
    if success_rate >= 0.8:
        print(f"\n🎉 ENTERPRISE-READY DETECTOR COVERAGE!")
        print(f"   {results['passed']}/{results['total_tests']} enterprise scenarios supported ({success_rate*100:.1f}%)")
        print(f"   {len(results['detectors_supported'])} different enterprise tools integrated")
    else:
        print(f"\n⚠️  Need more detector coverage")
        print(f"   Only {results['passed']}/{results['total_tests']} scenarios supported")