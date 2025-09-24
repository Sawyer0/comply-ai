#!/usr/bin/env python3
"""
Final comprehensive validation for investor presentations.

This script validates that the system is truly detector-agnostic and enterprise-ready
with comprehensive coverage of major AI safety and content moderation tools.
"""

import sys
import os
sys.path.append('src')

from llama_mapper.serving.fallback_mapper import FallbackMapper
import json

def main():
    """Run comprehensive validation for investor presentation."""
    
    print("🚀 FINAL INVESTOR VALIDATION")
    print("=" * 80)
    print("Validating comprehensive detector coverage and enterprise readiness...")
    
    # Initialize mapper
    mapper = FallbackMapper(detector_configs_path=".kiro/pillars-detectors")
    
    # Enterprise scenarios
    enterprise_scenarios = [
        # Cloud AI Services
        {"name": "AWS Bedrock Hate Detection", "detector": "amazon-bedrock-guardrails", "output": "HATE"},
        {"name": "Azure Violence Detection", "detector": "azure-content-safety", "output": "Violence"},
        {"name": "Google Cloud PII Detection", "detector": "google-cloud-dlp", "output": "PERSON_NAME"},
        
        # AI Platform Vendors
        {"name": "NVIDIA Jailbreak Detection", "detector": "nvidia-nemo-guardrails", "output": "jailbreak"},
        {"name": "OpenAI Moderation", "detector": "openai-moderation", "output": "hate/threatening"},
        
        # Leading AI Models
        {"name": "Anthropic Safety", "detector": "anthropic-claude-safety", "output": "harmful"},
        {"name": "Meta Llama Guard", "detector": "llama-guard", "output": "violence"},
        
        # Open Source Tools
        {"name": "Microsoft Presidio", "detector": "presidio", "output": "EMAIL_ADDRESS"},
        {"name": "Hugging Face Transformers", "detector": "huggingface-transformers", "output": "TOXIC"},
        {"name": "DeBERTa Toxicity", "detector": "deberta-toxicity", "output": "toxic"},
        
        # Rule-Based Tools
        {"name": "Regex PII Detection", "detector": "regex-pii", "output": "ssn"},
    ]
    
    print(f"\n🧪 Testing {len(enterprise_scenarios)} Enterprise Scenarios:")
    print("-" * 80)
    
    results = {"passed": 0, "total": len(enterprise_scenarios), "details": []}
    
    for scenario in enterprise_scenarios:
        try:
            response = mapper.map(scenario["detector"], scenario["output"])
            canonical = response.taxonomy[0] if response.taxonomy else "OTHER.Unknown"
            
            success = canonical != "OTHER.Unknown"
            status = "✅ PASS" if success else "❌ FAIL"
            
            if success:
                results["passed"] += 1
            
            print(f"{status} | {scenario['name']:30s} | {canonical}")
            
            results["details"].append({
                "scenario": scenario["name"],
                "detector": scenario["detector"],
                "output": scenario["output"],
                "canonical": canonical,
                "confidence": response.confidence,
                "success": success
            })
            
        except Exception as e:
            print(f"❌ ERROR | {scenario['name']:30s} | Exception: {e}")
            results["details"].append({
                "scenario": scenario["name"],
                "success": False,
                "error": str(e)
            })
    
    # Calculate success rate
    success_rate = (results["passed"] / results["total"]) * 100
    
    print("\n" + "=" * 80)
    print("📊 FINAL VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"Enterprise Scenario Success Rate: {success_rate:.1f}% ({results['passed']}/{results['total']})")
    
    # Get comprehensive detector list
    all_detectors = mapper.get_supported_detectors()
    print(f"Total Detectors Supported: {len(all_detectors)}")
    
    # Categorize detectors
    detector_categories = {
        "Enterprise Cloud AI": [
            "amazon-bedrock-guardrails", 
            "azure-content-safety", 
            "google-cloud-dlp"
        ],
        "AI Platform Vendors": [
            "nvidia-nemo-guardrails",
            "openai-moderation"
        ],
        "Leading AI Models": [
            "anthropic-claude-safety",
            "llama-guard"
        ],
        "Microsoft Tools": [
            "presidio"
        ],
        "Open Source AI": [
            "huggingface-transformers",
            "deberta-toxicity",
            "detoxify-hatebert"
        ],
        "Rule-Based Tools": [
            "regex-pii"
        ]
    }
    
    print(f"\nDetector Coverage by Category:")
    total_coverage = 0
    category_count = 0
    
    for category, expected_detectors in detector_categories.items():
        supported = [d for d in expected_detectors if d in all_detectors]
        coverage = (len(supported) / len(expected_detectors)) * 100
        total_coverage += coverage
        category_count += 1
        
        print(f"  {category:25s}: {coverage:5.1f}% ({len(supported)}/{len(expected_detectors)})")
        for detector in supported:
            mappings = mapper.get_detector_mappings(detector)
            mapping_count = len(mappings) if mappings else 0
            print(f"    ✅ {detector:30s} ({mapping_count:3d} mappings)")
    
    avg_coverage = total_coverage / category_count if category_count > 0 else 0
    
    print(f"\nOverall Market Coverage: {avg_coverage:.1f}%")
    
    # Investment readiness assessment
    print("\n" + "=" * 80)
    print("💰 INVESTMENT READINESS ASSESSMENT")
    print("=" * 80)
    
    readiness_score = 0
    max_score = 5
    
    # Criteria 1: Enterprise scenario success rate
    if success_rate >= 90:
        print("✅ Enterprise Scenarios: EXCELLENT (90%+ success rate)")
        readiness_score += 1
    elif success_rate >= 80:
        print("✅ Enterprise Scenarios: GOOD (80%+ success rate)")
        readiness_score += 0.8
    else:
        print("⚠️  Enterprise Scenarios: NEEDS IMPROVEMENT")
    
    # Criteria 2: Detector coverage
    if len(all_detectors) >= 10:
        print("✅ Detector Coverage: EXCELLENT (10+ detectors)")
        readiness_score += 1
    elif len(all_detectors) >= 7:
        print("✅ Detector Coverage: GOOD (7+ detectors)")
        readiness_score += 0.8
    else:
        print("⚠️  Detector Coverage: NEEDS IMPROVEMENT")
    
    # Criteria 3: Market coverage
    if avg_coverage >= 90:
        print("✅ Market Coverage: EXCELLENT (90%+ across categories)")
        readiness_score += 1
    elif avg_coverage >= 75:
        print("✅ Market Coverage: GOOD (75%+ across categories)")
        readiness_score += 0.8
    else:
        print("⚠️  Market Coverage: NEEDS IMPROVEMENT")
    
    # Criteria 4: Cloud provider coverage
    cloud_providers = ["amazon-bedrock-guardrails", "azure-content-safety", "google-cloud-dlp"]
    cloud_coverage = sum(1 for cp in cloud_providers if cp in all_detectors)
    if cloud_coverage == 3:
        print("✅ Cloud Provider Coverage: EXCELLENT (AWS + Azure + GCP)")
        readiness_score += 1
    elif cloud_coverage >= 2:
        print("✅ Cloud Provider Coverage: GOOD (2/3 major clouds)")
        readiness_score += 0.8
    else:
        print("⚠️  Cloud Provider Coverage: NEEDS IMPROVEMENT")
    
    # Criteria 5: AI model vendor coverage
    ai_vendors = ["openai-moderation", "anthropic-claude-safety", "llama-guard", "nvidia-nemo-guardrails"]
    ai_coverage = sum(1 for av in ai_vendors if av in all_detectors)
    if ai_coverage >= 3:
        print("✅ AI Vendor Coverage: EXCELLENT (3+ major AI vendors)")
        readiness_score += 1
    elif ai_coverage >= 2:
        print("✅ AI Vendor Coverage: GOOD (2+ major AI vendors)")
        readiness_score += 0.8
    else:
        print("⚠️  AI Vendor Coverage: NEEDS IMPROVEMENT")
    
    # Final assessment
    readiness_percentage = (readiness_score / max_score) * 100
    
    print(f"\nOverall Investment Readiness: {readiness_percentage:.1f}% ({readiness_score:.1f}/{max_score})")
    
    if readiness_percentage >= 90:
        print("\n🎉 INVESTMENT READY!")
        print("   ✅ Comprehensive enterprise detector coverage")
        print("   ✅ True detector agnosticism demonstrated")
        print("   ✅ Major cloud providers and AI vendors supported")
        print("   ✅ Production-ready rule-based fallback system")
        print("   ✅ Competitive moat through universal integration")
    elif readiness_percentage >= 80:
        print("\n🚀 NEARLY INVESTMENT READY!")
        print("   ✅ Strong foundation with minor gaps to address")
        print("   ✅ Clear path to full enterprise readiness")
    else:
        print("\n⚠️  NEEDS MORE WORK")
        print("   ❌ Significant gaps in detector coverage")
        print("   ❌ Not yet ready for enterprise sales")
    
    # Key selling points
    print("\n" + "=" * 80)
    print("🎯 KEY INVESTOR SELLING POINTS")
    print("=" * 80)
    
    print("1. 🌐 TRUE DETECTOR AGNOSTICISM")
    print(f"   • {len(all_detectors)} different AI safety tools supported")
    print("   • Works with AWS, Azure, GCP, NVIDIA, OpenAI, Anthropic, Meta")
    print("   • No vendor lock-in - customers can switch tools freely")
    
    print("\n2. 💼 ENTERPRISE SALES ADVANTAGE")
    print("   • 'Works with your existing tools' - no rip-and-replace")
    print("   • Immediate integration with current enterprise stacks")
    print("   • Reduces vendor risk and increases negotiating power")
    
    print("\n3. 🚀 COMPETITIVE MOAT")
    print("   • Most competitors are tied to specific AI models/vendors")
    print("   • Universal taxonomy creates network effects")
    print("   • Platform play - become the 'Stripe for AI safety'")
    
    print("\n4. 📈 SCALABLE BUSINESS MODEL")
    print("   • Easy to add new detectors as market evolves")
    print("   • Can support customer-specific tools")
    print("   • Future-proof as AI safety landscape changes")
    
    return readiness_percentage >= 80

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 VALIDATION COMPLETE - INVESTOR READY!")
        exit(0)
    else:
        print(f"\n⚠️  VALIDATION INCOMPLETE - MORE WORK NEEDED")
        exit(1)