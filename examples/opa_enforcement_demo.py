#!/usr/bin/env python3
"""
OPA Enforcement Demo
Demonstrates the analysis → automated enforcement loop.
"""

import json
import tempfile
import subprocess
from pathlib import Path
import requests
import time

class OPAEnforcementDemo:
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url
        
    def demo_enforcement_loop(self):
        """Demonstrate the complete analysis → enforcement loop."""
        print("🔒 OPA ENFORCEMENT DEMO")
        print("=" * 50)
        
        # Step 1: Analyze metrics and get OPA policy
        print("\n1. Analyzing compliance metrics...")
        analysis_result = self._get_analysis_with_policy()
        
        if not analysis_result:
            print("❌ Failed to get analysis result")
            return False
            
        print(f"   Analysis: {analysis_result['reason']}")
        print(f"   Remediation: {analysis_result['remediation']}")
        
        # Step 2: Extract and save OPA policy
        print("\n2. Extracting OPA policy...")
        opa_policy = analysis_result['opa_diff']
        policy_file = self._save_opa_policy(opa_policy)
        
        if not policy_file:
            print("❌ Failed to save OPA policy")
            return False
            
        print(f"   Policy saved to: {policy_file}")
        
        # Step 3: Test policy enforcement
        print("\n3. Testing policy enforcement...")
        
        # Test cases: compliant vs non-compliant requests
        test_cases = [
            {
                "name": "Compliant Request",
                "input": {
                    "detectors": ["toxicity", "regex-pii"],
                    "observed_coverage": {"toxicity": 0.98, "regex-pii": 0.97},
                    "required_coverage": {"toxicity": 0.95, "regex-pii": 0.95}
                },
                "should_pass": True
            },
            {
                "name": "Non-Compliant Request (Coverage Gap)",
                "input": {
                    "detectors": ["toxicity", "regex-pii"],
                    "observed_coverage": {"toxicity": 0.85, "regex-pii": 0.88},
                    "required_coverage": {"toxicity": 0.95, "regex-pii": 0.95}
                },
                "should_pass": False
            },
            {
                "name": "Missing Detector",
                "input": {
                    "detectors": ["toxicity", "regex-pii"],
                    "observed_coverage": {"toxicity": 0.98},  # Missing regex-pii
                    "required_coverage": {"toxicity": 0.95, "regex-pii": 0.95}
                },
                "should_pass": False
            }
        ]
        
        enforcement_results = []
        
        for test_case in test_cases:
            print(f"\n   Testing: {test_case['name']}")
            
            # Evaluate policy against test input
            violations = self._evaluate_opa_policy(policy_file, test_case['input'])
            
            has_violations = len(violations) > 0
            should_block = not test_case['should_pass']
            
            if has_violations == should_block:
                print(f"   ✅ CORRECT: {'Blocked' if has_violations else 'Allowed'}")
                enforcement_results.append(True)
            else:
                print(f"   ❌ INCORRECT: Expected {'block' if should_block else 'allow'}, got {'block' if has_violations else 'allow'}")
                enforcement_results.append(False)
            
            if violations:
                print(f"      Violations: {violations}")
        
        # Step 4: Demonstrate route blocking
        print("\n4. Simulating route enforcement...")
        self._simulate_route_enforcement(policy_file)
        
        # Clean up
        Path(policy_file).unlink(missing_ok=True)
        
        success_rate = sum(enforcement_results) / len(enforcement_results)
        print(f"\n📊 Enforcement Test Results: {sum(enforcement_results)}/{len(enforcement_results)} passed")
        
        if success_rate == 1.0:
            print("🎉 ENFORCEMENT DEMO SUCCESS!")
            print("\nKey Achievements:")
            print("  • Analysis identified compliance gaps")
            print("  • Generated enforceable OPA policy")
            print("  • Policy correctly blocks non-compliant requests")
            print("  • Automated enforcement prevents violations")
            return True
        else:
            print("⚠️  Some enforcement tests failed")
            return False
    
    def _get_analysis_with_policy(self) -> dict | None:
        """Get analysis result with OPA policy."""
        sample_request = {
            "request": {
                "period": "2024-01-15T10:30:00Z/2024-01-15T11:30:00Z",
                "tenant": "enforcement-demo",
                "app": "coverage-gap-app",
                "route": "/api/critical",
                "required_detectors": ["toxicity", "regex-pii"],
                "observed_coverage": {"toxicity": 0.85, "regex-pii": 0.88},
                "required_coverage": {"toxicity": 0.95, "regex-pii": 0.95},
                "detector_errors": {},
                "high_sev_hits": [],
                "false_positive_bands": [],
                "policy_bundle": "enforcement-policy-1.0",
                "env": "prod"
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/analysis/analyze",
                json=sample_request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Request Error: {str(e)}")
            return None
    
    def _save_opa_policy(self, policy_content: str) -> str | None:
        """Save OPA policy to temp file."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
                f.write(policy_content)
                return f.name
        except Exception as e:
            print(f"Error saving policy: {str(e)}")
            return None
    
    def _evaluate_opa_policy(self, policy_file: str, input_data: dict) -> list:
        """Evaluate OPA policy against input data."""
        try:
            # Save input to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_data, f)
                input_file = f.name
            
            # Evaluate policy
            result = subprocess.run(
                ['opa', 'eval', '-d', policy_file, '-i', input_file, 'data.coverage.violation'],
                capture_output=True, text=True, timeout=10
            )
            
            # Clean up input file
            Path(input_file).unlink(missing_ok=True)
            
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout)
                    violations = output.get('result', [])
                    return violations if isinstance(violations, list) else []
                except json.JSONDecodeError:
                    return []
            else:
                print(f"OPA eval error: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"Policy evaluation error: {str(e)}")
            return []
    
    def _simulate_route_enforcement(self, policy_file: str):
        """Simulate route-level enforcement."""
        print("   Simulating API gateway enforcement...")
        
        # Simulate incoming requests to different routes
        simulated_requests = [
            {
                "route": "/api/user/profile",
                "coverage": {"toxicity": 0.98, "regex-pii": 0.97},
                "expected": "ALLOW"
            },
            {
                "route": "/api/payment/process",
                "coverage": {"toxicity": 0.85, "regex-pii": 0.88},
                "expected": "BLOCK"
            },
            {
                "route": "/api/admin/users",
                "coverage": {"toxicity": 0.96},  # Missing regex-pii
                "expected": "BLOCK"
            }
        ]
        
        for req in simulated_requests:
            input_data = {
                "detectors": ["toxicity", "regex-pii"],
                "observed_coverage": req["coverage"],
                "required_coverage": {"toxicity": 0.95, "regex-pii": 0.95}
            }
            
            violations = self._evaluate_opa_policy(policy_file, input_data)
            action = "BLOCK" if violations else "ALLOW"
            
            status = "✅" if action == req["expected"] else "❌"
            print(f"   {status} {req['route']}: {action}")
            
            if violations and action == "BLOCK":
                print(f"      Reason: Coverage insufficient")
                print(f"      Required: toxicity≥0.95, regex-pii≥0.95")
                print(f"      Observed: {req['coverage']}")

def main():
    """Run the OPA enforcement demo."""
    # Check if OPA is available
    try:
        result = subprocess.run(['opa', 'version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ OPA not installed. Please install OPA to run this demo.")
            print("   Installation: https://www.openpolicyagent.org/docs/latest/#running-opa")
            return
    except FileNotFoundError:
        print("❌ OPA not found in PATH. Please install OPA to run this demo.")
        return
    
    demo = OPAEnforcementDemo()
    demo.demo_enforcement_loop()

if __name__ == "__main__":
    main()
