#!/usr/bin/env python3
"""
Test script to validate the created API keys.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_mapper.analysis.infrastructure.auth import APIKeyManager, APIKeyScope


def test_created_keys():
    """Test the API keys that were created."""
    print("Testing created API keys...")
    
    # Load the created keys
    with open("api_keys.json", "r") as f:
        keys = json.load(f)
    
    # Create API key manager
    manager = APIKeyManager()
    
    for key_data in keys:
        print(f"\nTesting key: {key_data['name']}")
        print(f"Key ID: {key_data['key_id']}")
        print(f"Tenant: {key_data['tenant_id']}")
        print(f"Scopes: {key_data['scopes']}")
        
        # Test validation
        validated_key = manager.validate_api_key(
            key_data['api_key'],
            [APIKeyScope.ANALYZE]  # Test with analyze scope
        )
        
        if validated_key:
            print(f"✅ Validation successful")
            print(f"   Usage count: {validated_key.usage_count}")
            print(f"   Last used: {validated_key.last_used_at}")
        else:
            print(f"❌ Validation failed")
        
        # Test scope validation
        if "admin" in key_data['scopes']:
            admin_validated = manager.validate_api_key(
                key_data['api_key'],
                [APIKeyScope.ADMIN]
            )
            if admin_validated:
                print(f"✅ Admin scope validation successful")
            else:
                print(f"❌ Admin scope validation failed")
        
        # Test rate limiting (make multiple requests)
        print(f"   Testing rate limiting...")
        for i in range(3):
            rate_limited_key = manager.validate_api_key(
                key_data['api_key'],
                [APIKeyScope.ANALYZE]
            )
            if rate_limited_key:
                print(f"   Request {i+1}: ✅ (usage count: {rate_limited_key.usage_count})")
            else:
                print(f"   Request {i+1}: ❌")


def test_key_rotation():
    """Test key rotation functionality."""
    print("\n" + "="*50)
    print("Testing API key rotation...")
    
    # Load the created keys
    with open("api_keys.json", "r") as f:
        keys = json.load(f)
    
    # Create API key manager
    manager = APIKeyManager()
    
    # Test rotation with the first key
    test_key = keys[0]
    print(f"Rotating key: {test_key['name']}")
    
    # Rotate the key
    new_response = manager.rotate_api_key(test_key['key_id'])
    
    if new_response:
        print(f"✅ Rotation successful")
        print(f"   Old key ID: {test_key['key_id']}")
        print(f"   New key ID: {new_response.key_id}")
        print(f"   New API key: {new_response.api_key}")
        
        # Test that old key is invalid
        old_validated = manager.validate_api_key(
            test_key['api_key'],
            [APIKeyScope.ANALYZE]
        )
        if old_validated is None:
            print(f"✅ Old key properly invalidated")
        else:
            print(f"❌ Old key still valid")
        
        # Test that new key is valid
        new_validated = manager.validate_api_key(
            new_response.api_key,
            [APIKeyScope.ANALYZE]
        )
        if new_validated:
            print(f"✅ New key is valid")
            print(f"   Usage count: {new_validated.usage_count}")
        else:
            print(f"❌ New key is invalid")
    else:
        print(f"❌ Rotation failed")


def main():
    """Main function."""
    print("API Key Validation Test")
    print("="*50)
    
    try:
        test_created_keys()
        test_key_rotation()
        
        print("\n" + "="*50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
