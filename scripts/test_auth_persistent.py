#!/usr/bin/env python3
"""
Test script for persistent API key authentication system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_mapper.analysis.infrastructure.auth import APIKeyManager, APIKeyRequest, APIKeyScope


def test_persistent_auth():
    """Test authentication with persistent storage."""
    print("Testing persistent API key authentication...")
    
    # Create API key manager (same instance throughout test)
    manager = APIKeyManager()
    
    # Create a test API key
    print("1. Creating API key...")
    request = APIKeyRequest(
        tenant_id="test-tenant",
        name="Persistent Test Key",
        description="Key for testing persistent authentication",
        scopes=["analyze", "admin"],
        expires_in_days=7
    )
    
    response = manager.create_api_key(request)
    if response:
        print(f"✅ Created API key: {response.key_id}")
        print(f"   API Key: {response.api_key}")
        print(f"   Tenant: {response.tenant_id}")
        print(f"   Scopes: {response.scopes}")
        
        # Test validation with same manager instance
        print("\n2. Testing validation...")
        validated_key = manager.validate_api_key(
            response.api_key,
            [APIKeyScope.ANALYZE]
        )
        
        if validated_key:
            print(f"✅ Validation successful")
            print(f"   Usage count: {validated_key.usage_count}")
            print(f"   Last used: {validated_key.last_used_at}")
        else:
            print(f"❌ Validation failed")
        
        # Test admin scope validation
        print("\n3. Testing admin scope...")
        admin_validated = manager.validate_api_key(
            response.api_key,
            [APIKeyScope.ADMIN]
        )
        
        if admin_validated:
            print(f"✅ Admin scope validation successful")
            print(f"   Usage count: {admin_validated.usage_count}")
        else:
            print(f"❌ Admin scope validation failed")
        
        # Test rate limiting
        print("\n4. Testing rate limiting...")
        for i in range(5):
            rate_limited_key = manager.validate_api_key(
                response.api_key,
                [APIKeyScope.ANALYZE]
            )
            if rate_limited_key:
                print(f"   Request {i+1}: ✅ (usage count: {rate_limited_key.usage_count})")
            else:
                print(f"   Request {i+1}: ❌")
        
        # Test key rotation
        print("\n5. Testing key rotation...")
        new_response = manager.rotate_api_key(response.key_id)
        
        if new_response:
            print(f"✅ Rotation successful")
            print(f"   Old key ID: {response.key_id}")
            print(f"   New key ID: {new_response.key_id}")
            print(f"   New API key: {new_response.api_key}")
            
            # Test that old key is invalid
            old_validated = manager.validate_api_key(
                response.api_key,
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
        
        # Test key revocation
        print("\n6. Testing key revocation...")
        success = manager.revoke_api_key(new_response.key_id)
        
        if success:
            print(f"✅ Revocation successful")
            
            # Test that revoked key is invalid
            revoked_validated = manager.validate_api_key(
                new_response.api_key,
                [APIKeyScope.ANALYZE]
            )
            if revoked_validated is None:
                print(f"✅ Revoked key properly invalidated")
            else:
                print(f"❌ Revoked key still valid")
        else:
            print(f"❌ Revocation failed")
        
        # Test tenant key listing
        print("\n7. Testing tenant key listing...")
        tenant_keys = manager.list_tenant_keys("test-tenant")
        print(f"✅ Found {len(tenant_keys)} keys for tenant")
        for key in tenant_keys:
            print(f"   - {key.name} ({key.key_id}) - Status: {key.status.value}")
        
    else:
        print(f"❌ Failed to create API key")


def main():
    """Main function."""
    print("Persistent API Key Authentication Test")
    print("="*50)
    
    try:
        test_persistent_auth()
        
        print("\n" + "="*50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
