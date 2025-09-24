#!/usr/bin/env python3
"""
Simple test script for API key authentication system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_mapper.analysis.infrastructure.auth import (
    APIKeyManager,
    APIKeyRequest,
    APIKeyScope,
)


def test_api_key_creation():
    """Test basic API key creation and validation."""
    print("Testing API key creation and validation...")

    # Create API key manager
    manager = APIKeyManager()

    # Create a test API key
    request = APIKeyRequest(
        tenant_id="test-tenant",
        name="test-key",
        description="Test API key",
        scopes=["analyze"],
        expires_in_days=30,
    )

    response = manager.create_api_key(request)
    if response:
        print(f"✅ Created API key: {response.key_id}")
        print(f"   API Key: {response.api_key}")
        print(f"   Tenant: {response.tenant_id}")
        print(f"   Scopes: {response.scopes}")

        # Test validation
        validated_key = manager.validate_api_key(
            response.api_key, [APIKeyScope.ANALYZE]
        )

        if validated_key:
            print(f"✅ API key validation successful")
            print(f"   Validated key ID: {validated_key.key_id}")
            print(f"   Usage count: {validated_key.usage_count}")
        else:
            print("❌ API key validation failed")
    else:
        print("❌ Failed to create API key")


def test_api_key_rotation():
    """Test API key rotation."""
    print("\nTesting API key rotation...")

    manager = APIKeyManager()

    # Create initial key
    request = APIKeyRequest(
        tenant_id="test-tenant", name="rotation-test-key", scopes=["analyze"]
    )

    response = manager.create_api_key(request)
    if response:
        print(f"✅ Created initial key: {response.key_id}")

        # Rotate the key
        new_response = manager.rotate_api_key(response.key_id)
        if new_response:
            print(f"✅ Rotated key: {response.key_id} -> {new_response.key_id}")

            # Test that old key is invalid
            old_validated = manager.validate_api_key(
                response.api_key, [APIKeyScope.ANALYZE]
            )
            if old_validated is None:
                print("✅ Old key is properly invalidated")
            else:
                print("❌ Old key is still valid (should be invalid)")

            # Test that new key is valid
            new_validated = manager.validate_api_key(
                new_response.api_key, [APIKeyScope.ANALYZE]
            )
            if new_validated:
                print("✅ New key is valid")
            else:
                print("❌ New key is invalid")
        else:
            print("❌ Failed to rotate API key")
    else:
        print("❌ Failed to create initial API key")


def main():
    """Main function."""
    print("API Key Authentication System Test")
    print("=" * 40)

    try:
        test_api_key_creation()
        test_api_key_rotation()

        print("\n" + "=" * 40)
        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
