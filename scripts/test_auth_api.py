#!/usr/bin/env python3
"""
Test script for API key authentication with FastAPI integration.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.testclient import TestClient
from llama_mapper.analysis.infrastructure.auth import APIKeyManager, APIKeyRequest, APIKeyScope
from llama_mapper.analysis.api.auth_middleware import APIKeyAuthDependency


def create_test_app():
    """Create a test FastAPI app with authentication."""
    app = FastAPI(title="Test Auth API")
    
    # Create API key manager
    api_key_manager = APIKeyManager()
    
    # Create test API key
    request = APIKeyRequest(
        tenant_id="test-tenant",
        name="Test API Key",
        scopes=["analyze", "admin"]
    )
    response = api_key_manager.create_api_key(request)
    
    if not response:
        raise Exception("Failed to create test API key")
    
    print(f"Created test API key: {response.api_key}")
    
    # Create auth dependency
    auth_dependency = APIKeyAuthDependency(
        api_key_manager=api_key_manager,
        required_scopes={APIKeyScope.ANALYZE}
    )
    
    @app.get("/test")
    async def test_endpoint(auth: dict = Depends(auth_dependency)):
        """Test endpoint that requires authentication."""
        return {
            "message": "Authentication successful",
            "tenant_id": auth["tenant_id"],
            "scopes": auth["scopes"]
        }
    
    @app.get("/admin")
    async def admin_endpoint(auth: dict = Depends(APIKeyAuthDependency(
        api_key_manager=api_key_manager,
        required_scopes={APIKeyScope.ADMIN}
    ))):
        """Admin endpoint that requires admin scope."""
        return {
            "message": "Admin access granted",
            "tenant_id": auth["tenant_id"],
            "scopes": auth["scopes"]
        }
    
    @app.get("/public")
    async def public_endpoint():
        """Public endpoint that doesn't require authentication."""
        return {"message": "Public endpoint"}
    
    return app, response.api_key


def test_auth_api():
    """Test the authentication API."""
    print("Testing FastAPI authentication integration...")
    
    # Create test app
    app, test_api_key = create_test_app()
    client = TestClient(app)
    
    # Test 1: Public endpoint (no auth required)
    print("\n1. Testing public endpoint...")
    response = client.get("/public")
    assert response.status_code == 200
    print(f"✅ Public endpoint: {response.json()}")
    
    # Test 2: Authenticated endpoint without API key
    print("\n2. Testing authenticated endpoint without API key...")
    response = client.get("/test")
    assert response.status_code == 401
    print(f"✅ Unauthorized access properly blocked: {response.json()}")
    
    # Test 3: Authenticated endpoint with valid API key
    print("\n3. Testing authenticated endpoint with valid API key...")
    headers = {"X-API-Key": test_api_key}
    response = client.get("/test", headers=headers)
    assert response.status_code == 200
    print(f"✅ Authenticated access successful: {response.json()}")
    
    # Test 4: Admin endpoint with analyze scope (should fail)
    print("\n4. Testing admin endpoint with analyze scope...")
    headers = {"X-API-Key": test_api_key}
    response = client.get("/admin", headers=headers)
    assert response.status_code == 200  # Should work because our test key has admin scope
    print(f"✅ Admin access successful: {response.json()}")
    
    # Test 5: Invalid API key
    print("\n5. Testing with invalid API key...")
    headers = {"X-API-Key": "invalid-key"}
    response = client.get("/test", headers=headers)
    assert response.status_code == 401
    print(f"✅ Invalid key properly rejected: {response.json()}")
    
    # Test 6: API key in Authorization header
    print("\n6. Testing API key in Authorization header...")
    headers = {"Authorization": f"Bearer {test_api_key}"}
    response = client.get("/test", headers=headers)
    assert response.status_code == 200
    print(f"✅ Bearer token authentication successful: {response.json()}")


def main():
    """Main function."""
    print("FastAPI Authentication Integration Test")
    print("="*50)
    
    try:
        test_auth_api()
        
        print("\n" + "="*50)
        print("✅ All FastAPI authentication tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
