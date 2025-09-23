"""
API key management endpoints for the Analysis Module.

This module provides REST API endpoints for creating, managing, and rotating
API keys for the analysis module.
"""

import logging
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from ..infrastructure.auth import (
    APIKeyManager, 
    APIKeyRequest, 
    APIKeyResponse, 
    APIKeyScope,
    APIKeyStatus
)
from .auth_middleware import APIKeyAuthDependency


logger = logging.getLogger(__name__)


class APIKeyListResponse(BaseModel):
    """Response model for listing API keys."""
    keys: List[APIKeyResponse]
    total: int


class APIKeyRotateRequest(BaseModel):
    """Request model for rotating an API key."""
    key_id: str = Field(..., description="API key ID to rotate")


class APIKeyRevokeRequest(BaseModel):
    """Request model for revoking an API key."""
    key_id: str = Field(..., description="API key ID to revoke")


class APIKeyUpdateRequest(BaseModel):
    """Request model for updating an API key."""
    name: Optional[str] = Field(None, description="New name for the API key")
    description: Optional[str] = Field(None, description="New description for the API key")
    scopes: Optional[List[str]] = Field(None, description="New scopes for the API key")
    rate_limit: Optional[int] = Field(None, description="New rate limit for the API key")


class AuthEndpoints:
    """
    API key management endpoints.
    
    Provides REST API endpoints for API key lifecycle management.
    """
    
    def __init__(self, api_key_manager: APIKeyManager):
        """
        Initialize the auth endpoints.
        
        Args:
            api_key_manager: API key manager instance
        """
        self.api_key_manager = api_key_manager
        self.router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
        self._register_endpoints()
    
    def _register_endpoints(self):
        """Register all authentication endpoints."""
        
        @self.router.post("/keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
        async def create_api_key(
            request: APIKeyRequest,
            auth: dict = Depends(APIKeyAuthDependency(
                self.api_key_manager, 
                {APIKeyScope.ADMIN}
            ))
        ):
            """
            Create a new API key.
            
            Args:
                request: API key creation request
                auth: Authentication info (requires admin scope)
                
            Returns:
                Created API key response
            """
            try:
                # Validate scopes
                if not self._validate_scopes(request.scopes):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid scopes provided"
                    )
                
                # Create API key
                response = self.api_key_manager.create_api_key(request)
                if not response:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to create API key"
                    )
                
                logger.info(f"Created API key {response.key_id} for tenant {request.tenant_id}")
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error creating API key: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.router.get("/keys", response_model=APIKeyListResponse)
        async def list_api_keys(
            tenant_id: Optional[str] = None,
            auth: dict = Depends(APIKeyAuthDependency(
                self.api_key_manager, 
                {APIKeyScope.ADMIN}
            ))
        ):
            """
            List API keys for a tenant.
            
            Args:
                tenant_id: Tenant ID to list keys for (defaults to authenticated tenant)
                auth: Authentication info (requires admin scope)
                
            Returns:
                List of API keys
            """
            try:
                # Use authenticated tenant if not specified
                if not tenant_id:
                    tenant_id = auth["tenant_id"]
                
                # List API keys
                keys = self.api_key_manager.list_tenant_keys(tenant_id)
                
                # Convert to response format
                key_responses = []
                for key in keys:
                    key_responses.append(APIKeyResponse(
                        key_id=key.key_id,
                        api_key="***REDACTED***",  # Never return actual keys
                        tenant_id=key.tenant_id,
                        name=key.name,
                        description=key.description,
                        scopes=[scope.value for scope in key.scopes],
                        status=key.status.value,
                        created_at=key.created_at.isoformat(),
                        expires_at=key.expires_at.isoformat() if key.expires_at else None,
                        rate_limit=key.rate_limit
                    ))
                
                return APIKeyListResponse(keys=key_responses, total=len(key_responses))
                
            except Exception as e:
                logger.error(f"Error listing API keys: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.router.post("/keys/rotate", response_model=APIKeyResponse)
        async def rotate_api_key(
            request: APIKeyRotateRequest,
            auth: dict = Depends(APIKeyAuthDependency(
                self.api_key_manager, 
                {APIKeyScope.ADMIN}
            ))
        ):
            """
            Rotate an API key.
            
            Args:
                request: API key rotation request
                auth: Authentication info (requires admin scope)
                
            Returns:
                New API key response
            """
            try:
                # Rotate API key
                response = self.api_key_manager.rotate_api_key(request.key_id)
                if not response:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="API key not found or rotation failed"
                    )
                
                logger.info(f"Rotated API key {request.key_id} -> {response.key_id}")
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error rotating API key: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.router.post("/keys/revoke", status_code=status.HTTP_204_NO_CONTENT)
        async def revoke_api_key(
            request: APIKeyRevokeRequest,
            auth: dict = Depends(APIKeyAuthDependency(
                self.api_key_manager, 
                {APIKeyScope.ADMIN}
            ))
        ):
            """
            Revoke an API key.
            
            Args:
                request: API key revocation request
                auth: Authentication info (requires admin scope)
                
            Returns:
                No content on success
            """
            try:
                # Revoke API key
                success = self.api_key_manager.revoke_api_key(request.key_id)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="API key not found or revocation failed"
                    )
                
                logger.info(f"Revoked API key {request.key_id}")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error revoking API key: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.router.get("/keys/cleanup", response_model=dict)
        async def cleanup_expired_keys(
            auth: dict = Depends(APIKeyAuthDependency(
                self.api_key_manager, 
                {APIKeyScope.ADMIN}
            ))
        ):
            """
            Clean up expired API keys.
            
            Args:
                auth: Authentication info (requires admin scope)
                
            Returns:
                Cleanup results
            """
            try:
                # Clean up expired keys
                cleaned_count = self.api_key_manager.cleanup_expired_keys()
                
                return {
                    "message": f"Cleaned up {cleaned_count} expired API keys",
                    "cleaned_count": cleaned_count,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error cleaning up expired keys: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.router.get("/scopes", response_model=List[str])
        async def list_available_scopes():
            """
            List available API key scopes.
            
            Returns:
                List of available scopes
            """
            return [scope.value for scope in APIKeyScope]
    
    def _validate_scopes(self, scopes: List[str]) -> bool:
        """
        Validate that all provided scopes are valid.
        
        Args:
            scopes: List of scope strings
            
        Returns:
            True if all scopes are valid, False otherwise
        """
        try:
            for scope in scopes:
                APIKeyScope(scope)
            return True
        except ValueError:
            return False
