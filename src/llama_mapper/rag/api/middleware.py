"""
Middleware for RAG API endpoints.

Provides authentication, rate limiting, and other middleware functionality.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
try:
    import jwt
except ImportError:
    jwt = None  # type: ignore
from datetime import datetime, timedelta
try:
    import redis
except ImportError:
    redis = None  # type: ignore
from collections import defaultdict, deque
import asyncio

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


class AuthenticationMiddleware:
    """Authentication middleware for RAG API."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """Initialize authentication middleware.
        
        Args:
            secret_key: JWT secret key
            algorithm: JWT algorithm
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)
    
    async def authenticate(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """Authenticate user and return user information.
        
        Args:
            credentials: HTTP authorization credentials
            
        Returns:
            User information dictionary
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # Decode JWT token
            payload = jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Check token expiration
            if "exp" in payload:
                exp = datetime.fromtimestamp(payload["exp"])
                if exp < datetime.utcnow():
                    raise HTTPException(status_code=401, detail="Token expired")
            
            # Extract user information
            user_info = {
                "user_id": payload.get("user_id"),
                "tenant_id": payload.get("tenant_id"),
                "role": payload.get("role"),
                "permissions": payload.get("permissions", []),
                "exp": payload.get("exp")
            }
            
            return user_info
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    def create_token(self, user_id: str, tenant_id: str, role: str, 
                    permissions: List[str], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token for user.
        
        Args:
            user_id: User ID
            tenant_id: Tenant ID
            role: User role
            permissions: User permissions
            expires_delta: Token expiration delta
            
        Returns:
            JWT token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "role": role,
            "permissions": permissions,
            "exp": expire
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_permission(self, user_info: Dict[str, Any], required_permission: str) -> bool:
        """Verify user has required permission.
        
        Args:
            user_info: User information from authentication
            required_permission: Required permission
            
        Returns:
            True if user has permission, False otherwise
        """
        user_permissions = user_info.get("permissions", [])
        return required_permission in user_permissions or "admin" in user_permissions


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for RAG API."""
    
    def __init__(self, app, redis_url: str = "redis://localhost:6379", 
                 default_rate_limit: int = 100, window_size: int = 3600):
        """Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            redis_url: Redis URL for rate limiting
            default_rate_limit: Default rate limit per window
            window_size: Window size in seconds
        """
        super().__init__(app)
        self.redis_url = redis_url
        self.default_rate_limit = default_rate_limit
        self.window_size = window_size
        self.redis_client = None
        self.logger = logging.getLogger(__name__)
        
        # In-memory rate limiting as fallback
        self.memory_limits = defaultdict(lambda: deque())
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        try:
            # Get client identifier
            client_id = self._get_client_id(request)
            
            # Check rate limit
            if not await self._check_rate_limit(client_id, request):
                return Response(
                    content="Rate limit exceeded",
                    status_code=429,
                    headers={"Retry-After": str(self.window_size)}
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.default_rate_limit)
            response.headers["X-RateLimit-Remaining"] = str(await self._get_remaining_requests(client_id))
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.window_size)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Rate limiting error: {e}")
            # Allow request to proceed if rate limiting fails
            return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from authentication
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, options={"verify_signature": False})
                return f"user:{payload.get('user_id', 'anonymous')}"
            except:
                pass
        
        # Fall back to IP address
        client_ip = request.client.host
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(self, client_id: str, request: Request) -> bool:
        """Check if client is within rate limit."""
        try:
            # Get Redis client
            if not self.redis_client:
                self.redis_client = redis.from_url(self.redis_url)
            
            # Get current time window
            current_window = int(time.time() // self.window_size)
            key = f"rate_limit:{client_id}:{current_window}"
            
            # Check current count
            current_count = await self.redis_client.get(key)
            if current_count is None:
                current_count = 0
            else:
                current_count = int(current_count)
            
            # Check if limit exceeded
            if current_count >= self.default_rate_limit:
                return False
            
            # Increment counter
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, self.window_size)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis rate limiting failed: {e}")
            # Fall back to in-memory rate limiting
            return self._check_memory_rate_limit(client_id)
    
    def _check_memory_rate_limit(self, client_id: str) -> bool:
        """Check rate limit using in-memory storage."""
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Clean old entries
        client_requests = self.memory_limits[client_id]
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()
        
        # Check if limit exceeded
        if len(client_requests) >= self.default_rate_limit:
            return False
        
        # Add current request
        client_requests.append(current_time)
        return True
    
    async def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        try:
            if self.redis_client:
                current_window = int(time.time() // self.window_size)
                key = f"rate_limit:{client_id}:{current_window}"
                current_count = await self.redis_client.get(key)
                if current_count is None:
                    return self.default_rate_limit
                else:
                    return max(0, self.default_rate_limit - int(current_count))
            else:
                # Fall back to memory calculation
                current_time = time.time()
                window_start = current_time - self.window_size
                client_requests = self.memory_limits[client_id]
                
                # Clean old entries
                while client_requests and client_requests[0] < window_start:
                    client_requests.popleft()
                
                return max(0, self.default_rate_limit - len(client_requests))
                
        except Exception as e:
            self.logger.error(f"Failed to get remaining requests: {e}")
            return 0


class RAGMiddleware(BaseHTTPMiddleware):
    """RAG-specific middleware for request processing."""
    
    def __init__(self, app, enable_logging: bool = True, enable_metrics: bool = True):
        """Initialize RAG middleware.
        
        Args:
            app: FastAPI application
            enable_logging: Enable request logging
            enable_metrics: Enable metrics collection
        """
        super().__init__(app)
        self.enable_logging = enable_logging
        self.enable_metrics = enable_metrics
        self.logger = logging.getLogger(__name__)
        self.metrics = defaultdict(int)
    
    async def __call__(self, request: Request, call_next):
        """Process request with RAG-specific middleware."""
        start_time = time.time()
        
        try:
            # Log request
            if self.enable_logging:
                self.logger.info(f"RAG request: {request.method} {request.url}")
            
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update metrics
            if self.enable_metrics:
                self._update_metrics(request, response, processing_time)
            
            # Add RAG-specific headers
            response.headers["X-RAG-Processing-Time"] = str(processing_time)
            response.headers["X-RAG-Version"] = "1.0.0"
            
            return response
            
        except Exception as e:
            self.logger.error(f"RAG middleware error: {e}")
            raise
    
    def _update_metrics(self, request: Request, response: Response, processing_time: float):
        """Update RAG metrics."""
        # Update request count
        self.metrics["total_requests"] += 1
        
        # Update response status metrics
        status_code = response.status_code
        self.metrics[f"status_{status_code}"] += 1
        
        # Update processing time metrics
        if processing_time < 1.0:
            self.metrics["fast_requests"] += 1
        elif processing_time < 5.0:
            self.metrics["medium_requests"] += 1
        else:
            self.metrics["slow_requests"] += 1
        
        # Update endpoint metrics
        endpoint = request.url.path
        self.metrics[f"endpoint_{endpoint}"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return dict(self.metrics)


class TenantIsolationMiddleware(BaseHTTPMiddleware):
    """Tenant isolation middleware for multi-tenant RAG system."""
    
    def __init__(self, app):
        """Initialize tenant isolation middleware."""
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, request: Request, call_next):
        """Process request with tenant isolation."""
        try:
            # Extract tenant ID from request
            tenant_id = self._extract_tenant_id(request)
            
            if tenant_id:
                # Set tenant context
                request.state.tenant_id = tenant_id
                
                # Add tenant isolation to response
                response = await call_next(request)
                response.headers["X-Tenant-ID"] = tenant_id
                
                return response
            else:
                # No tenant ID found, proceed without isolation
                return await call_next(request)
                
        except Exception as e:
            self.logger.error(f"Tenant isolation error: {e}")
            return await call_next(request)
    
    def _extract_tenant_id(self, request: Request) -> Optional[str]:
        """Extract tenant ID from request."""
        # Try to get from authentication header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, options={"verify_signature": False})
                return payload.get("tenant_id")
            except:
                pass
        
        # Try to get from custom header
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            return tenant_id
        
        # Try to get from query parameters
        tenant_id = request.query_params.get("tenant_id")
        if tenant_id:
            return tenant_id
        
        return None


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware for RAG API."""
    
    def __init__(self, app):
        """Initialize security headers middleware."""
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware for RAG API."""
    
    def __init__(self, app, allowed_origins: List[str] = None, allowed_methods: List[str] = None):
        """Initialize CORS middleware.
        
        Args:
            app: FastAPI application
            allowed_origins: List of allowed origins
            allowed_methods: List of allowed methods
        """
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, request: Request, call_next):
        """Process request with CORS headers."""
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            self._add_cors_headers(response, request)
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        self._add_cors_headers(response, request)
        
        return response
    
    def _add_cors_headers(self, response: Response, request: Request):
        """Add CORS headers to response."""
        origin = request.headers.get("Origin")
        
        if origin in self.allowed_origins or "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key, X-Tenant-ID"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "86400"
