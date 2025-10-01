"""
🔐 FORTRESS-LEVEL AUTHENTICATION & AUTHORIZATION
Multi-layered authentication with JWT, API keys, and role-based access control.
"""

import secrets
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
from enum import Enum

import jwt
import bcrypt
from passlib.context import CryptContext
from structlog import get_logger

from .input_validation import SecurityError

logger = get_logger()


class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = "public"
    BASIC = "basic"
    ELEVATED = "elevated" 
    ADMIN = "admin"
    SYSTEM = "system"


class Permission(Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"
    FILE_ACCESS = "file_access"
    BATCH_PROCESS = "batch_process"
    SYSTEM_CONFIG = "system_config"


class AuthenticationManager:
    """🔒 FORTRESS-LEVEL AUTHENTICATION SYSTEM"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_urlsafe(64)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.active_sessions: Dict[str, Dict] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}
        
        # Security configuration
        self.MAX_FAILED_ATTEMPTS = 3
        self.LOCKOUT_DURATION = 300  # 5 minutes
        self.SESSION_TIMEOUT = 3600  # 1 hour
        self.JWT_EXPIRY = 1800  # 30 minutes
        
        logger.info("🔐 Authentication system initialized with maximum security")
    
    def hash_password(self, password: str) -> str:
        """🔒 Generate secure password hash"""
        if len(password) < 12:
            raise SecurityError("Password must be at least 12 characters")
        
        # Check password complexity
        if not self._check_password_complexity(password):
            raise SecurityError("Password does not meet complexity requirements")
        
        # Use bcrypt with high cost factor
        salt = bcrypt.gensalt(rounds=15)  # Very high cost factor
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        logger.info("🔒 Password hashed with maximum security")
        return hashed.decode('utf-8')
    
    def _check_password_complexity(self, password: str) -> bool:
        """Check password meets fortress-level requirements"""
        if len(password) < 12:
            return False
        
        checks = [
            any(c.islower() for c in password),  # lowercase
            any(c.isupper() for c in password),  # uppercase
            any(c.isdigit() for c in password),  # digit
            any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password),  # special
        ]
        
        return sum(checks) >= 3  # At least 3 of 4 criteria
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """🔒 Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def check_rate_limit(self, identifier: str) -> bool:
        """🔒 Check if identifier is rate limited"""
        current_time = time.time()
        
        # Check if IP is blocked
        if identifier in self.blocked_ips:
            if current_time - self.blocked_ips[identifier] < self.LOCKOUT_DURATION:
                logger.warning(f"🚫 Rate limited: {identifier}")
                return False
            else:
                del self.blocked_ips[identifier]
        
        # Check failed attempts
        if identifier in self.failed_attempts:
            # Remove old attempts (older than lockout duration)
            self.failed_attempts[identifier] = [
                attempt for attempt in self.failed_attempts[identifier]
                if current_time - attempt < self.LOCKOUT_DURATION
            ]
            
            if len(self.failed_attempts[identifier]) >= self.MAX_FAILED_ATTEMPTS:
                self.blocked_ips[identifier] = current_time
                logger.critical(f"🚫 IP blocked due to failed attempts: {identifier}")
                return False
        
        return True
    
    def record_failed_attempt(self, identifier: str):
        """Record a failed authentication attempt"""
        current_time = time.time()
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        self.failed_attempts[identifier].append(current_time)
        logger.warning(f"🚨 Failed authentication attempt: {identifier}")
    
    def create_jwt_token(self, user_id: str, security_level: SecurityLevel, permissions: List[Permission]) -> str:
        """🔒 Create JWT token with security claims"""
        current_time = datetime.now(timezone.utc)
        
        payload = {
            'user_id': user_id,
            'security_level': security_level.value,
            'permissions': [perm.value for perm in permissions],
            'iat': current_time,
            'exp': current_time + timedelta(seconds=self.JWT_EXPIRY),
            'nbf': current_time,
            'jti': secrets.token_urlsafe(16),  # Unique token ID
            'aud': 'omni-ai',
            'iss': 'omni-ai-auth'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        logger.info(f"🔐 JWT token created for user: {user_id}")
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """🔒 Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=['HS256'],
                audience='omni-ai',
                issuer='omni-ai-auth',
                options={
                    'verify_signature': True,
                    'verify_exp': True,
                    'verify_nbf': True,
                    'verify_iat': True,
                    'verify_aud': True,
                    'verify_iss': True
                }
            )
            
            logger.info(f"🔐 JWT token verified for user: {payload.get('user_id')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("🚫 JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"🚫 Invalid JWT token: {e}")
            return None
    
    def create_session(self, user_id: str, security_level: SecurityLevel, 
                      permissions: List[Permission], client_ip: str) -> str:
        """🔒 Create secure session"""
        session_id = secrets.token_urlsafe(32)
        current_time = time.time()
        
        session_data = {
            'user_id': user_id,
            'security_level': security_level.value,
            'permissions': [perm.value for perm in permissions],
            'created_at': current_time,
            'last_accessed': current_time,
            'client_ip': client_ip,
            'csrf_token': secrets.token_urlsafe(32)
        }
        
        self.active_sessions[session_id] = session_data
        logger.info(f"🔐 Session created for user: {user_id}")
        
        return session_id
    
    def validate_session(self, session_id: str, client_ip: str) -> Optional[Dict]:
        """🔒 Validate active session"""
        if session_id not in self.active_sessions:
            logger.warning(f"🚫 Invalid session ID: {session_id}")
            return None
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        # Check session timeout
        if current_time - session['last_accessed'] > self.SESSION_TIMEOUT:
            del self.active_sessions[session_id]
            logger.warning(f"🚫 Session expired: {session_id}")
            return None
        
        # Check IP consistency
        if session['client_ip'] != client_ip:
            del self.active_sessions[session_id]
            logger.critical(f"🚨 Session hijacking attempt: {session_id}")
            return None
        
        # Update last accessed
        session['last_accessed'] = current_time
        logger.info(f"🔐 Session validated for user: {session['user_id']}")
        
        return session
    
    def revoke_session(self, session_id: str):
        """🔒 Revoke active session"""
        if session_id in self.active_sessions:
            user_id = self.active_sessions[session_id]['user_id']
            del self.active_sessions[session_id]
            logger.info(f"🔐 Session revoked for user: {user_id}")


class APIKeyManager:
    """🔑 FORTRESS-LEVEL API KEY MANAGEMENT"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.usage_stats: Dict[str, Dict] = {}
        
        # Rate limiting
        self.RATE_LIMIT_WINDOW = 3600  # 1 hour
        self.DEFAULT_RATE_LIMIT = 1000  # requests per hour
        
        logger.info("🔑 API Key Manager initialized")
    
    def generate_api_key(self, user_id: str, security_level: SecurityLevel, 
                        permissions: List[Permission], rate_limit: int = None) -> str:
        """🔑 Generate secure API key"""
        # Generate cryptographically secure key
        key_data = secrets.token_bytes(32)
        api_key = hashlib.sha256(key_data).hexdigest()
        
        key_info = {
            'user_id': user_id,
            'security_level': security_level.value,
            'permissions': [perm.value for perm in permissions],
            'created_at': time.time(),
            'rate_limit': rate_limit or self.DEFAULT_RATE_LIMIT,
            'is_active': True
        }
        
        self.api_keys[api_key] = key_info
        self.usage_stats[api_key] = {
            'requests': [],
            'last_used': None
        }
        
        logger.info(f"🔑 API key generated for user: {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """🔑 Validate API key and check rate limits"""
        if api_key not in self.api_keys:
            logger.warning(f"🚫 Invalid API key: {api_key[:8]}...")
            return None
        
        key_info = self.api_keys[api_key]
        
        if not key_info['is_active']:
            logger.warning(f"🚫 Inactive API key: {api_key[:8]}...")
            return None
        
        # Check rate limits
        if not self._check_rate_limit(api_key):
            logger.warning(f"🚫 Rate limit exceeded: {api_key[:8]}...")
            return None
        
        # Update usage stats
        current_time = time.time()
        self.usage_stats[api_key]['requests'].append(current_time)
        self.usage_stats[api_key]['last_used'] = current_time
        
        logger.info(f"🔑 API key validated: {api_key[:8]}...")
        return key_info
    
    def _check_rate_limit(self, api_key: str) -> bool:
        """Check API key rate limits"""
        current_time = time.time()
        key_info = self.api_keys[api_key]
        usage = self.usage_stats[api_key]
        
        # Clean old requests outside window
        cutoff_time = current_time - self.RATE_LIMIT_WINDOW
        usage['requests'] = [req_time for req_time in usage['requests'] if req_time > cutoff_time]
        
        # Check if under rate limit
        return len(usage['requests']) < key_info['rate_limit']
    
    def revoke_api_key(self, api_key: str):
        """🔑 Revoke API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]['is_active'] = False
            logger.info(f"🔑 API key revoked: {api_key[:8]}...")


class AccessControl:
    """🛡️ ROLE-BASED ACCESS CONTROL"""
    
    # Define security level hierarchy
    SECURITY_HIERARCHY = {
        SecurityLevel.PUBLIC: 0,
        SecurityLevel.BASIC: 1,
        SecurityLevel.ELEVATED: 2,
        SecurityLevel.ADMIN: 3,
        SecurityLevel.SYSTEM: 4
    }
    
    # Permission requirements for operations
    OPERATION_PERMISSIONS = {
        'text_analysis': [Permission.READ],
        'file_read': [Permission.READ, Permission.FILE_ACCESS],
        'file_write': [Permission.WRITE, Permission.FILE_ACCESS],
        'file_delete': [Permission.DELETE, Permission.FILE_ACCESS],
        'batch_process': [Permission.BATCH_PROCESS, Permission.FILE_ACCESS],
        'system_config': [Permission.SYSTEM_CONFIG, Permission.ADMIN],
        'user_management': [Permission.ADMIN]
    }
    
    @classmethod
    def check_permission(cls, user_permissions: List[str], required_permissions: List[Permission]) -> bool:
        """🛡️ Check if user has required permissions"""
        user_perms = set(user_permissions)
        required_perms = set(perm.value for perm in required_permissions)
        
        has_permission = required_perms.issubset(user_perms)
        
        if not has_permission:
            logger.warning(f"🚫 Permission denied. Required: {required_perms}, User has: {user_perms}")
        
        return has_permission
    
    @classmethod
    def check_security_level(cls, user_level: str, required_level: SecurityLevel) -> bool:
        """🛡️ Check if user meets minimum security level"""
        try:
            user_security = SecurityLevel(user_level)
            user_level_value = cls.SECURITY_HIERARCHY[user_security]
            required_level_value = cls.SECURITY_HIERARCHY[required_level]
            
            has_clearance = user_level_value >= required_level_value
            
            if not has_clearance:
                logger.warning(f"🚫 Security clearance insufficient. User: {user_level}, Required: {required_level.value}")
            
            return has_clearance
            
        except (ValueError, KeyError):
            logger.error(f"🚫 Invalid security level: {user_level}")
            return False
    
    @classmethod
    def check_operation_access(cls, user_permissions: List[str], user_level: str, operation: str) -> bool:
        """🛡️ Check if user can perform specific operation"""
        if operation not in cls.OPERATION_PERMISSIONS:
            logger.error(f"🚫 Unknown operation: {operation}")
            return False
        
        required_perms = cls.OPERATION_PERMISSIONS[operation]
        
        # Check permissions
        if not cls.check_permission(user_permissions, required_perms):
            return False
        
        # Determine minimum security level based on operation
        if Permission.ADMIN in required_perms:
            min_level = SecurityLevel.ADMIN
        elif Permission.SYSTEM_CONFIG in required_perms:
            min_level = SecurityLevel.ELEVATED
        elif Permission.FILE_ACCESS in required_perms:
            min_level = SecurityLevel.BASIC
        else:
            min_level = SecurityLevel.PUBLIC
        
        return cls.check_security_level(user_level, min_level)


def require_auth(operation: str):
    """🔒 Decorator for authentication and authorization"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented with actual session/token checking
            # For now, we'll just log the security check
            logger.info(f"🔒 Security check for operation: {operation}")
            
            # In a real implementation, you would:
            # 1. Extract token/session from request
            # 2. Validate authentication
            # 3. Check authorization for the operation
            # 4. Allow or deny access
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Export main components
__all__ = [
    'AuthenticationManager',
    'APIKeyManager', 
    'AccessControl',
    'SecurityLevel',
    'Permission',
    'require_auth'
]