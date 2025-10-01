"""
🛡️ FORTRESS LEVEL INPUT VALIDATION & SANITIZATION
Multi-layered input security to prevent all forms of injection attacks.
"""

import re
import html
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import bleach
import validators
from pydantic import BaseModel, Field, field_validator
from structlog import get_logger

logger = get_logger()


class SecurityError(Exception):
    """Custom exception for security violations."""
    pass


class InputSanitizer:
    """🔒 FORTRESS-LEVEL INPUT SANITIZATION"""
    
    # Extremely restrictive allow lists
    ALLOWED_TEXT_CHARS = re.compile(r'^[a-zA-Z0-9\s\.\,\!\?\-\'\"]*$')
    ALLOWED_PATH_CHARS = re.compile(r'^[a-zA-Z0-9\-\_\.\\/\:]*$')
    ALLOWED_FILENAME_CHARS = re.compile(r'^[a-zA-Z0-9\-\_\.]*$')
    
    # Dangerous patterns that are NEVER allowed
    DANGEROUS_PATTERNS = [
        # Command injection patterns
        r'[;&|`$(){}[\]<>]',
        r'\.\./',
        r'\\.\\.\\',
        r'/etc/',
        r'/bin/',
        r'/usr/',
        r'/var/',
        r'C:\\',
        r'\\Windows\\',
        r'\\System32\\',
        # Script injection patterns
        r'<script[^>]*>',
        r'javascript:',
        r'vbscript:',
        r'onload=',
        r'onerror=',
        r'onclick=',
        # SQL injection patterns
        r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|UNION|OR|AND)\b',
        # File inclusion patterns
        r'(file|ftp|http|https)://',
        r'\\\\',
        r'/dev/',
        r'/proc/',
        # Python code injection
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'subprocess',
        r'os\.',
        r'sys\.',
        # PowerShell patterns
        r'powershell',
        r'cmd\.exe',
        r'Invoke-',
    ]
    
    @classmethod
    def sanitize_text(cls, text: str, max_length: int = 10000) -> str:
        """🔒 Fortress-level text sanitization"""
        if not isinstance(text, str):
            raise SecurityError(f"Expected string, got {type(text)}")
        
        # Length check
        if len(text) > max_length:
            logger.warning(f"Text too long: {len(text)} > {max_length}")
            raise SecurityError(f"Text exceeds maximum length of {max_length}")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.critical(f"Dangerous pattern detected: {pattern}")
                raise SecurityError(f"Dangerous pattern detected in input")
        
        # HTML escape
        text = html.escape(text, quote=True)
        
        # Additional bleach sanitization
        text = bleach.clean(
            text,
            tags=[],  # No HTML tags allowed
            attributes={},  # No attributes allowed
            strip=True,
            strip_comments=True
        )
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        logger.info(f"Text sanitized: {len(text)} chars")
        return text
    
    @classmethod
    def sanitize_filename(cls, filename: str, max_length: int = 255) -> str:
        """🔒 Ultra-secure filename sanitization"""
        if not isinstance(filename, str):
            raise SecurityError(f"Expected string filename, got {type(filename)}")
        
        # Remove any path separators
        filename = Path(filename).name
        
        # Length check
        if len(filename) > max_length:
            raise SecurityError(f"Filename too long: {len(filename)} > {max_length}")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                logger.critical(f"Dangerous filename pattern: {pattern}")
                raise SecurityError("Dangerous pattern in filename")
        
        # Only allow safe characters
        if not cls.ALLOWED_FILENAME_CHARS.match(filename):
            logger.warning(f"Invalid characters in filename: {filename}")
            raise SecurityError("Invalid characters in filename")
        
        # Additional restrictions
        dangerous_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
            'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3',
            'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        if filename.upper() in dangerous_names:
            raise SecurityError("Reserved system filename")
        
        logger.info(f"Filename sanitized: {filename}")
        return filename
    
    @classmethod
    def sanitize_path(cls, path: Union[str, Path], base_dir: Path) -> Path:
        """🔒 Fortress-level path sanitization with sandboxing"""
        if isinstance(path, str):
            path = Path(path)
        
        # Convert to absolute path
        abs_path = path.resolve()
        abs_base = base_dir.resolve()
        
        # CRITICAL: Ensure path is within base directory (prevent directory traversal)
        try:
            abs_path.relative_to(abs_base)
        except ValueError:
            logger.critical(f"Path traversal attempt: {abs_path} not in {abs_base}")
            raise SecurityError("Path outside allowed directory")
        
        # Check each part of the path
        for part in abs_path.parts:
            if not cls.ALLOWED_FILENAME_CHARS.match(part):
                logger.critical(f"Invalid path component: {part}")
                raise SecurityError("Invalid path component")
        
        logger.info(f"Path sanitized: {abs_path}")
        return abs_path


class SecureTextInput(BaseModel):
    """🛡️ Pydantic model for secure text input validation"""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Input text with strict validation"
    )
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        """Validate text input using our fortress-level sanitizer"""
        return InputSanitizer.sanitize_text(v)


class SecureFileInput(BaseModel):
    """🛡️ Pydantic model for secure file input validation"""
    
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="File path with strict validation"
    )
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        """Validate file path"""
        # Basic path validation
        if not Path(v).is_file():
            raise ValueError("File does not exist")
        
        # Additional security checks will be done in the sanitizer
        return v


class SecurityValidator:
    """🔒 ADVANCED SECURITY VALIDATION ENGINE"""
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL with strict security checks"""
        try:
            # Basic validation
            if not validators.url(url):
                return False
            
            parsed = urlparse(url)
            
            # Only allow HTTPS
            if parsed.scheme != 'https':
                logger.warning(f"Non-HTTPS URL rejected: {url}")
                return False
            
            # Block local/private IP ranges
            hostname = parsed.hostname
            if hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                logger.warning(f"Local URL rejected: {url}")
                return False
            
            # Block private IP ranges
            import ipaddress
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    logger.warning(f"Private IP URL rejected: {url}")
                    return False
            except ValueError:
                pass  # Not an IP address, continue with domain validation
            
            return True
            
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email with security checks"""
        try:
            if not validators.email(email):
                return False
            
            # Additional checks
            if len(email) > 254:  # RFC 5321 limit
                return False
            
            # Block suspicious domains
            suspicious_domains = {
                '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
                'mailinator.com', 'yopmail.com'
            }
            domain = email.split('@')[1].lower()
            if domain in suspicious_domains:
                logger.warning(f"Suspicious email domain: {domain}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Email validation error: {e}")
            return False
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def validate_content_type(file_path: Path, allowed_types: List[str]) -> bool:
        """Validate file content type"""
        try:
            import magic
            
            file_type = magic.from_file(str(file_path), mime=True)
            
            if file_type not in allowed_types:
                logger.warning(f"Disallowed file type: {file_type}")
                return False
            
            return True
            
        except ImportError:
            # Fallback to extension check
            if file_path.suffix.lower() not in [f".{t.split('/')[1]}" for t in allowed_types]:
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Content type validation error: {e}")
            return False


def secure_input_decorator(max_length: int = 10000):
    """🔒 Decorator for automatic input sanitization"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Sanitize all string arguments
            sanitized_args = []
            for arg in args:
                if isinstance(arg, str):
                    try:
                        sanitized_args.append(InputSanitizer.sanitize_text(arg, max_length))
                    except SecurityError:
                        logger.critical(f"Security violation in function {func.__name__}")
                        raise
                else:
                    sanitized_args.append(arg)
            
            # Sanitize string values in kwargs
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str):
                    try:
                        sanitized_kwargs[key] = InputSanitizer.sanitize_text(value, max_length)
                    except SecurityError:
                        logger.critical(f"Security violation in function {func.__name__}")
                        raise
                else:
                    sanitized_kwargs[key] = value
            
            return func(*sanitized_args, **sanitized_kwargs)
        return wrapper
    return decorator


# Export main components
__all__ = [
    'InputSanitizer',
    'SecureTextInput', 
    'SecureFileInput',
    'SecurityValidator',
    'SecurityError',
    'secure_input_decorator'
]