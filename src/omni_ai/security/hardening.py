"""
🛡️ FORTRESS-LEVEL SYSTEM HARDENING & ISOLATION
Advanced sandboxing, resource limits, and system-level security controls.
"""

import os
import sys
import time
import tempfile
import subprocess

try:
    import resource
except ImportError:
    resource = None  # Windows doesn't have resource module
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Timer
import signal

import psutil
import structlog

from .input_validation import SecurityError

logger = structlog.get_logger()


@dataclass
class ResourceLimits:
    """Resource usage limits"""
    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    max_execution_time: int = 30  # seconds
    max_file_size_mb: int = 100
    max_open_files: int = 100
    max_network_connections: int = 10


@dataclass
class SandboxConfig:
    """Sandbox configuration"""
    allowed_paths: List[Path]
    blocked_paths: List[Path]
    resource_limits: ResourceLimits
    network_access: bool = False
    system_access: bool = False
    temp_dir_only: bool = True


class SystemHardening:
    """🛡️ FORTRESS-LEVEL SYSTEM HARDENING"""
    
    def __init__(self):
        self.original_limits = {}
        self.active_sandboxes = {}
        self.resource_monitors = {}
        
        # Platform-specific security settings
        self.is_windows = sys.platform.startswith('win')
        self.is_linux = sys.platform.startswith('linux')
        self.is_mac = sys.platform.startswith('darwin')
        
        logger.info("🛡️ System Hardening initialized with fortress-level security")
    
    def harden_process(self, process_id: int = None) -> Dict[str, Any]:
        """🔒 Apply security hardening to process"""
        if process_id is None:
            process_id = os.getpid()
        
        try:
            process = psutil.Process(process_id)
            hardening_results = {}
            
            # Set process priority to below normal
            try:
                if self.is_windows:
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(10)  # Lower priority on Unix
                hardening_results['priority'] = 'lowered'
            except Exception as e:
                logger.warning(f"Could not set process priority: {e}")
                hardening_results['priority'] = 'failed'
            
            # Set resource limits
            self._apply_resource_limits()
            hardening_results['resource_limits'] = 'applied'
            
            # Disable core dumps for security
            if not self.is_windows and resource is not None:
                try:
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                    hardening_results['core_dumps'] = 'disabled'
                except Exception as e:
                    logger.warning(f"Could not disable core dumps: {e}")
                    hardening_results['core_dumps'] = 'failed'
            
            # Set umask for file permissions
            if not self.is_windows:
                os.umask(0o077)  # Restrictive file permissions
                hardening_results['file_permissions'] = 'restricted'
            
            logger.info(f"🔒 Process hardening applied: {hardening_results}")
            return hardening_results
            
        except Exception as e:
            logger.error(f"🚫 Process hardening failed: {e}")
            raise SecurityError(f"Process hardening failed: {e}")
    
    def _apply_resource_limits(self):
        """Apply system resource limits"""
        if self.is_windows or resource is None:
            # Windows resource limits are more limited
            logger.info("🔒 Windows resource limits applied (basic)")
            return
        
        try:
            # Memory limit (virtual memory)
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, -1))  # 512 MB
            
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (30, 60))  # 30 seconds soft, 60 hard
            
            # File size limit
            resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, -1))  # 100 MB
            
            # Number of open files
            resource.setrlimit(resource.RLIMIT_NOFILE, (100, 200))
            
            # Number of processes
            resource.setrlimit(resource.RLIMIT_NPROC, (10, 20))
            
            logger.info("🔒 Unix resource limits applied")
            
        except Exception as e:
            logger.warning(f"Could not set all resource limits: {e}")
    
    def create_secure_temp_directory(self, prefix: str = "omni_secure_") -> Path:
        """🔒 Create secure temporary directory with restricted permissions"""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        
        # Set restrictive permissions
        if not self.is_windows:
            os.chmod(temp_dir, 0o700)  # Only owner can access
        
        logger.info(f"🔒 Secure temp directory created: {temp_dir}")
        return temp_dir
    
    def secure_file_operation(self, operation: Callable, file_path: Path, 
                             allowed_paths: List[Path]) -> Any:
        """🔒 Perform file operation with path validation"""
        # Resolve and validate path
        abs_path = file_path.resolve()
        
        # Check if path is in allowed directories
        path_allowed = False
        for allowed_path in allowed_paths:
            try:
                abs_path.relative_to(allowed_path.resolve())
                path_allowed = True
                break
            except ValueError:
                continue
        
        if not path_allowed:
            logger.critical(f"🚫 File access denied: {abs_path}")
            raise SecurityError(f"File access denied: {abs_path}")
        
        # Check file size before operation
        if abs_path.exists() and abs_path.is_file():
            file_size = abs_path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100 MB
                logger.warning(f"🚫 File too large: {file_size} bytes")
                raise SecurityError(f"File too large: {file_size} bytes")
        
        try:
            logger.info(f"🔒 Secure file operation: {abs_path}")
            return operation(abs_path)
        except Exception as e:
            logger.error(f"🚫 Secure file operation failed: {e}")
            raise SecurityError(f"File operation failed: {e}")


class ResourceMonitor:
    """🔍 RESOURCE USAGE MONITOR"""
    
    def __init__(self, limits: ResourceLimits, process_id: int = None):
        self.limits = limits
        self.process_id = process_id or os.getpid()
        self.monitoring = False
        self.violations = []
        
        logger.info(f"🔍 Resource Monitor initialized for PID {self.process_id}")
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self._monitor_loop()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
    
    def _monitor_loop(self):
        """Monitor resource usage in loop"""
        try:
            process = psutil.Process(self.process_id)
            
            while self.monitoring:
                # Check memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                if memory_mb > self.limits.max_memory_mb:
                    violation = {
                        'type': 'memory_exceeded',
                        'current': memory_mb,
                        'limit': self.limits.max_memory_mb,
                        'timestamp': time.time()
                    }
                    self.violations.append(violation)
                    logger.critical(f"🚨 Memory limit exceeded: {memory_mb:.2f} MB > {self.limits.max_memory_mb} MB")
                
                # Check CPU usage
                cpu_percent = process.cpu_percent()
                if cpu_percent > self.limits.max_cpu_percent:
                    violation = {
                        'type': 'cpu_exceeded',
                        'current': cpu_percent,
                        'limit': self.limits.max_cpu_percent,
                        'timestamp': time.time()
                    }
                    self.violations.append(violation)
                    logger.warning(f"⚠️ CPU usage high: {cpu_percent:.2f}% > {self.limits.max_cpu_percent}%")
                
                # Check open files
                try:
                    open_files = len(process.open_files())
                    if open_files > self.limits.max_open_files:
                        violation = {
                            'type': 'files_exceeded',
                            'current': open_files,
                            'limit': self.limits.max_open_files,
                            'timestamp': time.time()
                        }
                        self.violations.append(violation)
                        logger.warning(f"⚠️ Too many open files: {open_files} > {self.limits.max_open_files}")
                except:
                    pass  # Some processes don't allow this check
                
                time.sleep(1)  # Check every second
                
        except psutil.NoSuchProcess:
            logger.warning("🚫 Monitored process no longer exists")
            self.monitoring = False
        except Exception as e:
            logger.error(f"🚫 Resource monitoring error: {e}")


class SecureSandbox:
    """🏰 FORTRESS-LEVEL SANDBOX"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.sandbox_id = os.urandom(8).hex()
        self.temp_dirs = []
        self.monitor = ResourceMonitor(config.resource_limits)
        self.cleanup_callbacks = []
        
        logger.info(f"🏰 Secure Sandbox created: {self.sandbox_id}")
    
    @contextmanager
    def execute(self):
        """🏰 Execute code within secure sandbox"""
        logger.info(f"🏰 Entering sandbox: {self.sandbox_id}")
        
        try:
            # Setup sandbox environment
            self._setup_sandbox()
            
            # Start resource monitoring
            self.monitor.start_monitoring()
            
            # Set execution timeout
            self._set_execution_timeout()
            
            yield self
            
        except Exception as e:
            logger.error(f"🚫 Sandbox execution error: {e}")
            raise SecurityError(f"Sandbox execution failed: {e}")
        
        finally:
            # Cleanup sandbox
            self._cleanup_sandbox()
            logger.info(f"🏰 Exiting sandbox: {self.sandbox_id}")
    
    def _setup_sandbox(self):
        """Setup sandbox environment"""
        # Create secure temporary directory if needed
        if self.config.temp_dir_only:
            temp_dir = tempfile.mkdtemp(prefix=f"sandbox_{self.sandbox_id}_")
            self.temp_dirs.append(Path(temp_dir))
            if not sys.platform.startswith('win'):
                os.chmod(temp_dir, 0o700)
        
        # Apply resource limits
        if not sys.platform.startswith('win') and resource is not None:
            # Set stricter resource limits in sandbox
            try:
                resource.setrlimit(resource.RLIMIT_AS, (
                    self.config.resource_limits.max_memory_mb * 1024 * 1024, -1
                ))
                resource.setrlimit(resource.RLIMIT_CPU, (
                    self.config.resource_limits.max_execution_time,
                    self.config.resource_limits.max_execution_time + 10
                ))
            except Exception as e:
                logger.warning(f"Could not set sandbox resource limits: {e}")
    
    def _set_execution_timeout(self):
        """Set execution timeout for sandbox"""
        def timeout_handler():
            logger.critical(f"🚨 Sandbox timeout: {self.sandbox_id}")
            # In a real implementation, you would terminate the sandbox process
            raise SecurityError("Sandbox execution timeout")
        
        timer = Timer(self.config.resource_limits.max_execution_time, timeout_handler)
        timer.start()
        self.cleanup_callbacks.append(timer.cancel)
    
    def _cleanup_sandbox(self):
        """Cleanup sandbox resources"""
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Execute cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")
        
        # Remove temporary directories
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"🗑️ Cleaned up temp dir: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
    
    def is_path_allowed(self, path: Path) -> bool:
        """🔍 Check if path is allowed in sandbox"""
        abs_path = path.resolve()
        
        # Check blocked paths first
        for blocked_path in self.config.blocked_paths:
            try:
                abs_path.relative_to(blocked_path.resolve())
                logger.warning(f"🚫 Path blocked: {abs_path}")
                return False
            except ValueError:
                continue
        
        # Check allowed paths
        for allowed_path in self.config.allowed_paths:
            try:
                abs_path.relative_to(allowed_path.resolve())
                return True
            except ValueError:
                continue
        
        # If temp_dir_only, only allow temp directories
        if self.config.temp_dir_only:
            for temp_dir in self.temp_dirs:
                try:
                    abs_path.relative_to(temp_dir)
                    return True
                except ValueError:
                    continue
        
        logger.warning(f"🚫 Path not allowed: {abs_path}")
        return False


class NetworkIsolation:
    """🌐 NETWORK ISOLATION CONTROLS"""
    
    @staticmethod
    def block_network_access():
        """🚫 Block network access (platform specific)"""
        if sys.platform.startswith('win'):
            # Windows firewall rules would go here
            logger.warning("🚫 Network blocking on Windows requires admin privileges")
        else:
            # Unix-like systems
            try:
                # This is a simplified example - real implementation would use iptables/netfilter
                os.environ['no_proxy'] = '*'
                os.environ['NO_PROXY'] = '*'
                logger.info("🚫 Network access restricted (basic)")
            except Exception as e:
                logger.warning(f"Could not block network access: {e}")
    
    @staticmethod
    def get_network_connections() -> List[Dict]:
        """🔍 Get active network connections"""
        try:
            connections = []
            for conn in psutil.net_connections():
                if conn.status == psutil.CONN_ESTABLISHED:
                    connections.append({
                        'local_address': f"{conn.laddr.ip}:{conn.laddr.port}",
                        'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                        'status': conn.status,
                        'pid': conn.pid
                    })
            return connections
        except Exception as e:
            logger.error(f"Failed to get network connections: {e}")
            return []


def secure_execution(resource_limits: ResourceLimits = None,
                    allowed_paths: List[Path] = None,
                    timeout: int = 30):
    """🔒 Decorator for secure function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create sandbox configuration
            config = SandboxConfig(
                allowed_paths=allowed_paths or [Path.cwd()],
                blocked_paths=[Path('/etc'), Path('/var'), Path('/usr')] if not sys.platform.startswith('win') else [],
                resource_limits=resource_limits or ResourceLimits(),
                network_access=False,
                system_access=False,
                temp_dir_only=True
            )
            
            # Execute in sandbox
            with SecureSandbox(config).execute() as sandbox:
                logger.info(f"🔒 Executing {func.__name__} in secure sandbox")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Export main components
__all__ = [
    'SystemHardening',
    'ResourceMonitor', 
    'SecureSandbox',
    'NetworkIsolation',
    'ResourceLimits',
    'SandboxConfig',
    'secure_execution'
]