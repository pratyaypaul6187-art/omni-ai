"""
🔍 FORTRESS-LEVEL SECURITY MONITORING & INTRUSION DETECTION
Real-time monitoring, logging, and threat detection system.
"""

import time
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

import psutil
import structlog
from structlog import get_logger

logger = get_logger()


class SecurityEventType(Enum):
    """Security event types for monitoring"""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_DENIED = "authz_denied" 
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    FILE_ACCESS_DENIED = "file_access_denied"
    SYSTEM_RESOURCE_ABUSE = "resource_abuse"
    POTENTIAL_ATTACK = "potential_attack"
    SECURITY_VIOLATION = "security_violation"


class ThreatLevel(Enum):
    """Threat severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    details: Dict[str, Any]
    action_taken: Optional[str] = None
    event_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.event_id:
            # Generate unique event ID
            data = f"{self.event_type.value}{self.timestamp}{self.source_ip}{self.user_id}"
            self.event_id = hashlib.sha256(data.encode()).hexdigest()[:16]


class SecurityMonitor:
    """🔍 FORTRESS-LEVEL SECURITY MONITORING SYSTEM"""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("security_events.log")
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events in memory
        self.threat_counters: Dict[str, int] = defaultdict(int)
        self.ip_activity: Dict[str, List[float]] = defaultdict(list)
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring thresholds
        self.SUSPICIOUS_ACTIVITY_THRESHOLD = 10  # events per minute
        self.RATE_LIMIT_WINDOW = 60  # seconds
        self.MAX_EVENTS_PER_IP = 100  # per minute
        
        # Initialize structured logging
        self._setup_security_logging()
        
        logger.info("🔍 Security Monitor initialized with fortress-level detection")
    
    def _setup_security_logging(self):
        """Setup structured security logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def log_security_event(self, event_type: SecurityEventType, threat_level: ThreatLevel,
                          source_ip: str, user_id: Optional[str] = None, 
                          details: Optional[Dict[str, Any]] = None,
                          action_taken: Optional[str] = None):
        """🔍 Log security event with full context"""
        
        event = SecurityEvent(
            event_type=event_type,
            threat_level=threat_level,
            timestamp=datetime.now(timezone.utc),
            source_ip=source_ip,
            user_id=user_id,
            details=details or {},
            action_taken=action_taken
        )
        
        # Store event
        self.events.append(event)
        
        # Update threat counters
        self.threat_counters[f"{source_ip}_{event_type.value}"] += 1
        
        # Log to structured logger
        log_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'threat_level': event.threat_level.value,
            'source_ip': source_ip,
            'user_id': user_id,
            'details': details,
            'action_taken': action_taken
        }
        
        # Choose appropriate log level based on threat
        if threat_level == ThreatLevel.CRITICAL:
            logger.critical("🚨 CRITICAL SECURITY EVENT", **log_data)
        elif threat_level == ThreatLevel.HIGH:
            logger.error("⚠️ HIGH THREAT EVENT", **log_data)
        elif threat_level == ThreatLevel.MEDIUM:
            logger.warning("⚠️ MEDIUM THREAT EVENT", **log_data)
        elif threat_level == ThreatLevel.LOW:
            logger.info("ℹ️ LOW THREAT EVENT", **log_data)
        else:
            logger.info("ℹ️ SECURITY EVENT", **log_data)
        
        # Write to security log file
        self._write_to_security_log(event)
        
        # Check for patterns and anomalies
        self._analyze_security_patterns(event)
        
        # Trigger alerts if necessary
        if threat_level.value in ['high', 'critical']:
            self._trigger_security_alert(event)
    
    def _write_to_security_log(self, event: SecurityEvent):
        """Write event to dedicated security log file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                log_entry = {
                    'timestamp': event.timestamp.isoformat(),
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'threat_level': event.threat_level.value,
                    'source_ip': event.source_ip,
                    'user_id': event.user_id,
                    'details': event.details,
                    'action_taken': event.action_taken
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write security log: {e}")
    
    def _analyze_security_patterns(self, event: SecurityEvent):
        """🔍 Analyze patterns to detect potential attacks"""
        current_time = time.time()
        source_ip = event.source_ip
        
        # Track IP activity
        if source_ip not in self.ip_activity:
            self.ip_activity[source_ip] = []
        
        self.ip_activity[source_ip].append(current_time)
        
        # Clean old entries (outside monitoring window)
        cutoff_time = current_time - self.RATE_LIMIT_WINDOW
        self.ip_activity[source_ip] = [
            t for t in self.ip_activity[source_ip] if t > cutoff_time
        ]
        
        # Check for suspicious patterns
        recent_events = len(self.ip_activity[source_ip])
        
        if recent_events > self.MAX_EVENTS_PER_IP:
            self.log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                ThreatLevel.HIGH,
                source_ip,
                details={
                    'pattern': 'high_frequency_requests',
                    'event_count': recent_events,
                    'time_window': self.RATE_LIMIT_WINDOW
                },
                action_taken="IP flagged for review"
            )
        
        # Check for attack patterns
        self._check_attack_patterns(event)
    
    def _check_attack_patterns(self, event: SecurityEvent):
        """🔍 Check for known attack patterns"""
        source_ip = event.source_ip
        
        # Pattern 1: Multiple authentication failures
        auth_failures = sum(1 for e in self.events 
                           if e.source_ip == source_ip 
                           and e.event_type == SecurityEventType.AUTHENTICATION_FAILURE
                           and (event.timestamp - e.timestamp).seconds < 300)  # 5 minutes
        
        if auth_failures >= 5:
            self.log_security_event(
                SecurityEventType.POTENTIAL_ATTACK,
                ThreatLevel.CRITICAL,
                source_ip,
                details={
                    'attack_type': 'credential_stuffing',
                    'failure_count': auth_failures,
                    'pattern': 'multiple_auth_failures'
                },
                action_taken="IP blocked, credentials flagged"
            )
        
        # Pattern 2: Multiple input validation failures (injection attempts)
        validation_failures = sum(1 for e in self.events
                                if e.source_ip == source_ip
                                and e.event_type == SecurityEventType.INPUT_VALIDATION_FAILURE
                                and (event.timestamp - e.timestamp).seconds < 300)
        
        if validation_failures >= 3:
            self.log_security_event(
                SecurityEventType.POTENTIAL_ATTACK,
                ThreatLevel.HIGH,
                source_ip,
                details={
                    'attack_type': 'injection_attack',
                    'validation_failure_count': validation_failures,
                    'pattern': 'multiple_injection_attempts'
                },
                action_taken="IP flagged, enhanced monitoring enabled"
            )
    
    def _trigger_security_alert(self, event: SecurityEvent):
        """🚨 Trigger security alerts for high-priority events"""
        alert_data = {
            'event_id': event.event_id,
            'threat_level': event.threat_level.value,
            'event_type': event.event_type.value,
            'source_ip': event.source_ip,
            'user_id': event.user_id,
            'timestamp': event.timestamp.isoformat(),
            'details': event.details
        }
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for security alerts"""
        self.alert_callbacks.append(callback)
        logger.info("🔔 Alert callback registered")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """🔍 Get security monitoring dashboard data"""
        current_time = time.time()
        last_hour = current_time - 3600
        
        recent_events = [e for e in self.events 
                        if e.timestamp.timestamp() > last_hour]
        
        # Count events by type
        event_counts = defaultdict(int)
        threat_counts = defaultdict(int)
        ip_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type.value] += 1
            threat_counts[event.threat_level.value] += 1
            ip_counts[event.source_ip] += 1
        
        # Get top suspicious IPs
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        dashboard = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'monitoring_period': '1 hour',
            'total_events': len(recent_events),
            'event_breakdown': dict(event_counts),
            'threat_breakdown': dict(threat_counts),
            'top_source_ips': top_ips,
            'system_status': self._get_system_health(),
            'active_threats': self._get_active_threats()
        }
        
        return dashboard
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'active_connections': len(psutil.net_connections()),
                'system_status': 'healthy' if cpu_percent < 80 and memory.percent < 80 else 'warning'
            }
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def _get_active_threats(self) -> List[Dict[str, Any]]:
        """Get list of active threats"""
        current_time = time.time()
        threat_window = current_time - 300  # 5 minutes
        
        recent_threats = [
            e for e in self.events
            if e.timestamp.timestamp() > threat_window
            and e.threat_level.value in ['high', 'critical']
        ]
        
        threats = []
        for event in recent_threats[-10:]:  # Last 10 threats
            threats.append({
                'event_id': event.event_id,
                'type': event.event_type.value,
                'threat_level': event.threat_level.value,
                'source_ip': event.source_ip,
                'timestamp': event.timestamp.isoformat(),
                'details': event.details
            })
        
        return threats


class IntrusionDetectionSystem:
    """🛡️ ADVANCED INTRUSION DETECTION SYSTEM"""
    
    def __init__(self, monitor: SecurityMonitor):
        self.monitor = monitor
        self.behavioral_baselines: Dict[str, Any] = {}
        self.anomaly_thresholds = {
            'request_rate_multiplier': 3.0,  # 3x normal rate
            'unusual_hours_threshold': 0.1,  # Less than 10% of normal activity
            'geographic_anomaly_threshold': 0.05  # 5% of requests from unusual locations
        }
        
        logger.info("🛡️ Intrusion Detection System activated")
    
    def analyze_behavioral_anomaly(self, user_id: str, activity_data: Dict[str, Any]) -> bool:
        """🔍 Detect behavioral anomalies"""
        if user_id not in self.behavioral_baselines:
            # First time seeing this user, establish baseline
            self.behavioral_baselines[user_id] = activity_data
            return False
        
        baseline = self.behavioral_baselines[user_id]
        
        # Check for anomalies
        anomalies_detected = []
        
        # Request rate anomaly
        current_rate = activity_data.get('requests_per_minute', 0)
        normal_rate = baseline.get('avg_requests_per_minute', 1)
        
        if current_rate > normal_rate * self.anomaly_thresholds['request_rate_multiplier']:
            anomalies_detected.append('high_request_rate')
        
        # Time-based anomaly
        current_hour = datetime.now().hour
        normal_hours = baseline.get('active_hours', [])
        
        if normal_hours and current_hour not in normal_hours:
            anomalies_detected.append('unusual_activity_hours')
        
        if anomalies_detected:
            self.monitor.log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                ThreatLevel.MEDIUM,
                activity_data.get('source_ip', 'unknown'),
                user_id=user_id,
                details={
                    'anomalies': anomalies_detected,
                    'current_activity': activity_data,
                    'baseline': baseline
                },
                action_taken="Behavioral analysis flagged for review"
            )
            return True
        
        # Update baseline with new data
        self._update_baseline(user_id, activity_data)
        return False
    
    def _update_baseline(self, user_id: str, new_data: Dict[str, Any]):
        """Update user behavioral baseline"""
        baseline = self.behavioral_baselines.get(user_id, {})
        
        # Simple moving average update
        alpha = 0.1  # Learning rate
        for key, value in new_data.items():
            if key in baseline and isinstance(value, (int, float)):
                baseline[key] = baseline[key] * (1 - alpha) + value * alpha
            else:
                baseline[key] = value
        
        self.behavioral_baselines[user_id] = baseline


def security_event_handler(event_type: SecurityEventType, threat_level: ThreatLevel = ThreatLevel.INFO):
    """🔍 Decorator for automatic security event logging"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract source IP if available (you'd implement this based on your context)
            source_ip = kwargs.get('source_ip', 'localhost')
            user_id = kwargs.get('user_id', None)
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful operation
                if hasattr(wrapper, '_security_monitor'):
                    wrapper._security_monitor.log_security_event(
                        event_type,
                        threat_level,
                        source_ip,
                        user_id=user_id,
                        details={
                            'function': func.__name__,
                            'execution_time': time.time() - start_time,
                            'status': 'success'
                        }
                    )
                
                return result
                
            except Exception as e:
                # Log failed operation
                if hasattr(wrapper, '_security_monitor'):
                    wrapper._security_monitor.log_security_event(
                        SecurityEventType.SECURITY_VIOLATION,
                        ThreatLevel.MEDIUM,
                        source_ip,
                        user_id=user_id,
                        details={
                            'function': func.__name__,
                            'error': str(e),
                            'execution_time': time.time() - start_time,
                            'status': 'failed'
                        }
                    )
                raise
        
        return wrapper
    return decorator


# Global security monitor instance
_global_monitor: Optional[SecurityMonitor] = None

def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SecurityMonitor()
    return _global_monitor


# Export main components
__all__ = [
    'SecurityMonitor',
    'IntrusionDetectionSystem',
    'SecurityEvent',
    'SecurityEventType',
    'ThreatLevel',
    'security_event_handler',
    'get_security_monitor'
]