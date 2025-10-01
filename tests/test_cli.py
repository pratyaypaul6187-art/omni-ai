"""Tests for CLI functionality."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
import tempfile

from omni_ai.cli import app


runner = CliRunner()


class TestCLI:
    """Test cases for CLI commands."""
    
    def test_help_command(self):
        """Test that help command works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Safe, ethical AI toolkit" in result.stdout
        assert "Commands" in result.stdout
    
    def test_hello_command_no_name(self):
        """Test hello command without name."""
        result = runner.invoke(app, ["hello"])
        assert result.exit_code == 0
        assert "Hello from Omni AI!" in result.stdout
        assert "Staying safe and helpful!" in result.stdout
    
    def test_hello_command_with_name(self):
        """Test hello command with name."""
        result = runner.invoke(app, ["hello", "Alice"])
        assert result.exit_code == 0
        assert "Hello Alice!" in result.stdout
        assert "🤖" in result.stdout
    
    def test_safety_check_command(self):
        """Test safety check command."""
        result = runner.invoke(app, ["safety-check"])
        assert result.exit_code == 0
        assert "Safe Mode" in result.stdout
        assert "ethical" in result.stdout
        assert "Supported Use Cases" in result.stdout
    
    def test_analyze_text_command_direct(self):
        """Test analyze-text command with direct text input."""
        result = runner.invoke(app, ["analyze-text", "This is a test sentence."])
        assert result.exit_code == 0
        assert "Text Statistics" in result.stdout
        assert "Word Count" in result.stdout
        assert "Readability Analysis" in result.stdout
    
    def test_analyze_text_command_no_input(self):
        """Test analyze-text command without input (should fail)."""
        result = runner.invoke(app, ["analyze-text"])
        assert result.exit_code == 1
        assert "No text provided" in result.stdout
    
    def test_analyze_text_command_with_file(self):
        """Test analyze-text command with file input."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test file with some content for analysis.")
            temp_file = Path(f.name)
        
        try:
            result = runner.invoke(app, ["analyze-text", "--file", str(temp_file)])
            assert result.exit_code == 0
            assert "Text Statistics" in result.stdout
        finally:
            temp_file.unlink()  # Clean up
    
    def test_analyze_text_command_nonexistent_file(self):
        """Test analyze-text command with non-existent file."""
        result = runner.invoke(app, ["analyze-text", "--file", "nonexistent.txt"])
        assert result.exit_code == 1
        assert "does not exist" in result.stdout
    
    def test_process_file_nonexistent(self):
        """Test process-file command with non-existent file."""
        result = runner.invoke(app, ["process-file", "nonexistent.txt"])
        assert result.exit_code == 1
        assert "does not exist" in result.stdout
    
    def test_process_file_with_temp_file(self):
        """Test process-file command with temporary file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("   This is some   messy    text   with   extra   spaces.   \\n\\n\\n")
            temp_file = Path(f.name)
        
        try:
            result = runner.invoke(app, ["process-file", str(temp_file)])
            assert result.exit_code == 0
            assert "Processed Content" in result.stdout
        finally:
            temp_file.unlink()  # Clean up
    
    def test_batch_process_nonexistent_directory(self):
        """Test batch-process command with non-existent directory."""
        result = runner.invoke(app, ["batch-process", "nonexistent_dir"])
        assert result.exit_code == 1
        assert "does not exist" in result.stdout
    
    def test_batch_process_empty_directory(self):
        """Test batch-process command with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(app, ["batch-process", temp_dir])
            assert result.exit_code == 0
            assert "No files matching" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_analyze_text_detailed_flag(self):
        """Test analyze-text command with detailed flag."""
        result = runner.invoke(app, ["analyze-text", "--detailed", "This is a comprehensive test."])
        assert result.exit_code == 0
        assert "Text Statistics" in result.stdout
        assert "Detailed Analysis" in result.stdout
    
    def test_process_file_operations(self):
        """Test different process-file operations."""
        # Create a temporary file with markdown content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Header\\n\\n**Bold text** and *italic text*\\n\\n```python\\nprint('hello')\\n```")
            temp_file = Path(f.name)
        
        try:
            # Test extract operation
            result = runner.invoke(app, ["process-file", str(temp_file), "--operation", "extract"])
            assert result.exit_code == 0
            
            # Test clean operation
            result = runner.invoke(app, ["process-file", str(temp_file), "--operation", "clean"])
            assert result.exit_code == 0
            
            # Test format operation
            result = runner.invoke(app, ["process-file", str(temp_file), "--operation", "format"])
            assert result.exit_code == 0
            
        finally:
            temp_file.unlink()  # Clean up


@pytest.fixture
def temp_directory_with_files():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "test1.txt").write_text("This is the first test file.")
        (temp_path / "test2.txt").write_text("This is the second test file with more content.")
        (temp_path / "readme.md").write_text("# Test\\n\\nThis is a markdown file.")
        (temp_path / "binary.exe").write_bytes(b"\\x00\\x01\\x02\\x03")  # Binary file
        
        yield temp_path


def test_batch_process_with_files(temp_directory_with_files):
    """Test batch processing with actual files."""
    result = runner.invoke(app, ["batch-process", str(temp_directory_with_files), "--pattern", "*.txt"])
    assert result.exit_code == 0
    assert "Processing" in result.stdout
    assert "Batch Processing Summary" in result.stdout