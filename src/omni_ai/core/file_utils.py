"""File processing utilities for safe content manipulation."""

import re
from pathlib import Path
from typing import Union


def process_file(file_path: Path, operation: str) -> str:
    """
    Process a file with the specified operation.
    
    Args:
        file_path: Path to the input file
        operation: Operation to perform (clean, format, extract)
    
    Returns:
        Processed content as string
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            content = file_path.read_text(encoding='latin-1')
        except Exception as e:
            return f"Error reading file: {e}"
    except Exception as e:
        return f"Error reading file: {e}"
    
    if operation == "clean":
        return clean_text(content)
    elif operation == "format":
        return format_text(content)
    elif operation == "extract":
        return extract_content(content, file_path.suffix)
    else:
        return f"Unknown operation: {operation}. Available: clean, format, extract"


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, normalizing line endings, etc.
    
    Args:
        text: Input text to clean
    
    Returns:
        Cleaned text
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove multiple consecutive blank lines
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # Clean up extra spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove trailing spaces from lines
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove leading/trailing whitespace from the entire text
    text = text.strip()
    
    return text


def format_text(text: str) -> str:
    """
    Format text with proper paragraph breaks and basic formatting.
    
    Args:
        text: Input text to format
    
    Returns:
        Formatted text
    """
    # First clean the text
    text = clean_text(text)
    
    # Add proper paragraph spacing
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            # Wrap long lines (simple approach)
            lines = paragraph.split('\n')
            formatted_lines = []
            for line in lines:
                if len(line) > 80:
                    # Simple word wrapping
                    words = line.split()
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 <= 80:
                            current_line.append(word)
                            current_length += len(word) + 1
                        else:
                            if current_line:
                                formatted_lines.append(' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)
                    
                    if current_line:
                        formatted_lines.append(' '.join(current_line))
                else:
                    formatted_lines.append(line)
            
            formatted_paragraphs.append('\n'.join(formatted_lines))
    
    return '\n\n'.join(formatted_paragraphs)


def extract_content(text: str, file_extension: str) -> str:
    """
    Extract meaningful content based on file type.
    
    Args:
        text: Raw file content
        file_extension: File extension to determine extraction method
    
    Returns:
        Extracted content
    """
    if file_extension.lower() in ['.md', '.markdown']:
        return extract_markdown_content(text)
    elif file_extension.lower() in ['.html', '.htm']:
        return extract_html_content(text)
    elif file_extension.lower() in ['.py', '.js', '.java', '.cpp', '.c']:
        return extract_code_content(text)
    else:
        # For plain text files, just clean and return
        return clean_text(text)


def extract_markdown_content(text: str) -> str:
    """Extract readable content from Markdown, removing formatting."""
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove headers formatting but keep the text
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove bold/italic formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    return clean_text(text)


def extract_html_content(text: str) -> str:
    """Extract text content from HTML, removing tags."""
    # Remove script and style elements
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode common HTML entities
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' ',
    }
    
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    
    return clean_text(text)


def extract_code_content(text: str) -> str:
    """Extract documentation and comments from code files."""
    lines = text.split('\n')
    extracted_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Extract single-line comments
        if stripped_line.startswith('//') or stripped_line.startswith('#'):
            comment_text = stripped_line[2:].strip()
            if comment_text and not comment_text.startswith('TODO') and not comment_text.startswith('FIXME'):
                extracted_lines.append(comment_text)
        
        # Extract multi-line comment content (simplified)
        elif '/*' in stripped_line or '"""' in stripped_line or "'''" in stripped_line:
            # This is a simplified extraction - could be improved
            if not stripped_line.startswith('*') and not stripped_line.startswith('*/'):
                cleaned = re.sub(r'/\*+|\*+/|"""|\'\'\'', '', stripped_line).strip()
                if cleaned:
                    extracted_lines.append(cleaned)
    
    result = '\n'.join(extracted_lines)
    return clean_text(result) if result.strip() else "No documentation or comments found."


def get_file_info(file_path: Path) -> dict:
    """
    Get basic information about a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Dictionary with file information
    """
    try:
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size_bytes': stat.st_size,
            'size_kb': round(stat.st_size / 1024, 2),
            'extension': file_path.suffix,
            'modified': stat.st_mtime,
            'is_text_file': is_text_file(file_path),
        }
    except Exception as e:
        return {'error': str(e)}


def is_text_file(file_path: Path) -> bool:
    """
    Check if a file is likely a text file.
    
    Args:
        file_path: Path to check
    
    Returns:
        True if likely a text file
    """
    text_extensions = {
        '.txt', '.md', '.markdown', '.py', '.js', '.html', '.htm', '.css',
        '.json', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf',
        '.log', '.csv', '.tsv', '.sql', '.sh', '.bat', '.ps1',
        '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go',
        '.rs', '.swift', '.kt', '.scala', '.r', '.m', '.pl',
    }
    
    if file_path.suffix.lower() in text_extensions:
        return True
    
    # Additional check by trying to read a small portion
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(512)
            # Check for null bytes (common in binary files)
            if b'\x00' in chunk:
                return False
            # Check if most bytes are printable
            try:
                chunk.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
    except Exception:
        return False