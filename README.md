# 🤖 Omni AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/pratyaypaul6187-art/omni-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **safe, ethical AI toolkit** for text analysis, file processing, and automation. This project explicitly prohibits and will not include any harmful functionality, including but not limited to cyber attacks, malware creation, unauthorized access, or any activity that violates laws or ethical guidelines.

## ✨ Features

- 📊 **Text Analysis**: Comprehensive readability analysis, statistics, and keyword extraction
- 📁 **File Processing**: Safe content extraction, cleaning, and formatting for multiple file types
- 🔄 **Batch Operations**: Process multiple files with progress tracking and detailed reports
- 🛡️ **Safety First**: Built-in ethical guidelines and safety checks
- 🎨 **Beautiful CLI**: Rich, colorful output with tables, progress bars, and helpful formatting
- 🧪 **Well Tested**: Comprehensive test suite with 64% coverage

## 🚀 Quick Start

### Requirements
- Python 3.9+
- Windows, macOS, or Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/pratyaypaul6187-art/omni-ai.git
cd omni-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# On macOS/Linux:
source .venv/bin/activate

# Install the package
pip install -e .

# Verify installation
omni-ai --help
```

## 📋 Commands

### Basic Usage

```bash
# Get help
omni-ai --help

# Safety check (view ethical guidelines)
omni-ai safety-check

# Friendly greeting
omni-ai hello [name]
```

### Text Analysis

```bash
# Analyze text directly
omni-ai analyze-text "Your text here"

# Analyze a file
omni-ai analyze-text --file document.txt

# Detailed analysis with additional metrics
omni-ai analyze-text --file document.txt --detailed
```

**Features:**
- Word, character, sentence, and paragraph counts
- Readability scores (Flesch Reading Ease, Flesch-Kincaid Grade Level, Coleman-Liau Index)
- Reading level classification
- Syllable counts and lexical diversity (detailed mode)

### File Processing

```bash
# Clean text file (remove extra whitespace, normalize formatting)
omni-ai process-file document.txt --operation clean

# Extract content from various file types
omni-ai process-file webpage.html --operation extract

# Format text with proper line wrapping
omni-ai process-file messy.txt --operation format --output clean.txt
```

**Supported file types:**
- Plain text (`.txt`)
- Markdown (`.md`, `.markdown`)
- HTML (`.html`, `.htm`)
- Code files (`.py`, `.js`, `.java`, `.cpp`, etc.)

### Batch Processing

```bash
# Analyze all text files in a directory
omni-ai batch-process ./documents --pattern "*.txt" --operation analyze

# Get info about all markdown files
omni-ai batch-process ./docs --pattern "*.md" --operation info

# Clean all text files (creates backups)
omni-ai batch-process ./content --pattern "*.txt" --operation clean
```

## 🛡️ Safety & Ethics

Omni AI is designed with safety and ethics as core principles:

### ✅ Supported Use Cases
- Text analysis and processing for research and education
- Content organization and cleanup
- Document formatting and extraction
- Writing assistance and readability improvement
- Batch file processing for productivity

### ❌ Explicitly Prohibited
- Harmful content generation
- Privacy violations or unauthorized data access
- Malicious code creation or security exploits
- Any illegal or unethical activities

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=omni_ai --cov-report=term-missing

# Run specific test file
pytest tests/test_text_analysis.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
omni-ai/
├── src/omni_ai/
│   ├── __init__.py
│   ├── cli.py              # CLI interface
│   └── core/
│       ├── text_analysis.py    # Text analysis functions
│       ├── file_utils.py       # File processing utilities
│       └── safe_processing.py  # Batch processing functions
├── tests/
│   ├── test_cli.py
│   └── test_text_analysis.py
├── pyproject.toml          # Project configuration
└── README.md
```

## 📊 Examples

### Text Analysis Output

```
$ omni-ai analyze-text --file example.txt

          📊 Text Statistics           
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric                      ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Word Count                  │ 138   │
│ Character Count             │ 912   │
│ Sentence Count              │ 15    │
│ Avg Words per Sentence      │ 9.2   │
└─────────────────────────────┴───────┘

                     📖 Readability Analysis                      
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric               ┃ Score ┃ Interpretation                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Flesch Reading Ease  │ 45.2  │ Difficult (College level)       │
│ Flesch-Kincaid Grade │ 12.1  │ Grade 12 level                  │
└──────────────────────┴───────┴─────────────────────────────────┘
```

## 🤝 Contributing

We welcome contributions that align with our safety-first approach!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `pytest tests/`
6. Submit a pull request

**All contributions must:**
- Adhere to the safety and ethical guidelines
- Include appropriate tests
- Follow the existing code style
- Not introduce any harmful capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Typer](https://typer.tiangolo.com/) for the CLI framework
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- [TextStat](https://github.com/textstat/textstat) for readability analysis
- Tested with [Pytest](https://pytest.org/)

---

**🛡️ Remember: Omni AI is committed to safe, ethical AI practices. Let's build technology that helps, not harms.**
