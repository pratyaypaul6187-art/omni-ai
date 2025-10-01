# 🧠 Omni AI - Advanced Neurosymbolic AI System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/pratyaypaul6187-art/omni-ai.svg)](https://github.com/pratyaypaul6187-art/omni-ai/stargazers)

> **A sophisticated neurosymbolic AI system with enhanced memory, multimodal processing, and Hollywood-style interface**

🎬 **NOW WITH CINEMATIC GUI INTERFACE** • 🧠 **ENHANCED AI CONSCIOUSNESS** • 🔒 **SECURITY FORTRESS**

## ✨ Features

- 🧠 **Enhanced AI Consciousness** with memory integration and multiple thinking modes
- 🎭 **6 Thinking Modes** - Creative, Logical, Analytical, Reflective, Collaborative, Intuitive
- 📚 **Neurosymbolic Reasoning** with knowledge graphs and symbolic logic
- 🎬 **Hollywood-Style GUI** with cinematic interface and fortress control theme
- 🔒 **Security Fortress** with advanced protection and monitoring systems
- 💾 **8-Type Memory System** with million-token contexts and intelligent compression
- 🌍 **Multimodal Processing** supporting text, image, audio, video, sensor, and temporal data
- 🏢 **Scalable Architecture** designed for billions of parameters and distributed processing
- 🧪 **Comprehensive Testing** with extensive test suites and performance validation

## 🚀 Quick Start

### Requirements
- Python 3.9+
- Windows 10/11, macOS, or Linux
- 4GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/pratyaypaul6187-art/omni-ai.git
cd omni-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install GUI dependencies
pip install customtkinter
```

### 🎬 Launch Options

```bash
# Launch the Hollywood-style GUI
python launch_gui.py

# Run simple natural language demo
python simple_nl_demo.py

# Full integration demo
python demo_integration.py

# Advanced reasoning demo
python demo_advanced_reasoning.py
```

## 💻 Usage Examples

### 1. Enhanced AI Consciousness

```python
from src.omni_ai.brain import create_enhanced_consciousness, EnhancedThinkingMode

# Initialize the AI brain
consciousness = await create_enhanced_consciousness(
    memory_db_path="data/my_memory.db",
    enable_neurosymbolic=True
)

# Use different thinking modes
response = await consciousness.enhanced_think(
    "Create a story about robots",
    thinking_mode=EnhancedThinkingMode.CREATIVE
)

print(response['content'])
```

### 2. Natural Language Interface

```python
from src.omni_ai.neurosymbolic.nl_interface import NaturalLanguageInterface

# Quick setup
interface = NaturalLanguageInterface()

# Ask questions
response = await interface.process_query("What is artificial intelligence?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
```

### 3. Memory System

```python
from src.omni_ai.memory.enhanced_memory import EnhancedMemorySystem

# Create memory system
memory = EnhancedMemorySystem("data/memory.db")

# Store memories
await memory.store_memory(
    content="Paris is the capital of France",
    memory_type="semantic",
    importance=0.8
)

# Search memories
results = await memory.search_memories(
    query="capital of France",
    top_k=5
)
```

### 4. Knowledge Management

```python
from src.omni_ai.neurosymbolic.knowledge_manager import KnowledgeManager

# Create knowledge base
km = KnowledgeManager("demo_knowledge_bases")

# Add facts
kb = await km.create_knowledge_base("Science Facts")
await kb.add_fact("Earth", "orbits", "Sun")
await kb.add_fact("Sun", "is_type", "Star")

# Query knowledge
results = await kb.query("What does Earth orbit?")
```

## 🎭 Thinking Modes

Omni AI supports multiple thinking modes for different types of reasoning:

- 🧮 **LOGICAL** - Symbolic reasoning and logical deduction
- 🎨 **CREATIVE** - Creative thinking and storytelling  
- 🔬 **ANALYTICAL** - Deep analysis and problem-solving
- 🪞 **REFLECTIVE** - Self-reflection and introspection
- 🤝 **COLLABORATIVE** - Multi-perspective reasoning
- 💡 **INTUITIVE** - Pattern-based intuitive responses
- ⚡ **ADAPTIVE** - Automatically selects best mode

## 📚 Memory Types

The system uses 8 hierarchical memory types:

1. **Sensory** - Ultra-short term sensory input (milliseconds)
2. **Short-term** - Working memory for active processing (minutes)
3. **Long-term** - Persistent long-term storage (permanent)
4. **Episodic** - Event-based memory with context
5. **Semantic** - Factual knowledge and concepts
6. **Procedural** - Skills and procedures
7. **Autobiographical** - Personal history and experiences
8. **Contextual** - Context-dependent memory

## 🌍 Multimodal Capabilities

Supported modalities:
- 📝 **Text** - Natural language processing
- 🖼️ **Image** - Computer vision and image analysis
- 🔊 **Audio** - Speech and audio processing
- 🎥 **Video** - Video content analysis
- 📡 **Sensor** - IoT and sensor data
- 📋 **Structured** - Databases and structured data
- ⏰ **Temporal** - Time-series data

## 📁 Project Structure

```
omni-ai/
├── 🧠 src/omni_ai/
│   ├── brain/              # AI consciousness and reasoning
│   ├── memory/             # Enhanced memory system
│   ├── neurosymbolic/      # Symbolic reasoning & knowledge
│   ├── multimodal/         # Multi-modal processing
│   ├── security/           # Security and protection
│   └── core/               # Scalable architecture
├── 🎬 gui/                 # Cinematic interface
├── 📊 data/                # Memory databases
├── 🧪 tests/               # Test suites
├── 📚 docs/                # Documentation
└── 🎯 demos/               # Example scripts
```

## 🎬 Hollywood-Style GUI

Launch the cinematic interface:

```bash
python launch_gui.py
```

**GUI Features:**
- 🎭 Animated holographic displays
- 🏰 Fortress control center theme
- 📊 Real-time system monitoring
- 🤖 Interactive AI chat
- 🔒 Security status dashboard
- 🌌 Particle effects and animations

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
