# 🎯 Omni AI Usage Guide

This comprehensive guide will walk you through using all the advanced features of the Omni AI system.

## 📋 Table of Contents

1. [Basic Setup](#basic-setup)
2. [Enhanced AI Consciousness](#enhanced-ai-consciousness)
3. [Thinking Modes](#thinking-modes)
4. [Memory System](#memory-system)
5. [Neurosymbolic Reasoning](#neurosymbolic-reasoning)
6. [Knowledge Management](#knowledge-management)
7. [Multimodal Processing](#multimodal-processing)
8. [Security Features](#security-features)
9. [GUI Interface](#gui-interface)
10. [Advanced Examples](#advanced-examples)

## 🛠️ Basic Setup

### 1. Environment Setup

```bash
# Clone and setup
git clone https://github.com/pratyaypaul6187-art/omni-ai.git
cd omni-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install customtkinter  # For GUI
```

### 2. Test Installation

```bash
# Run simple demo
python simple_nl_demo.py

# Should output:
# 🔸 NATURAL LANGUAGE INTERFACE DEMO
# ==================================================
# 🚀 Initializing...
# 📚 Adding some basic knowledge...
# ✅ Ready! Try these queries:
```

## 🧠 Enhanced AI Consciousness

### Basic Usage

```python
import asyncio
from src.omni_ai.brain import create_enhanced_consciousness

async def basic_example():
    # Initialize consciousness
    consciousness = await create_enhanced_consciousness(
        memory_db_path="data/example_memory.db",
        enable_neurosymbolic=True
    )
    
    # Simple interaction
    response = await consciousness.enhanced_think("Hello, how are you?")
    print(response['content'])
    print(f"Confidence: {response['confidence']}")

# Run
asyncio.run(basic_example())
```

### Advanced Configuration

```python
import asyncio
from src.omni_ai.brain import create_enhanced_consciousness
from src.omni_ai.brain.enhanced_consciousness import ConsciousnessConfig

async def advanced_example():
    # Custom configuration
    config = ConsciousnessConfig(
        memory_enabled=True,
        neurosymbolic_enabled=True,
        multimodal_enabled=True,
        security_enabled=True,
        max_context_tokens=1000000,
        background_processing=True
    )
    
    consciousness = await create_enhanced_consciousness(
        memory_db_path="data/advanced_memory.db",
        config=config
    )
    
    # Get system statistics
    stats = consciousness.get_enhanced_statistics()
    print(f"Enhanced thoughts: {stats['enhanced_thoughts']}")
    print(f"Memory entries: {stats['memory_statistics']['total_memories']}")

asyncio.run(advanced_example())
```

## 🎭 Thinking Modes

### Using Different Thinking Modes

```python
import asyncio
from src.omni_ai.brain import create_enhanced_consciousness, EnhancedThinkingMode

async def thinking_modes_example():
    consciousness = await create_enhanced_consciousness(enable_neurosymbolic=True)
    
    # Creative thinking
    creative_response = await consciousness.enhanced_think(
        "Write a short story about AI",
        thinking_mode=EnhancedThinkingMode.CREATIVE
    )
    print("Creative response:", creative_response['content'])
    
    # Logical thinking
    logical_response = await consciousness.enhanced_think(
        "If all birds can fly and penguins are birds, can penguins fly?",
        thinking_mode=EnhancedThinkingMode.LOGICAL
    )
    print("Logical response:", logical_response['content'])
    
    # Analytical thinking
    analytical_response = await consciousness.enhanced_think(
        "Analyze the pros and cons of renewable energy",
        thinking_mode=EnhancedThinkingMode.ANALYTICAL
    )
    print("Analytical response:", analytical_response['content'])

asyncio.run(thinking_modes_example())
```

### Adaptive Mode Selection

```python
async def adaptive_thinking_example():
    consciousness = await create_enhanced_consciousness(enable_neurosymbolic=True)
    
    # Let the AI choose the best thinking mode
    questions = [
        "What is 2+2?",  # Will likely use LOGICAL
        "Tell me a joke",  # Will likely use CREATIVE
        "How does photosynthesis work?",  # Will likely use ANALYTICAL
        "What do you think about consciousness?"  # Will likely use REFLECTIVE
    ]
    
    for question in questions:
        response = await consciousness.enhanced_think(
            question,
            thinking_mode=EnhancedThinkingMode.ADAPTIVE  # Let AI choose
        )
        print(f"Q: {question}")
        print(f"A: {response['content']}")
        print(f"Mode used: {response.get('reasoning_type', 'unknown')}")
        print("---")

asyncio.run(adaptive_thinking_example())
```

## 💾 Memory System

### Basic Memory Operations

```python
import asyncio
from src.omni_ai.memory.enhanced_memory import EnhancedMemorySystem

async def memory_example():
    # Initialize memory system
    memory = EnhancedMemorySystem("data/example_memory.db")
    
    # Store different types of memories
    await memory.store_memory(
        content="The capital of France is Paris",
        memory_type="semantic",
        importance=0.8,
        tags=["geography", "capitals", "france"]
    )
    
    await memory.store_memory(
        content="Had lunch with John at the Italian restaurant",
        memory_type="episodic",
        importance=0.6,
        context={"location": "Italian restaurant", "person": "John"}
    )
    
    await memory.store_memory(
        content="How to ride a bicycle: balance, pedal, steer",
        memory_type="procedural",
        importance=0.7,
        tags=["skills", "cycling"]
    )
    
    # Search memories
    results = await memory.search_memories(
        query="capital",
        memory_types=["semantic"],
        top_k=5
    )
    
    for result in results:
        print(f"Memory: {result.content}")
        print(f"Type: {result.memory_type}")
        print(f"Confidence: {result.confidence}")

asyncio.run(memory_example())
```

### Advanced Memory Features

```python
async def advanced_memory_example():
    memory = EnhancedMemorySystem("data/advanced_memory.db")
    
    # Create episodic event
    event_id = await memory.start_episodic_event(
        title="Learning about AI",
        context={"location": "home", "activity": "studying"},
        participants=["user", "omni_ai"]
    )
    
    # Add memories to the episode
    await memory.store_memory(
        content="Learned about neural networks",
        memory_type="episodic",
        importance=0.8,
        episode_id=event_id
    )
    
    await memory.store_memory(
        content="Discussed machine learning algorithms", 
        memory_type="episodic",
        importance=0.7,
        episode_id=event_id
    )
    
    # End the episode
    await memory.end_episodic_event(
        event_id,
        outcome="Successfully learned AI concepts",
        emotional_tone=0.8
    )
    
    # Retrieve episode memories
    episode_memories = await memory.get_episode_memories(event_id)
    print(f"Episode contains {len(episode_memories)} memories")

asyncio.run(advanced_memory_example())
```

### Memory Context Windows

```python
async def context_window_example():
    memory = EnhancedMemorySystem("data/context_memory.db")
    
    # Create a large context
    large_context = "This is a very large context with lots of information. " * 1000
    
    # Store with automatic compression
    context_id = await memory.create_context_window(
        context=large_context,
        max_tokens=10000,  # Will compress if larger
        compression_strategy="importance_based"
    )
    
    # Retrieve compressed context
    compressed_context = await memory.get_context_window(context_id)
    print(f"Original size: {len(large_context)} chars")
    print(f"Compressed size: {len(compressed_context)} chars")
    print(f"Compression ratio: {len(compressed_context)/len(large_context):.2%}")

asyncio.run(context_window_example())
```

## 🧮 Neurosymbolic Reasoning

### Basic Symbolic Reasoning

```python
import asyncio
from src.omni_ai.neurosymbolic.symbolic_reasoner import SymbolicReasoner

async def symbolic_reasoning_example():
    reasoner = SymbolicReasoner()
    
    # Add facts
    reasoner.add_fact("Socrates", "is_a", "human")
    reasoner.add_fact("Plato", "is_a", "human")
    
    # Add rules
    reasoner.add_rule("IF ?x is_a human THEN ?x is_mortal")
    
    # Query the system
    result = await reasoner.query("?x is_mortal")
    print("Who is mortal?")
    for binding in result.bindings:
        print(f"- {binding['?x']}")

asyncio.run(symbolic_reasoning_example())
```

### Neural-Symbolic Bridge

```python
import asyncio
from src.omni_ai.neurosymbolic.neural_symbolic_bridge import NeuralSymbolicBridge
from src.omni_ai.neurosymbolic.symbolic_reasoner import SymbolicReasoner
from src.omni_ai.neurosymbolic.knowledge_graph import KnowledgeGraph

async def neural_symbolic_example():
    # Initialize components
    reasoner = SymbolicReasoner()
    knowledge_graph = KnowledgeGraph()
    bridge = NeuralSymbolicBridge(reasoner, knowledge_graph)
    
    # Add knowledge
    await bridge.add_knowledge("Birds can fly")
    await bridge.add_knowledge("Penguins are birds")
    await bridge.add_knowledge("Penguins cannot fly")
    
    # Hybrid reasoning query
    result = await bridge.hybrid_reasoning(
        query="Can penguins fly?",
        use_neural=True,
        use_symbolic=True,
        confidence_threshold=0.7
    )
    
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning type: {result.reasoning_type}")

asyncio.run(neural_symbolic_example())
```

## 📚 Knowledge Management

### Creating Knowledge Bases

```python
import asyncio
from src.omni_ai.neurosymbolic.knowledge_manager import KnowledgeManager

async def knowledge_management_example():
    # Initialize knowledge manager
    km = KnowledgeManager("demo_knowledge_bases")
    
    # Create a science knowledge base
    science_kb = await km.create_knowledge_base(
        name="Science Facts",
        description="Basic scientific facts and relationships"
    )
    
    # Add facts
    await science_kb.add_fact("Earth", "orbits", "Sun")
    await science_kb.add_fact("Sun", "is_type", "Star")
    await science_kb.add_fact("Moon", "orbits", "Earth")
    await science_kb.add_fact("Water", "consists_of", "H2O")
    
    # Add rules
    await science_kb.add_rule("IF ?planet orbits ?star AND ?star is_type Star THEN ?planet is_in solar_system")
    
    # Query knowledge
    results = await science_kb.query("What orbits the Earth?")
    print("What orbits the Earth?")
    for result in results:
        print(f"- {result}")
    
    # Complex reasoning
    planets = await science_kb.query("?x is_in solar_system")
    print("What is in the solar system?")
    for planet in planets:
        print(f"- {planet}")

asyncio.run(knowledge_management_example())
```

### Knowledge Graph Operations

```python
async def knowledge_graph_example():
    km = KnowledgeManager("demo_knowledge_bases")
    kb = await km.create_knowledge_base("Animal Kingdom")
    
    # Build animal taxonomy
    animals = [
        ("Dog", "is_a", "Mammal"),
        ("Cat", "is_a", "Mammal"), 
        ("Eagle", "is_a", "Bird"),
        ("Penguin", "is_a", "Bird"),
        ("Mammal", "is_a", "Animal"),
        ("Bird", "is_a", "Animal"),
        ("Dog", "has_trait", "Loyalty"),
        ("Cat", "has_trait", "Independence"),
        ("Eagle", "can", "Fly"),
        ("Bird", "has", "Feathers")
    ]
    
    # Add all facts
    for subject, predicate, obj in animals:
        await kb.add_fact(subject, predicate, obj)
    
    # Query the graph
    mammals = await kb.query("?x is_a Mammal")
    print("Mammals:", mammals)
    
    animals_that_fly = await kb.query("?x can Fly")
    print("Animals that can fly:", animals_that_fly)
    
    # Get knowledge graph visualization
    graph_data = await kb.export_as_graph()
    print(f"Graph has {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")

asyncio.run(knowledge_graph_example())
```

## 🌍 Multimodal Processing

### Basic Multimodal Usage

```python
import asyncio
from src.omni_ai.multimodal.multimodal_processor import MultimodalProcessor

async def multimodal_example():
    processor = MultimodalProcessor()
    
    # Process text
    text_result = await processor.process_modality(
        data="This is a sample text for processing",
        modality_type="text",
        context={"language": "english", "domain": "general"}
    )
    print(f"Text embedding shape: {text_result.embedding.shape}")
    
    # Simulate image processing
    image_result = await processor.process_modality(
        data="path/to/image.jpg",  # In real usage, this would be image data
        modality_type="image",
        context={"format": "jpg", "resolution": "1024x768"}
    )
    print(f"Image embedding shape: {image_result.embedding.shape}")
    
    # Cross-modal similarity
    similarity = await processor.cross_modal_similarity(
        text_result.embedding,
        image_result.embedding
    )
    print(f"Cross-modal similarity: {similarity}")

asyncio.run(multimodal_example())
```

### Advanced Multimodal Fusion

```python
async def multimodal_fusion_example():
    processor = MultimodalProcessor()
    
    # Process multiple modalities
    modalities = [
        ("This is a beautiful sunset", "text"),
        ("sunset_image.jpg", "image"),
        ("nature_sounds.wav", "audio")
    ]
    
    embeddings = []
    for data, modality_type in modalities:
        result = await processor.process_modality(data, modality_type)
        embeddings.append(result.embedding)
    
    # Fuse embeddings using attention
    fused_embedding = await processor.attention_fusion(embeddings)
    print(f"Fused embedding shape: {fused_embedding.shape}")
    
    # Use fused embedding for reasoning
    reasoning_result = await processor.multimodal_reasoning(
        fused_embedding,
        query="What is the mood of this scene?",
        context={"task": "mood_detection"}
    )
    print(f"Reasoning result: {reasoning_result}")

asyncio.run(multimodal_fusion_example())
```

## 🔒 Security Features

### Security Monitoring

```python
import asyncio
from src.omni_ai.security.monitoring import SecurityMonitor
from src.omni_ai.security.hardening import SecurityHardening

async def security_example():
    # Initialize security
    monitor = SecurityMonitor()
    hardening = SecurityHardening()
    
    # Apply security hardening
    hardening.apply_all_hardening()
    print("Security hardening applied")
    
    # Start monitoring
    await monitor.start_monitoring()
    print("Security monitoring started")
    
    # Simulate some operations
    for i in range(10):
        await asyncio.sleep(1)
        # Security monitor runs in background
        
    # Get security status
    status = await monitor.get_security_status()
    print(f"Security level: {status['security_level']}")
    print(f"Threats detected: {len(status['threats'])}")
    print(f"System health: {status['system_health']}")
    
    # Stop monitoring
    await monitor.stop_monitoring()

asyncio.run(security_example())
```

### Input Validation

```python
from src.omni_ai.security.input_validation import InputValidator

def input_validation_example():
    validator = InputValidator()
    
    # Test various inputs
    test_inputs = [
        "Hello, how are you?",  # Safe
        "<script>alert('xss')</script>",  # Potentially dangerous
        "SELECT * FROM users",  # SQL-like
        "rm -rf /",  # Dangerous command
        "What is 2+2?"  # Safe
    ]
    
    for input_text in test_inputs:
        result = validator.validate_input(input_text)
        print(f"Input: {input_text}")
        print(f"Safe: {result.is_safe}")
        if not result.is_safe:
            print(f"Threats: {result.threats}")
        print("---")

input_validation_example()
```

## 🎬 GUI Interface

### Launching the GUI

```bash
python launch_gui.py
```

The GUI provides:
- **Holographic Displays**: Animated particle effects
- **AI Chat Interface**: Interactive conversation with the AI
- **System Monitoring**: Real-time performance metrics
- **Knowledge Browser**: Explore knowledge bases
- **Security Dashboard**: Monitor security status
- **Memory Viewer**: Browse stored memories

### GUI Features

1. **Main Chat Interface**
   - Natural language conversation
   - Thinking mode selection
   - Response confidence indicators

2. **System Monitor**
   - CPU and memory usage
   - AI processing statistics
   - Security status indicators

3. **Knowledge Management**
   - Browse knowledge bases
   - Add new facts and rules
   - Visualize knowledge graphs

4. **Memory Browser**
   - Search memories by type
   - View episodic events
   - Memory importance ratings

## 🎯 Advanced Examples

### Complete AI Assistant

```python
import asyncio
from src.omni_ai.brain import create_enhanced_consciousness
from src.omni_ai.neurosymbolic.knowledge_manager import KnowledgeManager

class OmniAIAssistant:
    def __init__(self):
        self.consciousness = None
        self.knowledge_manager = None
    
    async def initialize(self):
        # Initialize AI consciousness
        self.consciousness = await create_enhanced_consciousness(
            memory_db_path="data/assistant_memory.db",
            enable_neurosymbolic=True
        )
        
        # Initialize knowledge management
        self.knowledge_manager = KnowledgeManager("assistant_knowledge")
        
        # Load basic knowledge
        await self._load_basic_knowledge()
    
    async def _load_basic_knowledge(self):
        kb = await self.knowledge_manager.create_knowledge_base("General Knowledge")
        
        # Add some basic facts
        facts = [
            ("Python", "is_a", "Programming Language"),
            ("AI", "includes", "Machine Learning"),
            ("Machine Learning", "includes", "Neural Networks")
        ]
        
        for subject, predicate, obj in facts:
            await kb.add_fact(subject, predicate, obj)
    
    async def chat(self, message: str):
        # Process with AI consciousness
        response = await self.consciousness.enhanced_think(
            message,
            context={"user_message": True, "timestamp": "now"}
        )
        
        # Learn from the interaction
        await self.consciousness.learn_from_interaction(
            interaction=message,
            response=response['content']
        )
        
        return response['content']
    
    async def ask_knowledge_question(self, question: str):
        # Use knowledge base for factual queries
        kb = await self.knowledge_manager.get_knowledge_base("General Knowledge")
        results = await kb.query(question)
        return results

# Usage
async def assistant_example():
    assistant = OmniAIAssistant()
    await assistant.initialize()
    
    # Chat with the assistant
    response = await assistant.chat("Tell me about Python programming")
    print(f"Assistant: {response}")
    
    # Ask knowledge-based questions
    knowledge = await assistant.ask_knowledge_question("?x is_a Programming Language")
    print(f"Programming languages: {knowledge}")

asyncio.run(assistant_example())
```

### Interactive Learning Session

```python
async def learning_session_example():
    consciousness = await create_enhanced_consciousness(enable_neurosymbolic=True)
    
    # Start a learning session
    print("🎓 Starting Interactive Learning Session")
    print("Type 'quit' to end the session")
    
    session_memories = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        # Process the input
        response = await consciousness.enhanced_think(
            user_input,
            thinking_mode=EnhancedThinkingMode.COLLABORATIVE
        )
        
        print(f"AI: {response['content']}")
        
        # Store the interaction
        session_memories.append({
            'user': user_input,
            'ai': response['content'],
            'confidence': response['confidence']
        })
        
        # Learn from the interaction
        await consciousness.learn_from_interaction(user_input, response['content'])
    
    # Session summary
    print("\n📊 Session Summary:")
    print(f"Total interactions: {len(session_memories)}")
    avg_confidence = sum(m['confidence'] for m in session_memories) / len(session_memories)
    print(f"Average confidence: {avg_confidence:.2f}")
    
    # Get final statistics
    stats = consciousness.get_enhanced_statistics()
    print(f"New memories created: {stats['memory_statistics']['total_memories']}")

# asyncio.run(learning_session_example())  # Uncomment to run
```

## 🚀 Performance Tips

1. **Memory Management**
   - Use appropriate memory types for different content
   - Set importance levels to control retention
   - Enable background consolidation

2. **Thinking Modes**
   - Use ADAPTIVE mode when unsure
   - LOGICAL mode for factual queries
   - CREATIVE mode for open-ended tasks

3. **Knowledge Bases**
   - Organize facts into separate knowledge bases
   - Use meaningful predicates and entities
   - Regular exports for backup

4. **Security**
   - Enable monitoring for production use
   - Apply security hardening
   - Validate all user inputs

## 🔧 Troubleshooting

### Common Issues

1. **Memory Database Locked**
   ```python
   # Solution: Use different database files for concurrent access
   memory1 = EnhancedMemorySystem("data/memory1.db")
   memory2 = EnhancedMemorySystem("data/memory2.db")
   ```

2. **GUI Not Loading**
   ```bash
   # Install GUI dependencies
   pip install customtkinter
   pip install pillow
   ```

3. **Slow Performance**
   ```python
   # Enable caching
   processor = MultimodalProcessor(enable_cache=True)
   
   # Reduce context window size
   memory = EnhancedMemorySystem("memory.db", max_context_tokens=100000)
   ```

---

This guide covers the major features of Omni AI. For more examples, check the demo files in the repository!