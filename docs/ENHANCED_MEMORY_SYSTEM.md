# 🔹 Enhanced Memory System Documentation

## Overview

The Enhanced Memory System is a sophisticated, biologically-inspired memory architecture that provides hierarchical storage, million-token context windows, and episodic memory formation for AI systems. It bridges the gap between traditional AI memory models and human-like cognitive memory systems.

## Key Features

### 🧠 Hierarchical Memory Types
- **Sensory Memory**: Ultra-short term (milliseconds to seconds) - fast in-memory storage
- **Short-term Memory**: Working memory (seconds to minutes) - fast in-memory storage
- **Long-term Memory**: Persistent memory (hours to years) - SQLite-backed storage
- **Episodic Memory**: Event-based memories with context - SQLite-backed storage
- **Semantic Memory**: Factual knowledge - SQLite-backed storage
- **Procedural Memory**: Skill and process knowledge - SQLite-backed storage
- **Autobiographical Memory**: Personal/system history - SQLite-backed storage
- **Contextual Memory**: Context-dependent information - fast in-memory storage

### 🖥️ Million-Token Context Windows
- Support for context windows up to 1,000,000 tokens
- Intelligent compression based on importance scores
- Automatic summarization for inactive contexts
- Token-aware sliding window management

### 📖 Episodic Event Management
- Structured episodic events with participants, locations, outcomes
- Automatic memory association with ongoing episodes
- Emotional tone tracking and analysis
- Duration and outcome tracking

### 🔄 Background Memory Management
- **Memory Consolidation**: Automatic promotion from short-term to long-term based on importance, access patterns, and emotional weight
- **Forgetting/Decay**: Natural memory decay for low-importance memories
- **Association Strengthening**: Co-accessed memories develop stronger associations
- **Context Management**: Automatic cleanup of inactive contexts

## Architecture Components

### EnhancedMemorySystem
The main orchestration class that manages all memory operations.

```python
from omni_ai.memory import EnhancedMemorySystem, MemoryType

# Initialize with custom configuration
memory_system = EnhancedMemorySystem(
    max_context_tokens=1_000_000,
    consolidation_interval=3600,  # 1 hour
    forgetting_enabled=True
)

# Store a memory
memory_id = await memory_system.store_memory(
    content="Neural networks are computational models...",
    memory_type=MemoryType.SEMANTIC,
    importance=0.9,
    tags={'neural_networks', 'ai_basics'},
    emotional_weight=0.2
)
```

### HierarchicalMemoryStorage
Manages different storage layers optimized for each memory type:
- **In-memory storage** for fast-access memories (sensory, short-term, contextual)
- **SQLite storage** for persistent memories (long-term, episodic, semantic, procedural, autobiographical)

### Memory Indexing and Search
- Full-text search across memory content
- Tag-based filtering and categorization
- Importance and time-based ranking
- Association-aware result re-ranking

### Context Window Management
- Automatic token counting and limit enforcement
- Importance-based compression when limits exceeded
- Background summarization of compressed content
- Multi-context support for parallel conversations

## Integration with AI Consciousness

### MemoryIntegratedConsciousness
Bridges the memory system with AI consciousness cores and multimodal processors:

```python
from omni_ai.memory.memory_integration import MemoryIntegratedConsciousness

# Initialize with consciousness and multimodal components
integrated_ai = MemoryIntegratedConsciousness(
    consciousness_core=your_consciousness,
    multimodal_processor=your_multimodal,
    memory_config={'max_context_tokens': 500_000}
)

# Enhanced thinking with memory context
result = await integrated_ai.think(
    input_data="Tell me about transformers in AI",
    conversation_id="user_123",
    context_tags={'question', 'transformers'}
)
```

### Memory-Enhanced Processing
- **Pre-thought memory retrieval**: Relevant memories are automatically retrieved before processing
- **Context window creation**: Each thought gets its own context window with input and memory content
- **Thought memory storage**: Processing results are stored as memories with rich metadata
- **Episodic event tracking**: Conversations and interactions are tracked as episodic events

## Memory Operations

### Storage Operations
```python
# Store different types of memories
semantic_id = await memory_system.store_memory(
    content="Facts about the world",
    memory_type=MemoryType.SEMANTIC,
    importance=0.8,
    tags={'knowledge', 'facts'}
)

episodic_id = await memory_system.store_memory(
    content="User asked about AI",
    memory_type=MemoryType.EPISODIC,
    importance=0.6,
    tags={'conversation', 'question'},
    context={'user_id': 'user_123', 'timestamp': '2025-01-01T12:00:00'}
)
```

### Retrieval Operations
```python
# Direct memory retrieval
memory = await memory_system.retrieve_memory(memory_id, MemoryType.SEMANTIC)

# Search across memories
results = await memory_system.search_memories(
    query="neural networks transformers",
    memory_types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
    max_results=10,
    importance_threshold=0.5
)
```

### Context Management
```python
# Create context window
context = await memory_system.create_context_window(
    context_id="conversation_123",
    initial_content=["User:", "Hello!", "AI:", "Hi there!"],
    max_tokens=100_000
)

# Add to context
await memory_system.add_to_context(
    context_id="conversation_123",
    content="Let's talk about AI and machine learning",
    importance=0.7
)
```

### Episodic Management
```python
# Start an episode
episode_id = await memory_system.start_episode(
    event_type="learning_session",
    summary="User learning about neural networks",
    participants=['user', 'ai_assistant'],
    location="chat_interface"
)

# Memories stored during episode are automatically associated
await memory_system.store_memory(
    content="Explained backpropagation",
    memory_type=MemoryType.EPISODIC,
    importance=0.7
)

# End episode
episode = await memory_system.end_episode(
    outcomes=['concepts_learned', 'questions_answered'],
    emotional_tone='positive',
    importance=0.8
)
```

## Memory Consolidation Process

The system automatically consolidates memories based on multiple criteria:

### Consolidation Criteria
1. **High importance** (> 0.7)
2. **High access count** (> 3 accesses)
3. **Strong emotional weight** (|weight| > 0.5)
4. **Rich associations** (> 2 associated memories)
5. **Recent access** (within last hour)

Memories meeting 2+ criteria are candidates for consolidation from short-term to long-term storage.

### Forgetting Process
Low-importance memories undergo natural decay:
- Decay rate inversely proportional to importance
- Time-based strength reduction
- Automatic removal when strength < 0.1
- Important memories (importance > 0.8) are protected from forgetting

## Performance Characteristics

### Tested Performance
- **Storage**: 1,000 memories in ~0.5 seconds
- **Retrieval**: 100 memories in ~0.2 seconds
- **Search**: 10 complex queries in ~0.3 seconds
- **Memory overhead**: ~50MB for 10,000 memories
- **Context compression**: Maintains importance-based content when limits exceeded

### Scalability Features
- Hierarchical storage optimized for access patterns
- Background processing doesn't block main operations
- SQLite backend supports millions of memories
- Configurable consolidation and forgetting intervals
- Efficient indexing for fast search and retrieval

## Configuration Options

### Memory System Configuration
```python
config = {
    'max_context_tokens': 1_000_000,  # Maximum tokens per context window
    'consolidation_interval': 3600,   # Memory consolidation frequency (seconds)
    'forgetting_enabled': True,       # Enable automatic forgetting
}
```

### Storage Configuration
- Custom database paths for different memory types
- Configurable in-memory vs persistent storage allocation
- Adjustable background task frequencies
- Memory importance thresholds for various operations

## Error Handling and Resilience

### Robust Error Management
- Graceful degradation when storage is unavailable
- Automatic database creation and migration
- Background task error recovery
- Memory corruption detection and repair

### Data Integrity
- ACID compliance through SQLite transactions
- Automatic backup of critical memory indexes
- Version tracking for memory updates
- Comprehensive logging for debugging

## Usage Examples

### Basic Memory Operations
```python
import asyncio
from omni_ai.memory import EnhancedMemorySystem, MemoryType

async def basic_example():
    # Initialize system
    memory = EnhancedMemorySystem()
    
    # Store knowledge
    memory_id = await memory.store_memory(
        content="Python is a programming language",
        memory_type=MemoryType.SEMANTIC,
        importance=0.8,
        tags={'programming', 'python'}
    )
    
    # Retrieve memory
    retrieved = await memory.retrieve_memory(memory_id, MemoryType.SEMANTIC)
    print(f"Retrieved: {retrieved.content}")
    
    # Search memories
    results = await memory.search_memories(
        query="programming language",
        memory_types=[MemoryType.SEMANTIC]
    )
    print(f"Found {len(results)} related memories")

asyncio.run(basic_example())
```

### Integrated AI System
```python
from omni_ai.memory.memory_integration import MemoryIntegratedConsciousness

async def ai_integration_example():
    # Create integrated AI with memory
    ai = MemoryIntegratedConsciousness(
        consciousness_core=your_consciousness_core,
        multimodal_processor=your_multimodal_processor
    )
    
    # Enhanced thinking with memory context
    response = await ai.think(
        input_data="How do neural networks learn?",
        conversation_id="educational_chat",
        context_tags={'learning', 'neural_networks'}
    )
    
    print(f"AI Response: {response['output']}")
    print(f"Used {response['memories_used']} contextual memories")
    
    # Get conversation summary
    summary = await ai.get_conversation_summary("educational_chat")
    print(f"Conversation had {summary['memories']['total_memories']} memories")

asyncio.run(ai_integration_example())
```

## Advanced Features

### Custom Memory Hooks
```python
async def pre_thought_hook(input_data, memories, context):
    print(f"About to process with {len(memories)} memories")

async def post_thought_hook(result, memories, context):
    print(f"Processed with confidence {result.get('confidence', 0)}")

# Add hooks to memory-integrated consciousness
ai.add_memory_hook('pre_thought', pre_thought_hook)
ai.add_memory_hook('post_thought', post_thought_hook)
```

### Memory Statistics and Monitoring
```python
# Get comprehensive statistics
stats = memory_system.get_memory_statistics()
print(f"Total memories: {stats['total_memories']}")
print(f"Active contexts: {stats['context_statistics']['active_contexts']}")
print(f"Average retrieval time: {stats['average_retrieval_time']:.4f}s")
print(f"Memory consolidations: {stats['consolidations_performed']}")
```

## Best Practices

### Performance Optimization
1. **Use appropriate memory types** - Store frequently accessed data in faster memory types
2. **Set meaningful importance scores** - Guide consolidation and search ranking
3. **Use descriptive tags** - Enable efficient filtering and categorization  
4. **Configure context limits** - Balance context size with processing speed
5. **Monitor background tasks** - Ensure consolidation and forgetting run smoothly

### Memory Management
1. **Regular cleanup** - End conversations to consolidate episodic memories
2. **Meaningful episodes** - Group related memories in episodic events
3. **Balanced importance** - Avoid marking everything as highly important
4. **Rich context** - Include metadata for better memory association
5. **Graceful shutdown** - Clean up background tasks properly

### Integration Guidelines
1. **Memory-first design** - Consider memory implications in AI architecture
2. **Context awareness** - Use conversation IDs and context tags consistently
3. **Error handling** - Handle memory system failures gracefully
4. **Testing** - Test memory operations with realistic data volumes
5. **Monitoring** - Track memory usage and performance metrics

## Troubleshooting

### Common Issues
- **Database lock errors**: Ensure proper cleanup of background tasks
- **Memory growth**: Monitor and tune forgetting thresholds
- **Slow search**: Check indexing and query complexity
- **Context overflow**: Adjust importance scoring for compression
- **Missing memories**: Verify memory type and search parameters

### Debug Tools
- Comprehensive logging with structured output
- Memory statistics for performance monitoring
- Background task status checking
- Database integrity verification tools
- Memory relationship visualization capabilities

This Enhanced Memory System provides a solid foundation for building sophisticated AI systems with human-like memory capabilities, supporting everything from simple knowledge storage to complex episodic reasoning and contextual understanding.