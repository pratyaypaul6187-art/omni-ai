"""
🔹 MEMORY INTEGRATION TEST
Demonstrates the enhanced memory system integrated with AI consciousness and multimodal processing
"""

import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omni_ai.memory.memory_integration import MemoryIntegratedConsciousness
from omni_ai.memory import MemoryType, EnhancedMemorySystem

# Mock AI Consciousness Core for testing
class MockAIConsciousness:
    """Mock AI consciousness for testing integration"""
    
    def __init__(self):
        self.neural_activity = {
            'global_activity': 0.7,
            'attention_focus': 0.8,
            'working_memory_load': 0.6,
            'emotional_state': 0.4
        }
        
    async def process_thought(self, enhanced_input):
        """Process thought with enhanced memory context"""
        
        original_input = enhanced_input['original_input']
        memories = enhanced_input['contextual_memories']
        
        # Simulate thinking process with memory integration
        response = {
            'output': f"Processed '{original_input}' with {len(memories)} contextual memories",
            'output_type': 'text_response',
            'confidence': 0.8,
            'emotional_tone': 'positive',
            'importance': 0.7,
            'processing_steps': [
                'input_analysis',
                'memory_context_integration',
                'response_synthesis'
            ],
            'neural_activity': self.neural_activity,
            'memories_integrated': len(memories)
        }
        
        return response
    
    def get_neural_activity(self):
        """Get current neural activity state"""
        return self.neural_activity.copy()

# Mock Multimodal Processor
class MockMultimodalProcessor:
    """Mock multimodal processor for testing integration"""
    
    async def process_multimodal(self, input_data):
        """Process multimodal input"""
        
        results = {}
        
        for modality, data in input_data.items():
            if modality == 'text':
                results[modality] = {
                    'features': f"Text features: {len(data)} chars",
                    'embedding': np.random.rand(512).astype(np.float32),
                    'stats': {'processing_time': 0.1, 'confidence': 0.9}
                }
            elif modality == 'image':
                results[modality] = {
                    'features': f"Image features: {data.get('size', 'unknown')}",
                    'embedding': np.random.rand(2048).astype(np.float32),
                    'stats': {'processing_time': 0.3, 'confidence': 0.8}
                }
            elif modality == 'audio':
                results[modality] = {
                    'features': f"Audio features: {data.get('duration', 'unknown')}",
                    'embedding': np.random.rand(1024).astype(np.float32),
                    'stats': {'processing_time': 0.2, 'confidence': 0.85}
                }
        
        return results

async def demonstrate_memory_integration():
    """Comprehensive demonstration of memory system integration"""
    
    print("🔹 MEMORY INTEGRATION DEMONSTRATION")
    print("=" * 50)
    
    # Initialize components
    mock_consciousness = MockAIConsciousness()
    mock_multimodal = MockMultimodalProcessor()
    
    # Create memory-integrated consciousness
    memory_config = {
        'max_context_tokens': 50000,  # Smaller for demo
        'consolidation_interval': 60,  # 1 minute for demo
        'forgetting_enabled': True
    }
    
    integrated_consciousness = MemoryIntegratedConsciousness(
        consciousness_core=mock_consciousness,
        multimodal_processor=mock_multimodal,
        memory_config=memory_config
    )
    
    print("✅ Memory-Integrated Consciousness initialized")
    print(f"✅ Max context tokens: {memory_config['max_context_tokens']:,}")
    
    # Test 1: Basic thinking with memory
    print("\n📝 Test 1: Basic Thinking with Memory Integration")
    print("-" * 40)
    
    conversation_id = "demo_conversation_001"
    
    result1 = await integrated_consciousness.think(
        input_data="Hello, I want to learn about artificial intelligence",
        conversation_id=conversation_id,
        context_tags={'learning', 'ai_introduction'}
    )
    
    print(f"🧠 Thought result: {result1['output']}")
    print(f"🔍 Processing time: {result1['processing_time']:.3f}s")
    print(f"💭 Memories used: {result1['memories_used']}")
    
    # Test 2: Follow-up thinking (should use previous context)
    print("\n📝 Test 2: Follow-up Thinking with Context")
    print("-" * 40)
    
    result2 = await integrated_consciousness.think(
        input_data="Can you tell me more about neural networks?",
        conversation_id=conversation_id,
        context_tags={'neural_networks', 'deep_learning'}
    )
    
    print(f"🧠 Follow-up result: {result2['output']}")
    print(f"🔍 Processing time: {result2['processing_time']:.3f}s")
    print(f"💭 Memories used: {result2['memories_used']}")
    
    # Test 3: Multimodal input processing
    print("\n📝 Test 3: Multimodal Input Processing")
    print("-" * 40)
    
    multimodal_input = {
        'text': "This is an image of a neural network diagram",
        'image': {'size': '1024x768', 'format': 'png'},
        'audio': {'duration': '5.2s', 'format': 'wav'}
    }
    
    multimodal_result = await integrated_consciousness.process_multimodal_input(
        input_data=multimodal_input,
        conversation_id=conversation_id
    )
    
    print(f"🎭 Multimodal result: {multimodal_result['output']}")
    print(f"🔍 Processing time: {multimodal_result['processing_time']:.3f}s")
    
    # Test 4: Store semantic knowledge
    print("\n📝 Test 4: Storing Semantic Knowledge")
    print("-" * 40)
    
    knowledge_items = [
        {
            'content': "Neural networks are computational models inspired by biological neural networks",
            'importance': 0.9,
            'tags': {'neural_networks', 'definition', 'ai_basics'}
        },
        {
            'content': "Deep learning uses multi-layer neural networks to learn complex patterns",
            'importance': 0.8,
            'tags': {'deep_learning', 'neural_networks', 'machine_learning'}
        },
        {
            'content': "Transformers are a type of neural network architecture used in NLP",
            'importance': 0.7,
            'tags': {'transformers', 'nlp', 'neural_networks'}
        }
    ]
    
    for item in knowledge_items:
        memory_id = await integrated_consciousness.memory_system.store_memory(
            content=item['content'],
            memory_type=MemoryType.SEMANTIC,
            importance=item['importance'],
            tags=item['tags']
        )
        print(f"📚 Stored knowledge: {memory_id[:8]}... - {item['content'][:50]}...")
    
    # Test 5: Query with stored knowledge
    print("\n📝 Test 5: Thinking with Stored Knowledge")
    print("-" * 40)
    
    knowledge_query_result = await integrated_consciousness.think(
        input_data="What are transformers in AI?",
        conversation_id=conversation_id,
        context_tags={'transformers', 'question'}
    )
    
    print(f"🧠 Knowledge-enhanced result: {knowledge_query_result['output']}")
    print(f"💭 Memories used: {knowledge_query_result['memories_used']}")
    
    # Test 6: Memory statistics and conversation summary
    print("\n📝 Test 6: Memory Statistics & Conversation Summary")
    print("-" * 40)
    
    memory_stats = integrated_consciousness.memory_system.get_memory_statistics()
    print(f"📊 Total memories: {memory_stats['total_memories']}")
    print(f"📊 Active contexts: {memory_stats['context_statistics']['active_contexts']}")
    print(f"📊 Memory retrievals: {memory_stats['memory_retrievals']}")
    print(f"📊 Avg retrieval time: {memory_stats['average_retrieval_time']:.4f}s")
    
    conversation_summary = await integrated_consciousness.get_conversation_summary(conversation_id)
    print(f"💬 Conversation duration: {conversation_summary['context']['duration']}")
    print(f"💬 Total memories: {conversation_summary['memories']['total_memories']}")
    print(f"💬 Total thoughts: {conversation_summary['thought_processes']['total_thoughts']}")
    
    # Test 7: Episodic memory demonstration
    print("\n📝 Test 7: Episodic Memory Episodes")
    print("-" * 40)
    
    # Manual episode creation for demonstration
    episode_id = await integrated_consciousness.memory_system.start_episode(
        event_type="learning_session",
        summary="User learning about AI and neural networks",
        participants=['user', 'ai_assistant'],
        location="educational_conversation"
    )
    
    print(f"📖 Started learning episode: {episode_id[:8]}...")
    
    # Add some learning activities
    learning_activities = [
        "Explained basic neural network concepts",
        "Discussed deep learning applications",
        "Demonstrated transformer architecture knowledge"
    ]
    
    for activity in learning_activities:
        await integrated_consciousness.memory_system.store_memory(
            content=f"Learning activity: {activity}",
            memory_type=MemoryType.EPISODIC,
            importance=0.6,
            tags={'learning', 'episode_activity'}
        )
    
    # End the episode
    ended_episode = await integrated_consciousness.memory_system.end_episode(
        outcomes=['concepts_learned', 'questions_answered', 'knowledge_transferred'],
        emotional_tone='positive',
        importance=0.8
    )
    
    print(f"📖 Episode ended. Duration: {ended_episode.duration}")
    print(f"📖 Outcomes: {', '.join(ended_episode.outcomes)}")
    
    # Test 8: Memory consolidation simulation
    print("\n📝 Test 8: Memory Consolidation Process")
    print("-" * 40)
    
    # Create some short-term memories that should be consolidated
    short_term_memories = []
    for i in range(5):
        memory_id = await integrated_consciousness.memory_system.store_memory(
            content=f"Important short-term insight {i}: Neural networks learn through backpropagation",
            memory_type=MemoryType.SHORT_TERM,
            importance=0.8,
            tags={'insights', 'important', f'insight_{i}'}
        )
        short_term_memories.append(memory_id)
        
        # Access memories multiple times to increase consolidation probability
        for _ in range(3):
            await integrated_consciousness.memory_system.retrieve_memory(memory_id, MemoryType.SHORT_TERM)
    
    print(f"📝 Created {len(short_term_memories)} short-term memories")
    
    # Manually trigger consolidation
    consolidations_before = integrated_consciousness.memory_system.metrics['consolidations_performed']
    await integrated_consciousness.memory_system._perform_memory_consolidation()
    consolidations_after = integrated_consciousness.memory_system.metrics['consolidations_performed']
    
    print(f"🔄 Consolidations performed: {consolidations_after - consolidations_before}")
    
    # Test 9: End conversation and final summary
    print("\n📝 Test 9: Conversation End & Final Summary")
    print("-" * 40)
    
    final_summary = await integrated_consciousness.end_conversation(conversation_id)
    
    print(f"👋 Conversation ended: {conversation_id}")
    print(f"📊 Final memory count: {final_summary['memory_statistics']['total_memories']}")
    print(f"📊 Total thoughts processed: {final_summary['thought_processes']['total_thoughts']}")
    
    # Display some important memories from the conversation
    important_memories = final_summary['memories']['important_memories'][:3]
    print(f"\n🌟 Top Important Memories:")
    for i, mem in enumerate(important_memories, 1):
        print(f"  {i}. [{mem['importance']:.2f}] {mem['content'][:60]}...")
    
    # Clean up background tasks
    print("\n🔧 Cleaning up background tasks...")
    for task in integrated_consciousness.memory_system.background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    print("\n✅ Memory Integration Demonstration Complete!")
    return integrated_consciousness

async def run_quick_memory_test():
    """Quick memory system functionality test"""
    
    print("🔹 QUICK MEMORY SYSTEM TEST")
    print("=" * 30)
    
    # Basic memory system test
    memory_system = EnhancedMemorySystem(
        max_context_tokens=10000,
        consolidation_interval=3600,
        forgetting_enabled=False  # Disable for quick test
    )
    
    # Store some test memories
    test_memories = [
        ("Python is a programming language", MemoryType.SEMANTIC, 0.8, {'programming', 'python'}),
        ("I learned about neural networks today", MemoryType.EPISODIC, 0.7, {'learning', 'neural_networks'}),
        ("Remember to practice coding daily", MemoryType.PROCEDURAL, 0.6, {'habits', 'coding'}),
        ("This conversation started well", MemoryType.AUTOBIOGRAPHICAL, 0.5, {'conversation', 'positive'})
    ]
    
    memory_ids = []
    for content, mem_type, importance, tags in test_memories:
        memory_id = await memory_system.store_memory(
            content=content,
            memory_type=mem_type,
            importance=importance,
            tags=tags
        )
        memory_ids.append(memory_id)
        print(f"✅ Stored: {mem_type.value} - {content[:30]}...")
    
    # Test memory retrieval
    print(f"\n🔍 Testing Memory Retrieval:")
    for i, (memory_id, (content, mem_type, _, _)) in enumerate(zip(memory_ids, test_memories)):
        retrieved = await memory_system.retrieve_memory(memory_id, mem_type)
        if retrieved:
            print(f"  ✅ Retrieved: {retrieved.content[:40]}...")
        else:
            print(f"  ❌ Failed to retrieve memory {i}")
    
    # Test memory search
    print(f"\n🔎 Testing Memory Search:")
    search_results = await memory_system.search_memories(
        query="programming neural networks",
        memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
        max_results=5
    )
    
    print(f"  Found {len(search_results)} matching memories:")
    for result in search_results:
        print(f"    - [{result.importance:.2f}] {result.content[:50]}...")
    
    # Test context window
    print(f"\n🖥️ Testing Context Window:")
    context = await memory_system.create_context_window(
        context_id="test_context",
        initial_content=["Hello", "world", "this", "is", "a", "test"],
        max_tokens=100
    )
    
    print(f"  ✅ Created context: {context.current_tokens} tokens")
    
    # Add content to context
    await memory_system.add_to_context(
        context_id="test_context",
        content="Additional content for context testing",
        importance=0.7
    )
    
    updated_context = memory_system.active_contexts["test_context"]
    print(f"  ✅ Updated context: {updated_context.current_tokens} tokens")
    
    # Get final statistics
    stats = memory_system.get_memory_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Active contexts: {stats['context_statistics']['active_contexts']}")
    print(f"  Memory retrievals: {stats['memory_retrievals']}")
    
    # Cleanup
    for task in memory_system.background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    print("✅ Quick Memory Test Complete!")

if __name__ == "__main__":
    print("Starting memory integration tests...\n")
    
    # Run quick test first
    asyncio.run(run_quick_memory_test())
    
    print("\n" + "="*60 + "\n")
    
    # Run full integration demonstration
    asyncio.run(demonstrate_memory_integration())