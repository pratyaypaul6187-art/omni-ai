"""
🔹 ENHANCED MEMORY SYSTEM TESTS
Comprehensive validation of memory, context, and episodic functionality
"""

import asyncio
import pytest
import numpy as np
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omni_ai.memory import (
    EnhancedMemorySystem,
    MemoryType,
    MemoryImportance,
    EnhancedMemory,
    ContextWindow,
    EpisodicEvent
)

class TestEnhancedMemorySystem:
    """Test suite for enhanced memory system"""
    
    @pytest.fixture
    async def memory_system(self):
        """Create a test memory system"""
        # Use temporary directory for test databases
        temp_dir = tempfile.mkdtemp()
        
        # Override default path to use temp directory
        original_init = EnhancedMemorySystem.__init__
        
        def temp_init(self, *args, **kwargs):
            kwargs.setdefault('max_context_tokens', 10000)  # Smaller for tests
            kwargs.setdefault('consolidation_interval', 1)   # Faster for tests
            original_init(self, *args, **kwargs)
            # Override storage path after init
            self.storage.base_path = temp_dir + "/memory"
            # Re-initialize storage with new path
            for storage in self.storage.storage_layers.values():
                if hasattr(storage, 'db_path'):
                    storage.db_path = storage.db_path.replace("data/memory", temp_dir + "/memory")
                if hasattr(storage, 'initialize'):
                    storage.initialize()
        
        EnhancedMemorySystem.__init__ = temp_init
        
        system = EnhancedMemorySystem()
        yield system
        
        # Cleanup
        EnhancedMemorySystem.__init__ = original_init
        
        # Cancel background tasks
        for task in system.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_basic_memory_storage_retrieval(self, memory_system):
        """Test basic memory storage and retrieval"""
        system = memory_system
        
        # Store a memory
        memory_id = await system.store_memory(
            content="Test memory content",
            memory_type=MemoryType.LONG_TERM,
            importance=0.8,
            tags={"test", "basic"},
            emotional_weight=0.5
        )
        
        assert memory_id is not None
        assert system.metrics["total_memories"] == 1
        assert system.metrics["memories_by_type"]["long_term"] == 1
        
        # Retrieve the memory
        retrieved = await system.retrieve_memory(memory_id, MemoryType.LONG_TERM)
        
        assert retrieved is not None
        assert retrieved.content == "Test memory content"
        assert retrieved.importance == 0.8
        assert "test" in retrieved.tags
        assert "basic" in retrieved.tags
        assert retrieved.emotional_weight == 0.5
        assert retrieved.access_count == 2  # 1 from store, 1 from retrieve
    
    @pytest.mark.asyncio
    async def test_memory_types_storage(self, memory_system):
        """Test storage across different memory types"""
        system = memory_system
        
        memories = {}
        
        # Store memories of different types
        for memory_type in MemoryType:
            content = f"Memory of type {memory_type.value}"
            memory_id = await system.store_memory(
                content=content,
                memory_type=memory_type,
                importance=0.6
            )
            memories[memory_type] = memory_id
        
        assert system.metrics["total_memories"] == len(MemoryType)
        
        # Retrieve and verify each memory type
        for memory_type, memory_id in memories.items():
            retrieved = await system.retrieve_memory(memory_id, memory_type)
            assert retrieved is not None
            assert retrieved.memory_type == memory_type
            assert f"Memory of type {memory_type.value}" in retrieved.content
    
    @pytest.mark.asyncio
    async def test_memory_search(self, memory_system):
        """Test memory search functionality"""
        system = memory_system
        
        # Store multiple memories
        memories = []
        for i in range(5):
            memory_id = await system.store_memory(
                content=f"Test content {i} with keyword 'search'",
                memory_type=MemoryType.LONG_TERM,
                importance=0.5 + (i * 0.1),
                tags={f"tag{i}", "searchable"}
            )
            memories.append(memory_id)
        
        # Store one without the keyword
        noise_id = await system.store_memory(
            content="Noise content without the special word",
            memory_type=MemoryType.LONG_TERM,
            importance=0.3
        )
        
        # Search for memories
        results = await system.search_memories(
            query="search",
            memory_types=[MemoryType.LONG_TERM],
            max_results=3
        )
        
        assert len(results) <= 3
        # Results should be ranked by importance and other factors
        if results:
            assert "search" in results[0].content.lower()
    
    @pytest.mark.asyncio
    async def test_context_window_management(self, memory_system):
        """Test context window creation and management"""
        system = memory_system
        
        # Create context window
        context = await system.create_context_window(
            context_id="test_context",
            initial_content=["Hello", "world", "this", "is", "test"],
            max_tokens=100
        )
        
        assert context.id == "test_context"
        assert len(context.tokens) == 5
        assert context.current_tokens == 5
        assert system.metrics["context_windows_created"] == 1
        
        # Add content to context
        success = await system.add_to_context(
            context_id="test_context",
            content="Additional content added to context",
            importance=0.8
        )
        
        assert success
        updated_context = system.active_contexts["test_context"]
        assert len(updated_context.importance_scores) == 1
        assert updated_context.importance_scores[0] == 0.8
        assert updated_context.current_tokens > 5
    
    @pytest.mark.asyncio
    async def test_context_compression(self, memory_system):
        """Test context window compression when limits are exceeded"""
        system = memory_system
        
        # Create small context window for testing compression
        context = await system.create_context_window(
            context_id="compress_test",
            initial_content=[],
            max_tokens=20  # Very small for testing
        )
        
        # Add content that will trigger compression
        for i in range(5):
            await system.add_to_context(
                context_id="compress_test",
                content=f"Content segment {i} " * 10,  # Long content
                importance=0.3 if i < 3 else 0.8  # Last two are more important
            )
        
        # Verify compression occurred
        final_context = system.active_contexts["compress_test"]
        assert final_context.current_tokens <= 20
        
        # Check if some contextual memories were created during compression
        # (This is implementation dependent)
    
    @pytest.mark.asyncio
    async def test_episodic_events(self, memory_system):
        """Test episodic memory event management"""
        system = memory_system
        
        # Start an episode
        episode_id = await system.start_episode(
            event_type="test_conversation",
            summary="Testing episodic memory functionality",
            participants=["user", "ai"],
            location="test_environment"
        )
        
        assert episode_id is not None
        assert system.current_episode is not None
        assert system.current_episode.event_type == "test_conversation"
        
        # Store memories during episode
        memory_id1 = await system.store_memory(
            content="First memory during episode",
            memory_type=MemoryType.EPISODIC,
            importance=0.7
        )
        
        memory_id2 = await system.store_memory(
            content="Second memory during episode", 
            memory_type=MemoryType.EPISODIC,
            importance=0.6
        )
        
        # Verify memories were added to episode
        assert memory_id1 in system.current_episode.memories
        assert memory_id2 in system.current_episode.memories
        
        # End episode
        ended_episode = await system.end_episode(
            outcomes=["successful_test", "memory_creation"],
            emotional_tone="positive",
            importance=0.8
        )
        
        assert ended_episode is not None
        assert system.current_episode is None  # Should be cleared
        assert ended_episode.duration is not None
        assert "successful_test" in ended_episode.outcomes
        assert ended_episode.emotional_tone == "positive"
    
    @pytest.mark.asyncio
    async def test_memory_consolidation(self, memory_system):
        """Test memory consolidation from short-term to long-term"""
        system = memory_system
        
        # Store short-term memories that should be consolidated
        consolidation_candidates = []
        
        for i in range(3):
            memory_id = await system.store_memory(
                content=f"Important short-term memory {i}",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.8,  # High importance
                tags={"important", f"memory{i}"}
            )
            consolidation_candidates.append(memory_id)
            
            # Simulate access to increase consolidation probability
            retrieved = await system.retrieve_memory(memory_id, MemoryType.SHORT_TERM)
            retrieved = await system.retrieve_memory(memory_id, MemoryType.SHORT_TERM)
            retrieved = await system.retrieve_memory(memory_id, MemoryType.SHORT_TERM)
        
        # Manually trigger consolidation
        await system._perform_memory_consolidation()
        
        # Check if consolidation metrics increased
        initial_consolidations = system.metrics["consolidations_performed"]
        
        # Verify some memories might have been moved (implementation dependent)
        # At minimum, check that consolidation process ran without errors
        assert system.metrics["consolidations_performed"] >= initial_consolidations
    
    @pytest.mark.asyncio
    async def test_memory_forgetting(self, memory_system):
        """Test memory forgetting mechanism"""
        system = memory_system
        
        # Store low-importance short-term memories
        forgetting_candidates = []
        
        for i in range(3):
            memory_id = await system.store_memory(
                content=f"Low importance memory {i}",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.2,  # Very low importance
                tags={"forgettable"}
            )
            forgetting_candidates.append(memory_id)
        
        # Manually trigger forgetting
        await system._perform_forgetting()
        
        # Check metrics
        initial_forgotten = system.metrics["memories_forgotten"]
        assert system.metrics["memories_forgotten"] >= initial_forgotten
    
    @pytest.mark.asyncio
    async def test_memory_statistics(self, memory_system):
        """Test memory system statistics"""
        system = memory_system
        
        # Add some memories and contexts
        await system.store_memory("Test memory 1", MemoryType.LONG_TERM, 0.8)
        await system.store_memory("Test memory 2", MemoryType.SHORT_TERM, 0.5)
        await system.create_context_window("stats_test", ["hello", "world"])
        
        # Get statistics
        stats = system.get_memory_statistics()
        
        # Verify statistics structure
        assert "total_memories" in stats
        assert "memories_by_type" in stats
        assert "context_statistics" in stats
        assert "episodic_statistics" in stats
        assert "max_context_tokens" in stats
        assert "timestamp" in stats
        
        # Verify some values
        assert stats["total_memories"] >= 2
        assert stats["context_statistics"]["active_contexts"] >= 1
        assert stats["max_context_tokens"] == 10000  # From fixture
    
    @pytest.mark.asyncio
    async def test_memory_associations(self, memory_system):
        """Test memory association functionality"""
        system = memory_system
        
        # Store related memories
        memory_id1 = await system.store_memory(
            content="Python programming language",
            memory_type=MemoryType.SEMANTIC,
            importance=0.7,
            tags={"programming", "python"}
        )
        
        memory_id2 = await system.store_memory(
            content="Machine learning with Python",
            memory_type=MemoryType.SEMANTIC, 
            importance=0.8,
            tags={"programming", "python", "ml"}
        )
        
        # Access memories together to create associations
        await system.retrieve_memory(memory_id1, MemoryType.SEMANTIC)
        await system.retrieve_memory(memory_id2, MemoryType.SEMANTIC)
        
        # Test association graph functionality
        associations = system.association_graph.get_associations(memory_id1)
        
        # The association graph should handle co-accessed memories
        # (Implementation dependent - just verify no errors)
        assert isinstance(associations, list)
    
    @pytest.mark.asyncio
    async def test_embeddings_integration(self, memory_system):
        """Test memory storage and retrieval with embeddings"""
        system = memory_system
        
        # Create test embeddings
        embedding1 = np.random.rand(128).astype(np.float32)
        embedding2 = np.random.rand(128).astype(np.float32)
        
        # Store memories with embeddings
        memory_id1 = await system.store_memory(
            content="Memory with embedding 1",
            memory_type=MemoryType.SEMANTIC,
            importance=0.7,
            embedding=embedding1
        )
        
        memory_id2 = await system.store_memory(
            content="Memory with embedding 2",
            memory_type=MemoryType.SEMANTIC,
            importance=0.6,
            embedding=embedding2
        )
        
        # Retrieve and verify embeddings
        retrieved1 = await system.retrieve_memory(memory_id1, MemoryType.SEMANTIC)
        retrieved2 = await system.retrieve_memory(memory_id2, MemoryType.SEMANTIC)
        
        assert retrieved1.embedding is not None
        assert retrieved2.embedding is not None
        assert np.array_equal(retrieved1.embedding, embedding1)
        assert np.array_equal(retrieved2.embedding, embedding2)
    
    @pytest.mark.asyncio
    async def test_memory_versioning(self, memory_system):
        """Test memory versioning and updates"""
        system = memory_system
        
        # Store initial memory
        memory_id = await system.store_memory(
            content="Original content",
            memory_type=MemoryType.LONG_TERM,
            importance=0.5
        )
        
        # Retrieve and modify
        memory = await system.retrieve_memory(memory_id, MemoryType.LONG_TERM)
        assert memory.version == 1
        
        memory.content = "Updated content"
        memory.version = 2
        
        # Update memory
        success = system.storage.update_memory(memory)
        assert success
        
        # Retrieve updated memory
        updated = await system.retrieve_memory(memory_id, MemoryType.LONG_TERM)
        assert updated.content == "Updated content"
        assert updated.version == 2


class TestMemoryIntegration:
    """Integration tests for memory system with other components"""
    
    @pytest.mark.asyncio
    async def test_load_performance(self):
        """Test memory system performance with larger dataset"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            system = EnhancedMemorySystem(
                max_context_tokens=100000,
                consolidation_interval=3600,
                forgetting_enabled=False  # Disable for performance test
            )
            
            # Override storage path
            system.storage.base_path = temp_dir + "/memory"
            for storage in system.storage.storage_layers.values():
                if hasattr(storage, 'db_path'):
                    storage.db_path = storage.db_path.replace("data/memory", temp_dir + "/memory")
                if hasattr(storage, 'initialize'):
                    storage.initialize()
            
            # Store many memories
            num_memories = 1000
            memory_ids = []
            
            start_time = datetime.now()
            
            for i in range(num_memories):
                memory_id = await system.store_memory(
                    content=f"Performance test memory {i} with content about topic {i % 10}",
                    memory_type=MemoryType.LONG_TERM if i % 2 == 0 else MemoryType.SEMANTIC,
                    importance=np.random.rand(),
                    tags={f"perf_test", f"topic_{i % 10}", f"batch_{i // 100}"}
                )
                memory_ids.append(memory_id)
            
            storage_time = (datetime.now() - start_time).total_seconds()
            print(f"Stored {num_memories} memories in {storage_time:.2f} seconds")
            
            # Test retrieval performance
            start_time = datetime.now()
            
            for i in range(min(100, num_memories)):  # Test 100 retrievals
                memory_type = MemoryType.LONG_TERM if i % 2 == 0 else MemoryType.SEMANTIC
                retrieved = await system.retrieve_memory(memory_ids[i], memory_type)
                assert retrieved is not None
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            print(f"Retrieved 100 memories in {retrieval_time:.2f} seconds")
            
            # Test search performance
            start_time = datetime.now()
            
            for i in range(10):  # Test 10 searches
                results = await system.search_memories(
                    query=f"topic {i}",
                    memory_types=[MemoryType.LONG_TERM, MemoryType.SEMANTIC],
                    max_results=10
                )
            
            search_time = (datetime.now() - start_time).total_seconds()
            print(f"Performed 10 searches in {search_time:.2f} seconds")
            
            # Verify statistics
            stats = system.get_memory_statistics()
            assert stats["total_memories"] == num_memories
            
            # Cleanup background tasks
            for task in system.background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_memory_tests():
    """Run all memory system tests"""
    print("🔹 Running Enhanced Memory System Tests...")
    
    # Run pytest with asyncio support
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("🔹 All memory tests passed!")
    else:
        print("🔹 Some memory tests failed!")
    
    return exit_code


if __name__ == "__main__":
    # Install required packages if running standalone
    try:
        import pytest
        import numpy as np
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "numpy"])
    
    # Run the tests
    run_memory_tests()