"""
🔹 ENHANCED MEMORY & CONTEXT SYSTEM
Extended long-term memory with hierarchical storage, 
context windows up to millions of tokens, and episodic memory formation
"""

import asyncio
import sqlite3
import json
import uuid
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import threading
import heapq

from structlog import get_logger

logger = get_logger()

class MemoryType(Enum):
    SENSORY = "sensory"           # Ultra-short term (milliseconds to seconds)
    SHORT_TERM = "short_term"     # Working memory (seconds to minutes)
    LONG_TERM = "long_term"       # Persistent memory (hours to years)
    EPISODIC = "episodic"         # Event-based memories with context
    SEMANTIC = "semantic"         # Factual knowledge
    PROCEDURAL = "procedural"     # Skill and process knowledge
    AUTOBIOGRAPHICAL = "autobiographical"  # Personal/system history
    CONTEXTUAL = "contextual"     # Context-dependent information

class MemoryImportance(Enum):
    CRITICAL = 1.0      # Never forget
    HIGH = 0.8          # Important information
    MEDIUM = 0.6        # Standard information
    LOW = 0.4           # Background information
    MINIMAL = 0.2       # Easily forgettable

class MemoryAccess(Enum):
    RECENT = "recent"           # Recently accessed
    FREQUENT = "frequent"       # Frequently accessed
    IMPORTANT = "important"     # High importance memories
    CONTEXTUAL = "contextual"   # Context-dependent access
    ASSOCIATIVE = "associative" # Associated memories

@dataclass
class EnhancedMemory:
    """Enhanced memory structure with rich metadata"""
    id: str
    content: str
    memory_type: MemoryType
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    embedding: Optional[np.ndarray] = None
    tags: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    associations: Dict[str, float] = field(default_factory=dict)  # memory_id -> strength
    decay_rate: float = 0.01
    consolidation_level: float = 0.0  # How well consolidated the memory is
    emotional_weight: float = 0.0     # Emotional significance
    source: str = "system"            # Source of the memory
    confidence: float = 1.0           # Confidence in the memory
    version: int = 1                  # Version for memory updates

@dataclass
class ContextWindow:
    """Context window for maintaining conversation/task context"""
    id: str
    tokens: List[str]
    embeddings: List[np.ndarray]
    memory_ids: List[str]
    max_tokens: int
    current_tokens: int
    created_at: datetime
    last_updated: datetime
    importance_scores: List[float]
    attention_weights: List[float] = field(default_factory=list)
    summary: Optional[str] = None

@dataclass
class EpisodicEvent:
    """Represents an episodic memory event"""
    id: str
    event_type: str
    timestamp: datetime
    duration: Optional[timedelta]
    participants: List[str]
    location: str
    memories: List[str]  # Associated memory IDs
    summary: str
    importance: float
    emotional_tone: str
    outcomes: List[str]

class HierarchicalMemoryStorage:
    """Hierarchical storage system for different memory types"""
    
    def __init__(self, base_path: str = "data/memory"):
        self.base_path = base_path
        self.storage_layers = {
            MemoryType.SENSORY: InMemoryStorage(),
            MemoryType.SHORT_TERM: InMemoryStorage(),
            MemoryType.LONG_TERM: SQLiteStorage(f"{base_path}/long_term.db"),
            MemoryType.EPISODIC: SQLiteStorage(f"{base_path}/episodic.db"),
            MemoryType.SEMANTIC: SQLiteStorage(f"{base_path}/semantic.db"),
            MemoryType.PROCEDURAL: SQLiteStorage(f"{base_path}/procedural.db"),
            MemoryType.AUTOBIOGRAPHICAL: SQLiteStorage(f"{base_path}/autobiographical.db"),
            MemoryType.CONTEXTUAL: InMemoryStorage()  # Fast access for context
        }
        
        # Initialize storage layers
        for storage in self.storage_layers.values():
            if hasattr(storage, 'initialize'):
                storage.initialize()
    
    def store_memory(self, memory: EnhancedMemory) -> str:
        """Store memory in appropriate layer"""
        storage = self.storage_layers[memory.memory_type]
        return storage.store(memory)
    
    def retrieve_memory(self, memory_id: str, memory_type: MemoryType) -> Optional[EnhancedMemory]:
        """Retrieve memory from appropriate layer"""
        storage = self.storage_layers[memory_type]
        return storage.retrieve(memory_id)
    
    def search_memories(self, memory_type: MemoryType, **criteria) -> List[EnhancedMemory]:
        """Search memories in specific layer"""
        storage = self.storage_layers[memory_type]
        return storage.search(**criteria)
    
    def update_memory(self, memory: EnhancedMemory) -> bool:
        """Update memory in appropriate layer"""
        storage = self.storage_layers[memory.memory_type]
        return storage.update(memory)
    
    def delete_memory(self, memory_id: str, memory_type: MemoryType) -> bool:
        """Delete memory from appropriate layer"""
        storage = self.storage_layers[memory_type]
        return storage.delete(memory_id)


class EnhancedMemorySystem:
    """🔹 Enhanced memory system with hierarchical storage and context management"""
    
    def __init__(self, 
                 max_context_tokens: int = 1_000_000,  # 1M token context window
                 consolidation_interval: int = 3600,    # Memory consolidation every hour
                 forgetting_enabled: bool = True):
        
        self.max_context_tokens = max_context_tokens
        self.consolidation_interval = consolidation_interval
        self.forgetting_enabled = forgetting_enabled
        
        # Hierarchical storage system
        self.storage = HierarchicalMemoryStorage()
        
        # Context management
        self.active_contexts: Dict[str, ContextWindow] = {}
        self.context_history: List[str] = []  # History of context window IDs
        
        # Memory indexing and search
        self.memory_index = MemoryIndex()
        self.association_graph = AssociationGraph()
        
        # Episodic memory system
        self.episodic_events: Dict[str, EpisodicEvent] = {}
        self.current_episode: Optional[EpisodicEvent] = None
        
        # Memory consolidation and forgetting
        self.consolidation_queue = []
        self.forgetting_candidates = []
        
        # Performance metrics
        self.metrics = {
            "total_memories": 0,
            "memories_by_type": {mt.value: 0 for mt in MemoryType},
            "context_windows_created": 0,
            "memory_retrievals": 0,
            "consolidations_performed": 0,
            "memories_forgotten": 0,
            "average_retrieval_time": 0.0
        }
        
        # Background processes
        self.background_tasks = []
        self._start_background_processes()
        
        logger.info(f"🔹 Enhanced Memory System initialized")
        logger.info(f"🔹 Max context tokens: {max_context_tokens:,}")
    
    def _start_background_processes(self):
        """Start background memory management processes"""
        # Memory consolidation
        consolidation_task = asyncio.create_task(self._consolidation_loop())
        self.background_tasks.append(consolidation_task)
        
        # Memory decay and forgetting
        if self.forgetting_enabled:
            forgetting_task = asyncio.create_task(self._forgetting_loop())
            self.background_tasks.append(forgetting_task)
        
        # Association strengthening
        association_task = asyncio.create_task(self._association_update_loop())
        self.background_tasks.append(association_task)
        
        # Context window management
        context_task = asyncio.create_task(self._context_management_loop())
        self.background_tasks.append(context_task)
    
    async def store_memory(self, 
                          content: str,
                          memory_type: MemoryType = MemoryType.LONG_TERM,
                          importance: float = 0.6,
                          tags: Optional[Set[str]] = None,
                          context: Optional[Dict[str, Any]] = None,
                          embedding: Optional[np.ndarray] = None,
                          emotional_weight: float = 0.0) -> str:
        """Store a new memory with enhanced metadata"""
        
        memory_id = str(uuid.uuid4())
        now = datetime.now()
        
        memory = EnhancedMemory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=now,
            last_accessed=now,
            access_count=1,
            embedding=embedding,
            tags=tags or set(),
            context=context or {},
            decay_rate=self._calculate_decay_rate(importance),
            emotional_weight=emotional_weight
        )
        
        # Store in hierarchical storage
        stored_id = self.storage.store_memory(memory)
        
        # Add to memory index
        await self.memory_index.add_memory(memory)
        
        # Update metrics
        self.metrics["total_memories"] += 1
        self.metrics["memories_by_type"][memory_type.value] += 1
        
        # Add to current episode if applicable
        if self.current_episode and memory_type in [MemoryType.EPISODIC, MemoryType.AUTOBIOGRAPHICAL]:
            self.current_episode.memories.append(memory_id)
        
        logger.debug(f"🔹 Stored {memory_type.value} memory: {memory_id}")
        return stored_id
    
    async def retrieve_memory(self, memory_id: str, memory_type: MemoryType) -> Optional[EnhancedMemory]:
        """Retrieve memory and update access patterns"""
        start_time = time.time()
        
        memory = self.storage.retrieve_memory(memory_id, memory_type)
        
        if memory:
            # Update access patterns
            memory.last_accessed = datetime.now()
            memory.access_count += 1
            
            # Update memory in storage
            self.storage.update_memory(memory)
            
            # Update associations (memories accessed together)
            await self._update_access_associations(memory_id)
            
            self.metrics["memory_retrievals"] += 1
        
        # Update performance metrics
        retrieval_time = time.time() - start_time
        self._update_average_retrieval_time(retrieval_time)
        
        return memory
    
    async def search_memories(self,
                            query: str,
                            memory_types: Optional[List[MemoryType]] = None,
                            max_results: int = 10,
                            importance_threshold: float = 0.1,
                            time_range: Optional[Tuple[datetime, datetime]] = None) -> List[EnhancedMemory]:
        """Advanced memory search with multiple criteria"""
        
        # Use memory index for efficient search
        candidate_ids = await self.memory_index.search(
            query=query,
            memory_types=memory_types,
            max_results=max_results * 2,  # Get more candidates
            importance_threshold=importance_threshold,
            time_range=time_range
        )
        
        # Convert IDs to memory objects
        results = []
        for memory_id in candidate_ids:
            # Try to find the memory in different storage layers
            for memory_type in (memory_types or list(MemoryType)):
                memory = self.storage.retrieve_memory(memory_id, memory_type)
                if memory:
                    results.append(memory)
                    break
        
        # Re-rank results based on current context and associations
        ranked_results = await self._rerank_search_results(results, query)
        
        return ranked_results[:max_results]
    
    async def create_context_window(self, 
                                  context_id: str,
                                  initial_content: List[str],
                                  max_tokens: Optional[int] = None) -> ContextWindow:
        """Create new context window for conversation/task"""
        
        max_tokens = max_tokens or self.max_context_tokens
        
        context_window = ContextWindow(
            id=context_id,
            tokens=initial_content,
            embeddings=[],
            memory_ids=[],
            max_tokens=max_tokens,
            current_tokens=len(' '.join(initial_content).split()),
            created_at=datetime.now(),
            last_updated=datetime.now(),
            importance_scores=[]
        )
        
        self.active_contexts[context_id] = context_window
        self.context_history.append(context_id)
        self.metrics["context_windows_created"] += 1
        
        logger.info(f"🔹 Created context window: {context_id}")
        return context_window
    
    async def add_to_context(self,
                           context_id: str,
                           content: str,
                           importance: float = 0.5,
                           embedding: Optional[np.ndarray] = None) -> bool:
        """Add content to active context window"""
        
        if context_id not in self.active_contexts:
            return False
        
        context = self.active_contexts[context_id]
        tokens = content.split()
        
        # Check if adding content would exceed token limit
        new_token_count = len(tokens)
        
        if context.current_tokens + new_token_count > context.max_tokens:
            # Compress context to make room
            await self._compress_context(context, new_token_count)
        
        # Add new content
        context.tokens.extend(tokens)
        context.current_tokens += new_token_count
        context.last_updated = datetime.now()
        context.importance_scores.append(importance)
        
        if embedding is not None:
            context.embeddings.append(embedding)
        
        return True
    
    async def _compress_context(self, context: ContextWindow, required_space: int):
        """Compress context window to make room for new content"""
        
        # Calculate how many tokens to remove
        tokens_to_remove = required_space + (context.max_tokens // 10)  # Extra 10% buffer
        
        # Use importance scores to determine what to remove
        if context.importance_scores:
            # Create list of (index, importance) pairs
            indexed_importance = list(enumerate(context.importance_scores))
            
            # Sort by importance (lowest first)
            indexed_importance.sort(key=lambda x: x[1])
            
            # Remove lowest importance content
            removed_tokens = 0
            indices_to_remove = []
            
            for idx, importance in indexed_importance:
                if removed_tokens >= tokens_to_remove:
                    break
                
                # Estimate tokens for this segment
                segment_tokens = len(context.tokens) // len(context.importance_scores)
                indices_to_remove.append(idx)
                removed_tokens += segment_tokens
            
            # Remove content (in reverse order to maintain indices)
            for idx in sorted(indices_to_remove, reverse=True):
                segment_start = idx * (len(context.tokens) // len(context.importance_scores))
                segment_end = (idx + 1) * (len(context.tokens) // len(context.importance_scores))
                
                # Store removed content as compressed memory if important enough
                if context.importance_scores[idx] > 0.6:
                    removed_content = ' '.join(context.tokens[segment_start:segment_end])
                    await self.store_memory(
                        content=f"Context compression: {removed_content}",
                        memory_type=MemoryType.CONTEXTUAL,
                        importance=context.importance_scores[idx] * 0.5
                    )
                
                # Remove from context
                del context.tokens[segment_start:segment_end]
                del context.importance_scores[idx]
                
                if idx < len(context.embeddings):
                    del context.embeddings[idx]
                if idx < len(context.memory_ids):
                    del context.memory_ids[idx]
            
            context.current_tokens = len(context.tokens)
        
        else:
            # Simple FIFO removal if no importance scores
            tokens_removed = 0
            while tokens_removed < tokens_to_remove and context.tokens:
                context.tokens.pop(0)
                tokens_removed += 1
            
            context.current_tokens = len(context.tokens)
    
    async def start_episode(self, 
                          event_type: str,
                          summary: str,
                          participants: Optional[List[str]] = None,
                          location: str = "system") -> str:
        """Start a new episodic event"""
        
        episode_id = str(uuid.uuid4())
        
        self.current_episode = EpisodicEvent(
            id=episode_id,
            event_type=event_type,
            timestamp=datetime.now(),
            duration=None,
            participants=participants or [],
            location=location,
            memories=[],
            summary=summary,
            importance=0.5,
            emotional_tone="neutral",
            outcomes=[]
        )
        
        self.episodic_events[episode_id] = self.current_episode
        
        logger.info(f"🔹 Started episode: {event_type} ({episode_id})")
        return episode_id
    
    async def end_episode(self, outcomes: Optional[List[str]] = None, 
                        emotional_tone: str = "neutral",
                        importance: float = 0.5) -> Optional[EpisodicEvent]:
        """End the current episodic event"""
        
        if not self.current_episode:
            return None
        
        episode = self.current_episode
        episode.duration = datetime.now() - episode.timestamp
        episode.outcomes = outcomes or []
        episode.emotional_tone = emotional_tone
        episode.importance = importance
        
        # Store episode as memory
        episode_content = f"Episode: {episode.event_type} - {episode.summary}"
        await self.store_memory(
            content=episode_content,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            tags={"episode", episode.event_type},
            context={
                "episode_id": episode.id,
                "participants": episode.participants,
                "location": episode.location,
                "duration": episode.duration.total_seconds() if episode.duration else 0,
                "emotional_tone": emotional_tone
            },
            emotional_weight=self._emotional_tone_to_weight(emotional_tone)
        )
        
        logger.info(f"🔹 Ended episode: {episode.event_type} ({episode.id})")
        
        self.current_episode = None
        return episode
    
    def _emotional_tone_to_weight(self, emotional_tone: str) -> float:
        """Convert emotional tone to numerical weight"""
        weights = {
            "very_positive": 0.9,
            "positive": 0.7,
            "neutral": 0.0,
            "negative": -0.7,
            "very_negative": -0.9
        }
        return weights.get(emotional_tone, 0.0)
    
    def _calculate_decay_rate(self, importance: float) -> float:
        """Calculate memory decay rate based on importance"""
        # Important memories decay slower
        base_decay = 0.01
        return base_decay * (1.1 - importance)
    
    async def _consolidation_loop(self):
        """Background process for memory consolidation"""
        while True:
            try:
                await self._perform_memory_consolidation()
                await asyncio.sleep(self.consolidation_interval)
            except Exception as e:
                logger.error(f"Error in memory consolidation: {e}")
                await asyncio.sleep(60)
    
    async def _perform_memory_consolidation(self):
        """Perform memory consolidation process"""
        
        # Find memories that need consolidation
        current_time = datetime.now()
        consolidation_threshold = current_time - timedelta(hours=1)
        
        # Move short-term memories to long-term if they meet criteria
        short_term_memories = self.storage.search_memories(
            MemoryType.SHORT_TERM,
            created_before=consolidation_threshold,
            importance_above=0.5
        )
        
        for memory in short_term_memories:
            if self._should_consolidate(memory):
                # Move to long-term storage
                memory.memory_type = MemoryType.LONG_TERM
                memory.consolidation_level += 0.1
                
                # Store in long-term storage
                self.storage.store_memory(memory)
                
                # Remove from short-term
                self.storage.delete_memory(memory.id, MemoryType.SHORT_TERM)
                
                self.metrics["consolidations_performed"] += 1
        
        logger.debug(f"🔹 Consolidated {len(short_term_memories)} memories")
    
    def _should_consolidate(self, memory: EnhancedMemory) -> bool:
        """Determine if memory should be consolidated to long-term"""
        
        # Consolidation criteria
        criteria_met = 0
        
        # High importance
        if memory.importance > 0.7:
            criteria_met += 1
        
        # High access count
        if memory.access_count > 3:
            criteria_met += 1
        
        # Strong emotional weight
        if abs(memory.emotional_weight) > 0.5:
            criteria_met += 1
        
        # Has associations
        if len(memory.associations) > 2:
            criteria_met += 1
        
        # Recent access
        if (datetime.now() - memory.last_accessed).total_seconds() < 3600:  # Last hour
            criteria_met += 1
        
        return criteria_met >= 2
    
    async def _forgetting_loop(self):
        """Background process for memory forgetting/decay"""
        while True:
            try:
                await self._perform_forgetting()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Error in forgetting process: {e}")
                await asyncio.sleep(3600)
    
    async def _perform_forgetting(self):
        """Perform memory forgetting based on decay"""
        
        current_time = datetime.now()
        memories_to_forget = []
        
        # Check all memory types for forgetting candidates
        for memory_type in [MemoryType.SHORT_TERM, MemoryType.SENSORY]:
            memories = self.storage.search_memories(memory_type)
            
            for memory in memories:
                # Calculate time since last access
                time_since_access = (current_time - memory.last_accessed).total_seconds()
                
                # Apply decay formula
                decay_amount = memory.decay_rate * time_since_access / 3600  # Per hour
                current_strength = memory.importance - decay_amount
                
                # Mark for forgetting if strength drops too low
                if current_strength < 0.1 and memory.importance < 0.8:  # Don't forget important memories
                    memories_to_forget.append((memory.id, memory_type))
        
        # Remove forgotten memories
        for memory_id, memory_type in memories_to_forget:
            self.storage.delete_memory(memory_id, memory_type)
            self.metrics["memories_forgotten"] += 1
        
        if memories_to_forget:
            logger.debug(f"🔹 Forgot {len(memories_to_forget)} low-importance memories")
    
    async def _association_update_loop(self):
        """Background process for updating memory associations"""
        while True:
            try:
                await self._update_memory_associations()
                await asyncio.sleep(1800)  # Every 30 minutes
            except Exception as e:
                logger.error(f"Error updating associations: {e}")
                await asyncio.sleep(1800)
    
    async def _update_memory_associations(self):
        """Update memory associations based on co-access patterns"""
        # This would analyze which memories are accessed together
        # and strengthen their associations
        pass
    
    async def _context_management_loop(self):
        """Background process for context window management"""
        while True:
            try:
                await self._manage_context_windows()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Error managing contexts: {e}")
                await asyncio.sleep(300)
    
    async def _manage_context_windows(self):
        """Manage active context windows"""
        current_time = datetime.now()
        
        # Clean up old inactive contexts
        inactive_contexts = []
        for context_id, context in self.active_contexts.items():
            if (current_time - context.last_updated).total_seconds() > 7200:  # 2 hours
                inactive_contexts.append(context_id)
        
        for context_id in inactive_contexts:
            # Optionally store context summary before removal
            context = self.active_contexts[context_id]
            if len(context.tokens) > 100:  # Only summarize substantial contexts
                summary = await self._summarize_context(context)
                await self.store_memory(
                    content=f"Context summary: {summary}",
                    memory_type=MemoryType.CONTEXTUAL,
                    importance=0.4,
                    tags={"context_summary"},
                    context={"original_context_id": context_id}
                )
            
            del self.active_contexts[context_id]
    
    async def _summarize_context(self, context: ContextWindow) -> str:
        """Generate summary of context window content"""
        # Simple extractive summary - take highest importance segments
        if not context.importance_scores:
            return ' '.join(context.tokens[:100])  # First 100 tokens
        
        # Find segments with highest importance
        high_importance_indices = [
            i for i, score in enumerate(context.importance_scores)
            if score > np.mean(context.importance_scores)
        ]
        
        summary_tokens = []
        segment_size = len(context.tokens) // len(context.importance_scores)
        
        for idx in high_importance_indices[:5]:  # Top 5 important segments
            start = idx * segment_size
            end = (idx + 1) * segment_size
            summary_tokens.extend(context.tokens[start:end])
        
        return ' '.join(summary_tokens[:200])  # Limit summary length
    
    async def _rerank_search_results(self, results: List[EnhancedMemory], query: str) -> List[EnhancedMemory]:
        """Re-rank search results based on context and associations"""
        
        # Simple re-ranking based on recency, importance, and access patterns
        def calculate_score(memory: EnhancedMemory) -> float:
            # Base score from importance
            score = memory.importance
            
            # Boost recent memories
            hours_since_access = (datetime.now() - memory.last_accessed).total_seconds() / 3600
            recency_boost = max(0, 0.3 * (1 - hours_since_access / 24))  # Decay over 24 hours
            
            # Boost frequently accessed memories
            frequency_boost = min(0.2, memory.access_count * 0.02)
            
            # Boost emotionally significant memories
            emotion_boost = abs(memory.emotional_weight) * 0.1
            
            return score + recency_boost + frequency_boost + emotion_boost
        
        # Calculate scores and sort
        scored_results = [(memory, calculate_score(memory)) for memory in results]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, score in scored_results]
    
    async def _update_access_associations(self, accessed_memory_id: str):
        """Update associations based on memory access patterns"""
        # Track which memories are accessed together in temporal proximity
        # This would strengthen associations between co-accessed memories
        pass
    
    def _update_average_retrieval_time(self, retrieval_time: float):
        """Update average retrieval time metric"""
        current_avg = self.metrics["average_retrieval_time"]
        total_retrievals = self.metrics["memory_retrievals"]
        
        if total_retrievals == 1:
            self.metrics["average_retrieval_time"] = retrieval_time
        else:
            # Running average
            self.metrics["average_retrieval_time"] = (
                (current_avg * (total_retrievals - 1) + retrieval_time) / total_retrievals
            )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        
        # Calculate context statistics
        context_stats = {
            "active_contexts": len(self.active_contexts),
            "total_context_tokens": sum(c.current_tokens for c in self.active_contexts.values()),
            "average_context_size": (
                sum(c.current_tokens for c in self.active_contexts.values()) / 
                len(self.active_contexts) if self.active_contexts else 0
            )
        }
        
        # Calculate episodic statistics
        episodic_stats = {
            "total_episodes": len(self.episodic_events),
            "current_episode_active": self.current_episode is not None,
            "average_episode_duration": (
                sum((e.duration.total_seconds() if e.duration else 0) 
                    for e in self.episodic_events.values()) / 
                len(self.episodic_events) if self.episodic_events else 0
            )
        }
        
        return {
            **self.metrics,
            "context_statistics": context_stats,
            "episodic_statistics": episodic_stats,
            "max_context_tokens": self.max_context_tokens,
            "consolidation_interval": self.consolidation_interval,
            "forgetting_enabled": self.forgetting_enabled,
            "timestamp": datetime.now().isoformat()
        }


class InMemoryStorage:
    """In-memory storage for fast access"""
    
    def __init__(self):
        self.memories: Dict[str, EnhancedMemory] = {}
    
    def store(self, memory: EnhancedMemory) -> str:
        self.memories[memory.id] = memory
        return memory.id
    
    def retrieve(self, memory_id: str) -> Optional[EnhancedMemory]:
        return self.memories.get(memory_id)
    
    def search(self, **criteria) -> List[EnhancedMemory]:
        results = []
        for memory in self.memories.values():
            if self._matches_criteria(memory, criteria):
                results.append(memory)
        return results
    
    def update(self, memory: EnhancedMemory) -> bool:
        if memory.id in self.memories:
            self.memories[memory.id] = memory
            return True
        return False
    
    def delete(self, memory_id: str) -> bool:
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False
    
    def _matches_criteria(self, memory: EnhancedMemory, criteria: Dict[str, Any]) -> bool:
        # Simple criteria matching
        for key, value in criteria.items():
            if key == "importance_above" and memory.importance <= value:
                return False
            elif key == "created_before" and memory.created_at >= value:
                return False
            elif key == "tags" and not any(tag in memory.tags for tag in value):
                return False
        return True


class SQLiteStorage:
    """SQLite-based persistent storage"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
    
    def initialize(self):
        """Initialize SQLite database"""
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT,
                memory_type TEXT,
                importance REAL,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER,
                embedding BLOB,
                tags TEXT,
                context TEXT,
                associations TEXT,
                decay_rate REAL,
                consolidation_level REAL,
                emotional_weight REAL,
                source TEXT,
                confidence REAL,
                version INTEGER
            )
        """)
        self.connection.commit()
    
    def store(self, memory: EnhancedMemory) -> str:
        if not self.connection:
            self.initialize()
        
        # Serialize complex fields
        embedding_blob = pickle.dumps(memory.embedding) if memory.embedding is not None else None
        tags_json = json.dumps(list(memory.tags))
        context_json = json.dumps(memory.context)
        associations_json = json.dumps(memory.associations)
        
        self.connection.execute("""
            INSERT OR REPLACE INTO memories VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            memory.id, memory.content, memory.memory_type.value,
            memory.importance, memory.created_at.isoformat(),
            memory.last_accessed.isoformat(), memory.access_count,
            embedding_blob, tags_json, context_json, associations_json,
            memory.decay_rate, memory.consolidation_level, memory.emotional_weight,
            memory.source, memory.confidence, memory.version
        ))
        self.connection.commit()
        return memory.id
    
    def retrieve(self, memory_id: str) -> Optional[EnhancedMemory]:
        if not self.connection:
            self.initialize()
        
        cursor = self.connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_memory(row)
        return None
    
    def search(self, **criteria) -> List[EnhancedMemory]:
        if not self.connection:
            self.initialize()
        
        # Build query based on criteria
        where_clauses = []
        params = []
        
        if "importance_above" in criteria:
            where_clauses.append("importance > ?")
            params.append(criteria["importance_above"])
        
        if "created_before" in criteria:
            where_clauses.append("created_at < ?")
            params.append(criteria["created_before"].isoformat())
        
        query = "SELECT * FROM memories"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        cursor = self.connection.execute(query, params)
        rows = cursor.fetchall()
        
        return [self._row_to_memory(row) for row in rows]
    
    def update(self, memory: EnhancedMemory) -> bool:
        return self.store(memory) == memory.id
    
    def delete(self, memory_id: str) -> bool:
        if not self.connection:
            self.initialize()
        
        cursor = self.connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.connection.commit()
        return cursor.rowcount > 0
    
    def _row_to_memory(self, row) -> EnhancedMemory:
        """Convert database row to EnhancedMemory object"""
        (id, content, memory_type, importance, created_at, last_accessed,
         access_count, embedding_blob, tags_json, context_json, associations_json,
         decay_rate, consolidation_level, emotional_weight, source, confidence, version) = row
        
        # Deserialize complex fields
        embedding = pickle.loads(embedding_blob) if embedding_blob else None
        tags = set(json.loads(tags_json))
        context = json.loads(context_json)
        associations = json.loads(associations_json)
        
        return EnhancedMemory(
            id=id,
            content=content,
            memory_type=MemoryType(memory_type),
            importance=importance,
            created_at=datetime.fromisoformat(created_at),
            last_accessed=datetime.fromisoformat(last_accessed),
            access_count=access_count,
            embedding=embedding,
            tags=tags,
            context=context,
            associations=associations,
            decay_rate=decay_rate,
            consolidation_level=consolidation_level,
            emotional_weight=emotional_weight,
            source=source,
            confidence=confidence,
            version=version
        )


class MemoryIndex:
    """Efficient indexing and search for memories"""
    
    def __init__(self):
        self.text_index: Dict[str, Set[str]] = defaultdict(set)  # word -> memory_ids
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)   # tag -> memory_ids
        self.time_index: List[Tuple[datetime, str]] = []         # (timestamp, memory_id)
        self.importance_index: List[Tuple[float, str]] = []      # (importance, memory_id)
    
    async def add_memory(self, memory: EnhancedMemory):
        """Add memory to search indices"""
        # Text index
        words = memory.content.lower().split()
        for word in words:
            self.text_index[word].add(memory.id)
        
        # Tag index
        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)
        
        # Time index
        heapq.heappush(self.time_index, (memory.created_at, memory.id))
        
        # Importance index
        heapq.heappush(self.importance_index, (-memory.importance, memory.id))  # Negative for max heap
    
    async def search(self,
                   query: str,
                   memory_types: Optional[List[MemoryType]] = None,
                   max_results: int = 10,
                   importance_threshold: float = 0.1,
                   time_range: Optional[Tuple[datetime, datetime]] = None) -> List[str]:
        """Search memories and return memory IDs"""
        
        # Text search
        query_words = query.lower().split()
        candidate_ids = set()
        
        for word in query_words:
            if word in self.text_index:
                if not candidate_ids:
                    candidate_ids = self.text_index[word].copy()
                else:
                    candidate_ids &= self.text_index[word]  # Intersection for AND search
        
        # Convert to list for further filtering
        results = list(candidate_ids)
        
        # Additional filtering would be applied here based on other criteria
        # For now, return top results
        return results[:max_results]


class AssociationGraph:
    """Graph structure for memory associations"""
    
    def __init__(self):
        self.associations: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def add_association(self, memory_id1: str, memory_id2: str, strength: float):
        """Add bidirectional association between memories"""
        self.associations[memory_id1][memory_id2] = strength
        self.associations[memory_id2][memory_id1] = strength
    
    def get_associations(self, memory_id: str, min_strength: float = 0.1) -> List[Tuple[str, float]]:
        """Get associated memories above minimum strength"""
        if memory_id not in self.associations:
            return []
        
        return [
            (assoc_id, strength) 
            for assoc_id, strength in self.associations[memory_id].items()
            if strength >= min_strength
        ]
    
    def strengthen_association(self, memory_id1: str, memory_id2: str, increment: float = 0.1):
        """Strengthen association between memories"""
        current_strength = self.associations[memory_id1].get(memory_id2, 0.0)
        new_strength = min(1.0, current_strength + increment)
        self.add_association(memory_id1, memory_id2, new_strength)