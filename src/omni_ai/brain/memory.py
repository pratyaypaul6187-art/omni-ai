"""
🧠 MEMORY SYSTEM
Advanced memory capabilities for AI consciousness
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import hashlib
from collections import defaultdict, deque

from structlog import get_logger

logger = get_logger()

@dataclass
class Memory:
    """A single memory unit"""
    id: str
    content: str
    memory_type: str  # short_term, long_term, episodic, semantic
    importance: float  # 0.0 to 1.0
    created_at: datetime
    last_accessed: datetime
    access_count: int
    emotions: List[str]
    tags: List[str]
    context: Dict[str, Any]
    associations: List[str]  # IDs of related memories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)

class MemorySystem:
    """🧠 Advanced memory system with multiple memory types"""
    
    def __init__(self, memory_db_path: str = "data/memory.db"):
        self.memory_db_path = Path(memory_db_path)
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Memory stores
        self.short_term_memory = deque(maxlen=100)  # Recent memories
        self.working_memory = {}  # Currently active memories
        self.long_term_memory = {}  # Persistent important memories
        self.episodic_memory = {}  # Event-based memories
        self.semantic_memory = {}  # Knowledge and facts
        
        # Memory management
        self.memory_index = {}  # Fast lookup index
        self.association_graph = defaultdict(set)  # Memory associations
        self.access_patterns = defaultdict(int)  # Memory access frequency
        
        # Initialize database
        self._init_database()
        self._load_memories()
        
        # Start memory management thread
        self.memory_thread = threading.Thread(target=self._memory_management_loop, daemon=True)
        self.memory_thread.start()
        
        logger.info("🧠 Memory system initialized")
    
    def _init_database(self):
        """Initialize memory database"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        importance REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER NOT NULL,
                        emotions TEXT NOT NULL,
                        tags TEXT NOT NULL,
                        context TEXT NOT NULL,
                        associations TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed DESC)
                """)
                
        except Exception as e:
            logger.error(f"Failed to initialize memory database: {e}")
    
    def _load_memories(self):
        """Load memories from database"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.execute("SELECT * FROM memories")
                for row in cursor:
                    memory_data = {
                        'id': row[0],
                        'content': row[1],
                        'memory_type': row[2],
                        'importance': row[3],
                        'created_at': row[4],
                        'last_accessed': row[5],
                        'access_count': row[6],
                        'emotions': json.loads(row[7]),
                        'tags': json.loads(row[8]),
                        'context': json.loads(row[9]),
                        'associations': json.loads(row[10])
                    }
                    
                    memory = Memory.from_dict(memory_data)
                    self._store_memory_in_category(memory)
                    
            logger.info(f"🧠 Loaded {len(self.memory_index)} memories from database")
            
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def _store_memory_in_category(self, memory: Memory):
        """Store memory in appropriate category"""
        self.memory_index[memory.id] = memory
        
        if memory.memory_type == "short_term":
            self.short_term_memory.append(memory)
        elif memory.memory_type == "long_term":
            self.long_term_memory[memory.id] = memory
        elif memory.memory_type == "episodic":
            self.episodic_memory[memory.id] = memory
        elif memory.memory_type == "semantic":
            self.semantic_memory[memory.id] = memory
        elif memory.memory_type == "working":
            self.working_memory[memory.id] = memory
        
        # Build association graph
        for assoc_id in memory.associations:
            self.association_graph[memory.id].add(assoc_id)
            self.association_graph[assoc_id].add(memory.id)
    
    def store_memory(self, content: str, memory_type: str = "short_term", 
                    importance: float = 0.5, emotions: List[str] = None,
                    tags: List[str] = None, context: Dict[str, Any] = None) -> str:
        """🧠 Store a new memory"""
        
        # Generate unique ID
        memory_id = hashlib.md5(f"{content}_{time.time()}".encode()).hexdigest()
        
        # Create memory
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            emotions=emotions or [],
            tags=tags or [],
            context=context or {},
            associations=[]
        )
        
        # Find associations
        memory.associations = self._find_associations(memory)
        
        # Store in appropriate category
        self._store_memory_in_category(memory)
        
        # Save to database
        self._save_memory_to_db(memory)
        
        logger.info(f"🧠 Stored {memory_type} memory: {memory_id}")
        return memory_id
    
    def _find_associations(self, new_memory: Memory) -> List[str]:
        """Find related memories for association"""
        associations = []
        
        # Simple keyword-based association (can be enhanced with embeddings)
        new_words = set(new_memory.content.lower().split())
        new_tags = set(new_memory.tags)
        
        for memory_id, existing_memory in self.memory_index.items():
            if memory_id == new_memory.id:
                continue
                
            # Check content similarity
            existing_words = set(existing_memory.content.lower().split())
            word_overlap = len(new_words & existing_words)
            
            # Check tag similarity
            existing_tags = set(existing_memory.tags)
            tag_overlap = len(new_tags & existing_tags)
            
            # Check context similarity
            context_overlap = 0
            for key in new_memory.context:
                if key in existing_memory.context:
                    if new_memory.context[key] == existing_memory.context[key]:
                        context_overlap += 1
            
            # Score association strength
            association_score = (word_overlap * 0.4 + tag_overlap * 0.4 + context_overlap * 0.2)
            
            if association_score > 2:  # Threshold for association
                associations.append(memory_id)
        
        return associations[:10]  # Limit associations
    
    def recall_memory(self, query: str, memory_types: List[str] = None, 
                     max_results: int = 10) -> List[Memory]:
        """🧠 Recall memories based on query"""
        
        if memory_types is None:
            memory_types = ["short_term", "long_term", "episodic", "semantic", "working"]
        
        results = []
        query_words = set(query.lower().split())
        
        for memory_id, memory in self.memory_index.items():
            if memory.memory_type not in memory_types:
                continue
            
            # Calculate relevance score
            memory_words = set(memory.content.lower().split())
            word_overlap = len(query_words & memory_words)
            tag_overlap = len(query_words & set(memory.tags))
            
            relevance_score = (
                word_overlap * 0.5 + 
                tag_overlap * 0.3 + 
                memory.importance * 0.2
            )
            
            if relevance_score > 0:
                # Update access information
                memory.last_accessed = datetime.now()
                memory.access_count += 1
                self.access_patterns[memory_id] += 1
                
                results.append((relevance_score, memory))
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        recalled_memories = [memory for _, memory in results[:max_results]]
        
        logger.info(f"🧠 Recalled {len(recalled_memories)} memories for query: {query}")
        return recalled_memories
    
    def get_associated_memories(self, memory_id: str, max_depth: int = 2) -> List[Memory]:
        """🧠 Get memories associated with a given memory"""
        visited = set()
        associated = []
        
        def _get_associations(mid, depth):
            if depth > max_depth or mid in visited:
                return
            
            visited.add(mid)
            
            for assoc_id in self.association_graph[mid]:
                if assoc_id in self.memory_index:
                    associated.append(self.memory_index[assoc_id])
                    _get_associations(assoc_id, depth + 1)
        
        _get_associations(memory_id, 0)
        return associated
    
    def promote_memory(self, memory_id: str, new_type: str):
        """🧠 Promote memory to different type (e.g., short_term -> long_term)"""
        if memory_id not in self.memory_index:
            return
        
        memory = self.memory_index[memory_id]
        old_type = memory.memory_type
        memory.memory_type = new_type
        memory.importance = min(1.0, memory.importance + 0.1)  # Increase importance
        
        # Move to new category
        self._remove_from_old_category(memory, old_type)
        self._store_memory_in_category(memory)
        self._save_memory_to_db(memory)
        
        logger.info(f"🧠 Promoted memory {memory_id} from {old_type} to {new_type}")
    
    def _remove_from_old_category(self, memory: Memory, old_type: str):
        """Remove memory from old category"""
        if old_type == "short_term":
            if memory in self.short_term_memory:
                self.short_term_memory.remove(memory)
        elif old_type == "long_term":
            self.long_term_memory.pop(memory.id, None)
        elif old_type == "episodic":
            self.episodic_memory.pop(memory.id, None)
        elif old_type == "semantic":
            self.semantic_memory.pop(memory.id, None)
        elif old_type == "working":
            self.working_memory.pop(memory.id, None)
    
    def consolidate_memories(self):
        """🧠 Consolidate short-term memories into long-term"""
        promotion_candidates = []
        
        # Find memories to promote based on importance and access patterns
        for memory in self.short_term_memory:
            score = (
                memory.importance * 0.4 +
                (memory.access_count / max(1, len(self.short_term_memory))) * 0.3 +
                len(memory.associations) * 0.3
            )
            
            if score > 0.6:  # Promotion threshold
                promotion_candidates.append((score, memory))
        
        # Promote top candidates
        promotion_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, memory in promotion_candidates[:5]:  # Promote top 5
            self.promote_memory(memory.id, "long_term")
        
        logger.info(f"🧠 Consolidated {len(promotion_candidates)} memories to long-term")
    
    def _save_memory_to_db(self, memory: Memory):
        """Save memory to database"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, content, memory_type, importance, created_at, last_accessed, 
                     access_count, emotions, tags, context, associations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    memory.content,
                    memory.memory_type,
                    memory.importance,
                    memory.created_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    json.dumps(memory.emotions),
                    json.dumps(memory.tags),
                    json.dumps(memory.context),
                    json.dumps(memory.associations)
                ))
        except Exception as e:
            logger.error(f"Failed to save memory to database: {e}")
    
    def _memory_management_loop(self):
        """Background memory management"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                # Consolidate memories
                self.consolidate_memories()
                
                # Clean up old, unimportant short-term memories
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=24)
                
                memories_to_remove = []
                for memory in self.short_term_memory:
                    if (memory.last_accessed < cutoff_time and 
                        memory.importance < 0.3 and 
                        memory.access_count < 2):
                        memories_to_remove.append(memory)
                
                for memory in memories_to_remove:
                    if memory in self.short_term_memory:
                        self.short_term_memory.remove(memory)
                    self.memory_index.pop(memory.id, None)
                
                if memories_to_remove:
                    logger.info(f"🧠 Cleaned up {len(memories_to_remove)} old memories")
                
            except Exception as e:
                logger.error(f"Error in memory management loop: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """🧠 Get memory system statistics"""
        return {
            'total_memories': len(self.memory_index),
            'short_term': len(self.short_term_memory),
            'long_term': len(self.long_term_memory),
            'episodic': len(self.episodic_memory),
            'semantic': len(self.semantic_memory),
            'working': len(self.working_memory),
            'associations': sum(len(assocs) for assocs in self.association_graph.values()) // 2,
            'most_accessed': sorted(self.access_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def think_about(self, topic: str) -> List[Memory]:
        """🧠 Actively think about a topic by recalling related memories"""
        logger.info(f"🧠 Thinking about: {topic}")
        
        # Recall initial memories
        initial_memories = self.recall_memory(topic, max_results=5)
        
        # Get associated memories
        all_related = []
        for memory in initial_memories:
            associated = self.get_associated_memories(memory.id, max_depth=1)
            all_related.extend(associated)
        
        # Remove duplicates and combine
        all_memories = initial_memories + all_related
        unique_memories = []
        seen_ids = set()
        
        for memory in all_memories:
            if memory.id not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(memory.id)
        
        # Store this thinking session as an episodic memory
        self.store_memory(
            content=f"Thought about '{topic}' and recalled {len(unique_memories)} related memories",
            memory_type="episodic",
            importance=0.6,
            tags=["thinking", "reflection", topic],
            context={"thinking_session": True, "topic": topic}
        )
        
        return unique_memories