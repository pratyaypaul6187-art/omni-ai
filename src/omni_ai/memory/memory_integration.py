"""
🔹 MEMORY INTEGRATION BRIDGE
Connects enhanced memory system with AI consciousness and multimodal processing
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from structlog import get_logger

from .enhanced_memory import EnhancedMemorySystem, MemoryType, EnhancedMemory

logger = get_logger()

@dataclass 
class ThoughtMemory:
    """Memory of a thought process with rich context"""
    thought_id: str
    thought_content: str
    modalities: List[str]
    processing_time: float
    neural_activity: Dict[str, float]
    context_tags: set
    importance_score: float
    emotional_resonance: float

@dataclass
class ConversationContext:
    """Context for ongoing conversation/interaction"""
    conversation_id: str
    participants: List[str]
    topic_tags: set
    interaction_type: str
    start_time: datetime
    total_exchanges: int
    emotional_tone: str

class MemoryIntegratedConsciousness:
    """🔹 AI Consciousness with integrated enhanced memory system"""
    
    def __init__(self, 
                 consciousness_core=None,
                 multimodal_processor=None,
                 memory_config: Optional[Dict] = None):
        
        self.consciousness_core = consciousness_core
        self.multimodal_processor = multimodal_processor
        
        # Initialize enhanced memory system
        memory_config = memory_config or {}
        self.memory_system = EnhancedMemorySystem(
            max_context_tokens=memory_config.get('max_context_tokens', 1_000_000),
            consolidation_interval=memory_config.get('consolidation_interval', 3600),
            forgetting_enabled=memory_config.get('forgetting_enabled', True)
        )
        
        # Conversation and context management
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.thought_history: List[ThoughtMemory] = []
        
        # Memory hooks for consciousness
        self.pre_thought_hooks: List = []
        self.post_thought_hooks: List = []
        
        logger.info("🔹 Memory-Integrated Consciousness initialized")
    
    async def think(self, 
                   input_data: Any,
                   conversation_id: Optional[str] = None,
                   context_tags: Optional[set] = None) -> Dict[str, Any]:
        """Enhanced thinking with memory integration"""
        
        # Start thinking episode if conversation context exists
        episode_id = None
        if conversation_id:
            if conversation_id not in self.active_conversations:
                await self._start_conversation_context(conversation_id, input_data)
            
            episode_id = await self.memory_system.start_episode(
                event_type="thinking_process",
                summary=f"Processing input in conversation {conversation_id}",
                participants=self.active_conversations[conversation_id].participants,
                location="ai_consciousness"
            )
        
        try:
            # Retrieve relevant memories for context
            relevant_memories = await self._retrieve_contextual_memories(
                input_data, conversation_id, context_tags
            )
            
            # Create or update context window
            context_window_id = f"think_{conversation_id or 'standalone'}_{datetime.now().timestamp()}"
            context_content = self._prepare_context_content(input_data, relevant_memories)
            
            context = await self.memory_system.create_context_window(
                context_id=context_window_id,
                initial_content=context_content
            )
            
            # Execute pre-thought hooks
            for hook in self.pre_thought_hooks:
                await hook(input_data, relevant_memories, context)
            
            # Core thinking process
            thought_result = await self._execute_thinking_process(
                input_data, relevant_memories, context
            )
            
            # Store thought process as memory
            await self._store_thought_memory(
                thought_result, relevant_memories, context_tags or set()
            )
            
            # Execute post-thought hooks  
            for hook in self.post_thought_hooks:
                await hook(thought_result, relevant_memories, context)
            
            # End thinking episode
            if episode_id:
                await self.memory_system.end_episode(
                    outcomes=[
                        "thought_completed",
                        f"generated_{thought_result.get('output_type', 'response')}"
                    ],
                    emotional_tone=thought_result.get('emotional_tone', 'neutral'),
                    importance=thought_result.get('importance', 0.6)
                )
            
            return thought_result
            
        except Exception as e:
            logger.error(f"Error in memory-integrated thinking: {e}")
            # End episode with error outcome
            if episode_id:
                await self.memory_system.end_episode(
                    outcomes=["error_occurred", str(e)],
                    emotional_tone="negative",
                    importance=0.4
                )
            raise
    
    async def _start_conversation_context(self, conversation_id: str, input_data: Any):
        """Initialize conversation context"""
        
        # Extract context information from input
        participants = self._extract_participants(input_data)
        topic_tags = self._extract_topic_tags(input_data)
        interaction_type = self._determine_interaction_type(input_data)
        
        context = ConversationContext(
            conversation_id=conversation_id,
            participants=participants,
            topic_tags=topic_tags,
            interaction_type=interaction_type,
            start_time=datetime.now(),
            total_exchanges=0,
            emotional_tone="neutral"
        )
        
        self.active_conversations[conversation_id] = context
        
        # Store conversation start as episodic memory
        await self.memory_system.store_memory(
            content=f"Started {interaction_type} conversation {conversation_id}",
            memory_type=MemoryType.EPISODIC,
            importance=0.5,
            tags={"conversation_start", interaction_type} | topic_tags,
            context={
                "conversation_id": conversation_id,
                "participants": participants,
                "interaction_type": interaction_type
            }
        )
    
    async def _retrieve_contextual_memories(self,
                                          input_data: Any,
                                          conversation_id: Optional[str],
                                          context_tags: Optional[set]) -> List[EnhancedMemory]:
        """Retrieve relevant memories for current context"""
        
        relevant_memories = []
        
        # Extract search terms from input
        search_query = self._extract_search_terms(input_data)
        
        if search_query:
            # Search semantic and procedural memories
            semantic_memories = await self.memory_system.search_memories(
                query=search_query,
                memory_types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
                max_results=5,
                importance_threshold=0.3
            )
            relevant_memories.extend(semantic_memories)
            
            # Search conversation history if available
            if conversation_id:
                conversation_memories = await self.memory_system.search_memories(
                    query=f"conversation {conversation_id}",
                    memory_types=[MemoryType.EPISODIC, MemoryType.AUTOBIOGRAPHICAL],
                    max_results=3,
                    importance_threshold=0.4
                )
                relevant_memories.extend(conversation_memories)
        
        # Add recent thoughts from working memory
        recent_thoughts = self.memory_system.storage.search_memories(
            MemoryType.SHORT_TERM,
            importance_above=0.5
        )
        relevant_memories.extend(recent_thoughts[:3])
        
        # Remove duplicates and sort by relevance
        unique_memories = {mem.id: mem for mem in relevant_memories}
        sorted_memories = sorted(unique_memories.values(), 
                               key=lambda m: m.importance, reverse=True)
        
        return sorted_memories[:10]  # Limit to top 10 most relevant
    
    def _prepare_context_content(self, input_data: Any, memories: List[EnhancedMemory]) -> List[str]:
        """Prepare context content from input and memories"""
        
        context_content = []
        
        # Add input representation
        input_repr = self._represent_input_as_text(input_data)
        context_content.extend(input_repr.split())
        
        # Add memory content
        for memory in memories[:5]:  # Limit context from memories
            memory_tokens = f"[MEMORY: {memory.content}]".split()
            context_content.extend(memory_tokens)
        
        return context_content
    
    async def _execute_thinking_process(self,
                                      input_data: Any,
                                      memories: List[EnhancedMemory],
                                      context: Any) -> Dict[str, Any]:
        """Execute the core thinking process with memory context"""
        
        start_time = datetime.now()
        
        # If consciousness core is available, use it
        if self.consciousness_core:
            # Prepare enhanced input with memory context
            enhanced_input = {
                'original_input': input_data,
                'contextual_memories': [
                    {
                        'content': mem.content,
                        'type': mem.memory_type.value,
                        'importance': mem.importance,
                        'tags': list(mem.tags)
                    }
                    for mem in memories
                ],
                'context_window': context.id if context else None
            }
            
            # Use consciousness core's thinking method
            if hasattr(self.consciousness_core, 'process_thought'):
                result = await self.consciousness_core.process_thought(enhanced_input)
            else:
                result = await self._fallback_thinking_process(enhanced_input)
        else:
            # Fallback thinking process
            result = await self._fallback_thinking_process(input_data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Enhance result with processing metadata
        enhanced_result = {
            **result,
            'processing_time': processing_time,
            'memories_used': len(memories),
            'context_window_id': context.id if context else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return enhanced_result
    
    async def _fallback_thinking_process(self, input_data: Any) -> Dict[str, Any]:
        """Fallback thinking process when consciousness core is not available"""
        
        # Simple processing simulation
        processing_result = {
            'output': f"Processed: {str(input_data)[:100]}...",
            'output_type': 'text_response',
            'confidence': 0.7,
            'emotional_tone': 'neutral',
            'importance': 0.6,
            'processing_steps': [
                'input_analysis',
                'memory_integration', 
                'response_generation'
            ]
        }
        
        # Simulate neural activity
        if hasattr(self, 'consciousness_core') and self.consciousness_core:
            try:
                neural_activity = self.consciousness_core.get_neural_activity()
                processing_result['neural_activity'] = neural_activity
            except:
                processing_result['neural_activity'] = {'global_activity': 0.5}
        
        return processing_result
    
    async def _store_thought_memory(self,
                                  thought_result: Dict[str, Any],
                                  relevant_memories: List[EnhancedMemory],
                                  context_tags: set):
        """Store the thought process as memory"""
        
        # Create thought memory content
        thought_content = f"Thought process: {thought_result.get('output', '')[:200]}"
        
        # Determine memory type based on importance and context
        importance = thought_result.get('importance', 0.6)
        memory_type = MemoryType.SHORT_TERM if importance < 0.7 else MemoryType.LONG_TERM
        
        # Extract processing metadata
        processing_time = thought_result.get('processing_time', 0)
        neural_activity = thought_result.get('neural_activity', {})
        
        # Create comprehensive tags
        tags = {
            'thought_process',
            'ai_generated',
            thought_result.get('output_type', 'unknown')
        } | context_tags
        
        # Store as memory
        memory_id = await self.memory_system.store_memory(
            content=thought_content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            context={
                'processing_time': processing_time,
                'neural_activity': neural_activity,
                'memories_used': [mem.id for mem in relevant_memories],
                'output_type': thought_result.get('output_type'),
                'confidence': thought_result.get('confidence', 0.5)
            },
            emotional_weight=self._emotional_tone_to_weight(
                thought_result.get('emotional_tone', 'neutral')
            )
        )
        
        # Create thought memory record
        thought_memory = ThoughtMemory(
            thought_id=memory_id,
            thought_content=thought_content,
            modalities=self._extract_modalities(thought_result),
            processing_time=processing_time,
            neural_activity=neural_activity,
            context_tags=tags,
            importance_score=importance,
            emotional_resonance=self._emotional_tone_to_weight(
                thought_result.get('emotional_tone', 'neutral')
            )
        )
        
        self.thought_history.append(thought_memory)
        
        # Keep thought history limited
        if len(self.thought_history) > 100:
            self.thought_history.pop(0)
    
    async def process_multimodal_input(self,
                                     input_data: Dict[str, Any],
                                     conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process multimodal input with memory integration"""
        
        if not self.multimodal_processor:
            return await self.think(input_data, conversation_id)
        
        # Process through multimodal system
        multimodal_result = await self.multimodal_processor.process_multimodal(input_data)
        
        # Store multimodal processing results
        for modality, result in multimodal_result.items():
            if result.get('embedding') is not None:
                await self.memory_system.store_memory(
                    content=f"Multimodal {modality} processing: {str(result.get('features', ''))[:100]}",
                    memory_type=MemoryType.SENSORY,
                    importance=0.4,
                    tags={modality, 'multimodal', 'sensory'},
                    embedding=result['embedding'],
                    context={
                        'modality': modality,
                        'processing_stats': result.get('stats', {}),
                        'conversation_id': conversation_id
                    }
                )
        
        # Process unified understanding through consciousness
        unified_result = await self.think(
            multimodal_result, 
            conversation_id, 
            context_tags={'multimodal_input'}
        )
        
        return unified_result
    
    def add_memory_hook(self, hook_type: str, hook_function):
        """Add memory processing hooks"""
        if hook_type == 'pre_thought':
            self.pre_thought_hooks.append(hook_function)
        elif hook_type == 'post_thought':
            self.post_thought_hooks.append(hook_function)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
    
    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get comprehensive conversation summary"""
        
        if conversation_id not in self.active_conversations:
            return {"error": "Conversation not found"}
        
        context = self.active_conversations[conversation_id]
        
        # Get episodic memories for this conversation
        conversation_memories = await self.memory_system.search_memories(
            query=f"conversation {conversation_id}",
            memory_types=[MemoryType.EPISODIC, MemoryType.AUTOBIOGRAPHICAL],
            max_results=20
        )
        
        # Get thought history for this conversation
        conversation_thoughts = [
            thought for thought in self.thought_history
            if conversation_id in str(thought.context_tags)
        ]
        
        summary = {
            'conversation_id': conversation_id,
            'context': {
                'participants': context.participants,
                'topic_tags': list(context.topic_tags),
                'interaction_type': context.interaction_type,
                'start_time': context.start_time.isoformat(),
                'duration': str(datetime.now() - context.start_time),
                'total_exchanges': context.total_exchanges,
                'emotional_tone': context.emotional_tone
            },
            'memories': {
                'total_memories': len(conversation_memories),
                'important_memories': [
                    {
                        'content': mem.content,
                        'importance': mem.importance,
                        'tags': list(mem.tags),
                        'created_at': mem.created_at.isoformat()
                    }
                    for mem in conversation_memories[:5]
                ],
            },
            'thought_processes': {
                'total_thoughts': len(conversation_thoughts),
                'avg_processing_time': (
                    sum(t.processing_time for t in conversation_thoughts) / 
                    len(conversation_thoughts) if conversation_thoughts else 0
                ),
                'emotional_trends': self._analyze_emotional_trends(conversation_thoughts)
            },
            'memory_statistics': self.memory_system.get_memory_statistics()
        }
        
        return summary
    
    async def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """End conversation and consolidate memories"""
        
        if conversation_id not in self.active_conversations:
            return {"error": "Conversation not found"}
        
        context = self.active_conversations[conversation_id]
        
        # Create conversation summary memory
        summary = await self.get_conversation_summary(conversation_id)
        
        await self.memory_system.store_memory(
            content=f"Conversation {conversation_id} summary: "
                   f"{context.interaction_type} with {', '.join(context.participants)}",
            memory_type=MemoryType.AUTOBIOGRAPHICAL,
            importance=0.7,
            tags={'conversation_summary', context.interaction_type} | context.topic_tags,
            context={
                'conversation_id': conversation_id,
                'full_summary': summary,
                'duration': str(datetime.now() - context.start_time),
                'total_exchanges': context.total_exchanges
            }
        )
        
        # Remove from active conversations
        del self.active_conversations[conversation_id]
        
        return summary
    
    # Utility methods for context extraction and processing
    
    def _extract_participants(self, input_data: Any) -> List[str]:
        """Extract participants from input data"""
        if isinstance(input_data, dict) and 'participants' in input_data:
            return input_data['participants']
        return ['user', 'ai']
    
    def _extract_topic_tags(self, input_data: Any) -> set:
        """Extract topic tags from input data"""
        tags = set()
        
        if isinstance(input_data, dict):
            if 'tags' in input_data:
                tags.update(input_data['tags'])
            if 'topics' in input_data:
                tags.update(input_data['topics'])
        
        # Extract from text content if available
        text_content = self._extract_text_content(input_data)
        if text_content:
            # Simple keyword extraction (could be enhanced with NLP)
            words = text_content.lower().split()
            important_words = [word for word in words if len(word) > 4]
            tags.update(important_words[:5])  # Limit to 5 keywords
        
        return tags
    
    def _determine_interaction_type(self, input_data: Any) -> str:
        """Determine the type of interaction"""
        if isinstance(input_data, dict):
            if 'interaction_type' in input_data:
                return input_data['interaction_type']
            
            # Infer from modalities present
            modalities = set(input_data.keys())
            if 'audio' in modalities:
                return 'voice_conversation'
            elif 'image' in modalities or 'video' in modalities:
                return 'visual_interaction'
            elif 'text' in modalities:
                return 'text_conversation'
        
        return 'general_interaction'
    
    def _extract_search_terms(self, input_data: Any) -> str:
        """Extract search terms from input data"""
        text_content = self._extract_text_content(input_data)
        
        if text_content and isinstance(text_content, str):
            # Simple search term extraction
            words = text_content.split()
            # Remove common stop words and keep meaningful terms
            meaningful_words = [word for word in words if len(word) > 2]
            return ' '.join(meaningful_words[:10])  # Limit search terms
        
        return ""
    
    def _extract_text_content(self, input_data: Any) -> str:
        """Extract text content from various input formats"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            if 'text' in input_data:
                return str(input_data['text'])
            elif 'content' in input_data:
                return str(input_data['content'])
            elif 'query' in input_data:
                return str(input_data['query'])
            else:
                # For complex objects, try to extract meaningful text
                text_parts = []
                for key, value in input_data.items():
                    if isinstance(value, str) and len(value) > 3:
                        text_parts.append(value)
                    elif isinstance(value, dict) and 'text' in value:
                        text_parts.append(str(value['text']))
                return ' '.join(text_parts[:5])  # Limit to avoid too much text
        
        return str(input_data)
    
    def _represent_input_as_text(self, input_data: Any) -> str:
        """Convert input data to text representation"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return f"Input: {str(input_data)}"
        else:
            return f"Data: {str(input_data)}"
    
    def _extract_modalities(self, thought_result: Dict[str, Any]) -> List[str]:
        """Extract modalities involved in thought process"""
        modalities = ['text']  # Default
        
        if 'modalities' in thought_result:
            return thought_result['modalities']
        
        # Infer from output type
        output_type = thought_result.get('output_type', 'text')
        if 'audio' in output_type:
            modalities.append('audio')
        if 'visual' in output_type or 'image' in output_type:
            modalities.append('visual')
        
        return modalities
    
    def _emotional_tone_to_weight(self, emotional_tone: str) -> float:
        """Convert emotional tone to numerical weight"""
        weights = {
            'very_positive': 0.9,
            'positive': 0.7,
            'slightly_positive': 0.3,
            'neutral': 0.0,
            'slightly_negative': -0.3,
            'negative': -0.7,
            'very_negative': -0.9
        }
        return weights.get(emotional_tone, 0.0)
    
    def _analyze_emotional_trends(self, thoughts: List[ThoughtMemory]) -> Dict[str, float]:
        """Analyze emotional trends in thought history"""
        if not thoughts:
            return {'average_emotional_tone': 0.0, 'emotional_variance': 0.0}
        
        emotional_values = [thought.emotional_resonance for thought in thoughts]
        
        return {
            'average_emotional_tone': sum(emotional_values) / len(emotional_values),
            'emotional_variance': np.var(emotional_values) if len(emotional_values) > 1 else 0.0,
            'emotional_trend': 'stable'  # Could be enhanced with trend analysis
        }