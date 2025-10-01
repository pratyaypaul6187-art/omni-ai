"""
🧠 ENHANCED AI CONSCIOUSNESS WITH NEUROSYMBOLIC REASONING
Advanced AI consciousness that integrates neurosymbolic reasoning capabilities
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

from structlog import get_logger

# Import base consciousness components
from .consciousness import (
    AIConsciousness, 
    ThoughtType, 
    ConsciousnessState, 
    Thought, 
    CognitiveState
)
from .memory import MemorySystem
from .neurosymbolic_integration import (
    NeurosymbolicReasoningEngine,
    IntegratedReasoningMode,
    EnhancedReasoningResult,
    create_enhanced_reasoning_engine
)

logger = get_logger()

class EnhancedThinkingMode(Enum):
    """Enhanced thinking modes with neurosymbolic capabilities"""
    INTUITIVE = "intuitive"               # Fast, pattern-based thinking
    ANALYTICAL = "analytical"             # Deep, step-by-step reasoning  
    CREATIVE = "creative"                # Innovative and imaginative
    LOGICAL = "logical"                  # Formal logic and symbolic reasoning
    COLLABORATIVE = "collaborative"      # Multiple reasoning approaches
    REFLECTIVE = "reflective"           # Self-aware and introspective
    ADAPTIVE = "adaptive"               # Automatically choose best approach

@dataclass
class EnhancedThought:
    """Enhanced thought with neurosymbolic reasoning trace"""
    id: str
    content: str
    thought_type: ThoughtType
    thinking_mode: EnhancedThinkingMode
    confidence: float
    reasoning_result: Optional[EnhancedReasoningResult]
    context: Dict[str, Any]
    triggered_by: str
    associations: List[str]
    timestamp: datetime
    processing_time: float

class EnhancedAIConsciousness(AIConsciousness):
    """🧠 Enhanced AI consciousness with neurosymbolic reasoning capabilities"""
    
    def __init__(self, memory_db_path: str = "data/memory.db", enable_neurosymbolic: bool = True):
        # Initialize base consciousness
        super().__init__(memory_db_path)
        
        # Enhanced capabilities
        self.enable_neurosymbolic = enable_neurosymbolic
        self.enhanced_reasoning_engine = None
        
        # Enhanced thought stream
        self.enhanced_thought_stream = deque(maxlen=1000)
        self.thinking_patterns = defaultdict(int)
        
        # Reasoning preferences
        self.preferred_thinking_mode = EnhancedThinkingMode.ADAPTIVE
        self.reasoning_confidence_threshold = 0.6
        
        # Performance metrics
        self.enhanced_metrics = {
            "neurosymbolic_queries": 0,
            "traditional_queries": 0,
            "collaborative_queries": 0,
            "average_reasoning_confidence": 0.0,
            "average_reasoning_time": 0.0,
            "successful_symbolic_inferences": 0
        }
        
        # Initialize neurosymbolic reasoning if enabled
        if self.enable_neurosymbolic:
            asyncio.create_task(self._initialize_neurosymbolic_reasoning())
        
        logger.info("🧠 Enhanced AI Consciousness initialized with neurosymbolic capabilities")
    
    async def _initialize_neurosymbolic_reasoning(self):
        """Initialize the neurosymbolic reasoning engine"""
        try:
            self.enhanced_reasoning_engine = await create_enhanced_reasoning_engine(self.memory)
            
            # Add some initial knowledge
            await self._bootstrap_knowledge()
            
            logger.info("🧠 Neurosymbolic reasoning engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize neurosymbolic reasoning: {e}")
            self.enable_neurosymbolic = False
    
    async def _bootstrap_knowledge(self):
        """Bootstrap the neurosymbolic system with initial knowledge"""
        if not self.enhanced_reasoning_engine:
            return
        
        # Add basic facts about AI consciousness
        facts = [
            "is_a(omni_ai, artificial_intelligence)",
            "has_property(omni_ai, consciousness)",
            "has_property(omni_ai, memory)",
            "has_property(omni_ai, reasoning)",
            "has_property(artificial_intelligence, learning_capability)"
        ]
        
        # Add basic reasoning rules
        rules = [
            "IF is_a(X, artificial_intelligence) AND has_property(X, consciousness) THEN capable_of(X, thinking)",
            "IF has_property(X, memory) AND has_property(X, reasoning) THEN can_learn(X)",
            "IF capable_of(X, thinking) AND can_learn(X) THEN is_intelligent(X)"
        ]
        
        self.enhanced_reasoning_engine.add_knowledge(facts=facts, rules=rules)
        logger.info("🧠 Bootstrap knowledge added to neurosymbolic system")
    
    async def enhanced_think(self, 
                           stimulus: str, 
                           context: Dict[str, Any] = None,
                           thinking_mode: Optional[EnhancedThinkingMode] = None) -> Dict[str, Any]:
        """🧠 Enhanced thinking function with neurosymbolic reasoning"""
        
        start_time = datetime.now()
        thinking_mode = thinking_mode or self.preferred_thinking_mode
        
        # Update cognitive state
        self._update_cognitive_state("enhanced_thinking", stimulus)
        
        # Store the stimulus in memory
        stimulus_memory_id = self.memory.store_memory(
            f"Enhanced thinking stimulus: {stimulus}",
            memory_type="short_term",
            importance=0.7,
            tags=["stimulus", "enhanced_thinking", thinking_mode.value],
            context=context or {}
        )
        
        logger.info(f"🧠 Enhanced thinking about: '{stimulus}' with mode: {thinking_mode.value}")
        
        # Choose reasoning approach based on thinking mode
        if thinking_mode == EnhancedThinkingMode.ADAPTIVE:
            thinking_mode = self._select_optimal_thinking_mode(stimulus, context)
        
        response = None
        reasoning_result = None
        
        try:
            if thinking_mode == EnhancedThinkingMode.INTUITIVE:
                response = await self._intuitive_thinking(stimulus, context)
                
            elif thinking_mode == EnhancedThinkingMode.ANALYTICAL:
                response = await self._analytical_thinking(stimulus, context)
                
            elif thinking_mode == EnhancedThinkingMode.LOGICAL:
                response = await self._logical_thinking(stimulus, context)
                
            elif thinking_mode == EnhancedThinkingMode.CREATIVE:
                response = await self._creative_thinking(stimulus, context)
                
            elif thinking_mode == EnhancedThinkingMode.COLLABORATIVE:
                response = await self._collaborative_thinking(stimulus, context)
                
            elif thinking_mode == EnhancedThinkingMode.REFLECTIVE:
                response = await self._reflective_thinking(stimulus, context)
                
            else:
                # Fallback to traditional thinking
                response = await self._fallback_to_traditional_thinking(stimulus, context)
        
        except Exception as e:
            logger.error(f"Error in enhanced thinking: {e}")
            response = await self._fallback_to_traditional_thinking(stimulus, context)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create enhanced thought
        thought_id = f"enhanced_thought_{int(time.time())}"
        enhanced_thought = EnhancedThought(
            id=thought_id,
            content=response.get('content', ''),
            thought_type=self._determine_thought_type(stimulus, response),
            thinking_mode=thinking_mode,
            confidence=response.get('confidence', 0.5),
            reasoning_result=reasoning_result,
            context=context or {},
            triggered_by=stimulus,
            associations=response.get('associations', []),
            timestamp=start_time,
            processing_time=processing_time
        )
        
        # Store enhanced thought
        self.enhanced_thought_stream.append(enhanced_thought)
        
        # Update metrics
        self._update_enhanced_metrics(enhanced_thought)
        
        # Store thinking process in memory
        thinking_memory_id = self.memory.store_memory(
            f"Enhanced thought: {stimulus} | Response: {response['content']} | Mode: {thinking_mode.value}",
            memory_type="episodic",
            importance=response.get('confidence', 0.5),
            tags=["enhanced_thinking", thinking_mode.value, "response"],
            context={
                "stimulus": stimulus,
                "response": response,
                "thinking_mode": thinking_mode.value,
                "processing_time": processing_time,
                "confidence": response.get('confidence', 0.5)
            }
        )
        
        # Enhanced response with additional metadata
        return {
            **response,
            "thinking_mode": thinking_mode.value,
            "processing_time": processing_time,
            "thought_id": thought_id,
            "enhanced_reasoning": reasoning_result is not None,
            "confidence": response.get('confidence', 0.5)
        }
    
    def _select_optimal_thinking_mode(self, stimulus: str, context: Optional[Dict[str, Any]]) -> EnhancedThinkingMode:
        """Select the optimal thinking mode based on stimulus characteristics"""
        
        stimulus_lower = stimulus.lower()
        
        # Logical thinking for questions and formal reasoning
        if any(word in stimulus_lower for word in ['is', 'are', 'who', 'what', 'why', 'how', 'if', 'then']):
            return EnhancedThinkingMode.LOGICAL
        
        # Creative thinking for brainstorming and innovation
        if any(word in stimulus_lower for word in ['create', 'invent', 'imagine', 'brainstorm', 'design']):
            return EnhancedThinkingMode.CREATIVE
        
        # Reflective thinking for self-related queries
        if any(word in stimulus_lower for word in ['yourself', 'you are', 'your thoughts', 'reflect']):
            return EnhancedThinkingMode.REFLECTIVE
        
        # Analytical thinking for complex problems
        if len(stimulus.split()) > 15:
            return EnhancedThinkingMode.ANALYTICAL
        
        # Collaborative thinking for ambiguous situations
        if context and context.get('uncertainty_level', 0) > 0.7:
            return EnhancedThinkingMode.COLLABORATIVE
        
        # Default to analytical for most cases
        return EnhancedThinkingMode.ANALYTICAL
    
    async def _intuitive_thinking(self, stimulus: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fast, pattern-based intuitive thinking"""
        # Use traditional reasoning for quick responses
        traditional_response = self.think(stimulus, context)
        return {
            **traditional_response,
            "reasoning_type": "intuitive",
            "confidence": traditional_response.get('confidence', 0.6) * 0.8  # Slightly lower confidence for speed
        }
    
    async def _analytical_thinking(self, stimulus: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Deep, step-by-step analytical thinking"""
        if self.enhanced_reasoning_engine:
            reasoning_result = await self.enhanced_reasoning_engine.reason_about(
                stimulus, 
                mode=IntegratedReasoningMode.COLLABORATIVE,
                context=context
            )
            
            return {
                "content": reasoning_result.integrated_conclusion,
                "confidence": reasoning_result.confidence,
                "reasoning_type": "analytical",
                "processing_steps": reasoning_result.explanation,
                "reasoning_mode": reasoning_result.reasoning_mode.value
            }
        else:
            # Fallback to traditional reasoning
            return await self._fallback_to_traditional_thinking(stimulus, context)
    
    async def _logical_thinking(self, stimulus: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Formal logic and symbolic reasoning"""
        if self.enhanced_reasoning_engine:
            # Try natural language processing first
            try:
                nl_response = await self.enhanced_reasoning_engine.process_natural_language(stimulus)
                
                # If NL processing was successful, return that
                if nl_response and "Error" not in nl_response:
                    return {
                        "content": nl_response,
                        "confidence": 0.8,
                        "reasoning_type": "logical_nl",
                        "approach": "natural_language_interface"
                    }
            except Exception as e:
                logger.warning(f"Natural language processing failed: {e}")
            
            # Fallback to symbolic reasoning
            reasoning_result = await self.enhanced_reasoning_engine.reason_about(
                stimulus,
                mode=IntegratedReasoningMode.SYMBOLIC,
                context=context
            )
            
            return {
                "content": reasoning_result.integrated_conclusion,
                "confidence": reasoning_result.confidence,
                "reasoning_type": "logical_symbolic",
                "reasoning_steps": reasoning_result.explanation
            }
        else:
            return await self._fallback_to_traditional_thinking(stimulus, context)
    
    async def _creative_thinking(self, stimulus: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Creative and imaginative thinking"""
        # Use traditional reasoning for creative tasks (it has better creative reasoning)
        traditional_response = self.think(stimulus, context)
        
        # Enhance with some random associations if neurosymbolic is available
        if self.enhanced_reasoning_engine:
            try:
                # Get some knowledge graph associations for inspiration
                stats = self.enhanced_reasoning_engine.get_performance_stats()
                associations = [f"knowledge_entities: {stats.get('knowledge_graph_entities', 0)}"]
            except:
                associations = []
        else:
            associations = []
        
        return {
            **traditional_response,
            "reasoning_type": "creative",
            "associations": associations,
            "confidence": traditional_response.get('confidence', 0.7)
        }
    
    async def _collaborative_thinking(self, stimulus: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Multiple reasoning approaches working together"""
        if self.enhanced_reasoning_engine:
            reasoning_result = await self.enhanced_reasoning_engine.reason_about(
                stimulus,
                mode=IntegratedReasoningMode.COLLABORATIVE,
                context=context
            )
            
            return {
                "content": reasoning_result.integrated_conclusion,
                "confidence": reasoning_result.confidence,
                "reasoning_type": "collaborative",
                "explanation": reasoning_result.explanation,
                "approaches_used": ["traditional", "neurosymbolic"]
            }
        else:
            return await self._fallback_to_traditional_thinking(stimulus, context)
    
    async def _reflective_thinking(self, stimulus: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Self-aware and introspective thinking"""
        # Use reflection capabilities
        reflection_entry = self.reflection.reflect_on_thinking(
            stimulus, 
            confidence=0.7,
            context={"enhanced_thinking": True}
        )
        
        # Combine with traditional thinking
        traditional_response = self.think(stimulus, context)
        
        return {
            "content": f"Reflecting on this question: {traditional_response['content']}. "
                      f"My reflection: {reflection_entry.content}",
            "confidence": (traditional_response.get('confidence', 0.6) + reflection_entry.confidence) / 2,
            "reasoning_type": "reflective",
            "reflection_id": reflection_entry.id,
            "self_awareness_level": self.cognitive_state.consciousness_level
        }
    
    async def _fallback_to_traditional_thinking(self, stimulus: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback to traditional thinking methods"""
        traditional_response = self.think(stimulus, context)
        return {
            **traditional_response,
            "reasoning_type": "traditional_fallback",
            "note": "Using traditional reasoning due to neurosymbolic unavailability"
        }
    
    def _determine_thought_type(self, stimulus: str, response: Dict[str, Any]) -> ThoughtType:
        """Determine the type of thought based on stimulus and response"""
        reasoning_type = response.get('reasoning_type', '')
        
        if 'logical' in reasoning_type or 'analytical' in reasoning_type:
            return ThoughtType.ANALYTICAL
        elif 'creative' in reasoning_type:
            return ThoughtType.CREATIVE
        elif 'reflective' in reasoning_type:
            return ThoughtType.REFLECTIVE
        elif 'collaborative' in reasoning_type:
            return ThoughtType.EXPLORATORY
        else:
            return ThoughtType.ANALYTICAL
    
    def _update_enhanced_metrics(self, thought: EnhancedThought):
        """Update enhanced performance metrics"""
        self.enhanced_metrics["total_enhanced_thoughts"] = self.enhanced_metrics.get("total_enhanced_thoughts", 0) + 1
        
        if thought.thinking_mode == EnhancedThinkingMode.LOGICAL:
            self.enhanced_metrics["neurosymbolic_queries"] += 1
        elif thought.thinking_mode in [EnhancedThinkingMode.INTUITIVE, EnhancedThinkingMode.CREATIVE]:
            self.enhanced_metrics["traditional_queries"] += 1
        elif thought.thinking_mode == EnhancedThinkingMode.COLLABORATIVE:
            self.enhanced_metrics["collaborative_queries"] += 1
        
        # Update averages
        total = self.enhanced_metrics["total_enhanced_thoughts"]
        self.enhanced_metrics["average_reasoning_confidence"] = (
            (self.enhanced_metrics["average_reasoning_confidence"] * (total - 1) + thought.confidence) / total
        )
        self.enhanced_metrics["average_reasoning_time"] = (
            (self.enhanced_metrics["average_reasoning_time"] * (total - 1) + thought.processing_time) / total
        )
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced consciousness statistics"""
        base_stats = self.get_consciousness_report()
        
        enhanced_stats = {
            "enhanced_metrics": self.enhanced_metrics,
            "neurosymbolic_enabled": self.enable_neurosymbolic,
            "enhanced_thoughts": len(self.enhanced_thought_stream),
            "preferred_thinking_mode": self.preferred_thinking_mode.value,
            "reasoning_confidence_threshold": self.reasoning_confidence_threshold
        }
        
        if self.enhanced_reasoning_engine:
            enhanced_stats["neurosymbolic_performance"] = self.enhanced_reasoning_engine.get_performance_stats()
        
        return {
            **base_stats,
            **enhanced_stats
        }
    
    async def ask_question(self, question: str) -> str:
        """Ask a question and get a natural response"""
        response = await self.enhanced_think(question, context={"query_type": "question"})
        return response['content']
    
    def set_thinking_preference(self, mode: EnhancedThinkingMode):
        """Set preferred thinking mode"""
        self.preferred_thinking_mode = mode
        logger.info(f"🧠 Preferred thinking mode set to: {mode.value}")
    
    async def learn_from_interaction(self, interaction: str, feedback: str = None):
        """Learn from user interaction"""
        if self.enhanced_reasoning_engine and feedback:
            # Add the interaction as knowledge
            try:
                fact = f"learned({interaction.replace(' ', '_')}, {feedback.replace(' ', '_')})"
                self.enhanced_reasoning_engine.add_knowledge(facts=[fact])
                logger.info(f"🧠 Learned from interaction: {interaction}")
            except Exception as e:
                logger.warning(f"Could not learn from interaction: {e}")

# Factory function for easy creation
async def create_enhanced_consciousness(memory_db_path: str = "data/memory.db", 
                                      enable_neurosymbolic: bool = True) -> EnhancedAIConsciousness:
    """Create an enhanced AI consciousness with neurosymbolic capabilities"""
    consciousness = EnhancedAIConsciousness(memory_db_path, enable_neurosymbolic)
    
    # Give it a moment to initialize neurosymbolic components
    await asyncio.sleep(0.1)
    
    return consciousness