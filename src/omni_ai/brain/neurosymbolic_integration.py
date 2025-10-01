"""
🧠 NEUROSYMBOLIC INTEGRATION MODULE
Integrates the neurosymbolic reasoning framework with the main Omni AI brain
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from structlog import get_logger

# Import existing brain components
from .reasoning import ReasoningEngine, ReasoningType, ReasoningChain, ReasoningStep
from .memory import MemorySystem

# Import neurosymbolic framework
from ..neurosymbolic import (
    SymbolicReasoner, 
    KnowledgeGraph, 
    NeuralSymbolicBridge, 
    NaturalLanguageInterface,
    ReasoningMode,
    create_nl_interface
)

logger = get_logger()

class IntegratedReasoningMode(Enum):
    """Enhanced reasoning modes combining traditional and neurosymbolic approaches"""
    TRADITIONAL = "traditional"           # Original brain reasoning only
    SYMBOLIC = "symbolic"                # Pure symbolic logic
    NEURAL_SYMBOLIC = "neural_symbolic"  # Hybrid neural-symbolic
    COLLABORATIVE = "collaborative"      # Both systems working together
    ADAPTIVE = "adaptive"               # Automatically choose best approach

@dataclass
class EnhancedReasoningResult:
    """Result from enhanced reasoning combining multiple approaches"""
    problem: str
    traditional_reasoning: Optional[ReasoningChain]
    neurosymbolic_reasoning: Optional[Any]
    integrated_conclusion: str
    confidence: float
    reasoning_mode: IntegratedReasoningMode
    processing_time: float
    explanation: List[str]
    metadata: Dict[str, Any]

class NeurosymbolicReasoningEngine:
    """🧠 Enhanced reasoning engine integrating traditional and neurosymbolic approaches"""
    
    def __init__(self, memory_system: Optional[MemorySystem] = None):
        # Initialize traditional reasoning
        self.traditional_reasoner = ReasoningEngine()
        
        # Initialize neurosymbolic components
        self.symbolic_reasoner = SymbolicReasoner()
        self.knowledge_graph = KnowledgeGraph()
        self.neural_symbolic_bridge = NeuralSymbolicBridge(
            self.symbolic_reasoner, 
            self.knowledge_graph
        )
        self.nl_interface = NaturalLanguageInterface(self.neural_symbolic_bridge)
        
        # Memory integration
        self.memory_system = memory_system
        
        # Configuration
        self.default_mode = IntegratedReasoningMode.ADAPTIVE
        self.confidence_threshold = 0.7
        
        # Performance tracking
        self.reasoning_history = []
        self.performance_stats = {
            "total_queries": 0,
            "traditional_only": 0,
            "neurosymbolic_only": 0,
            "collaborative": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0
        }
        
        logger.info("🧠 Neurosymbolic Reasoning Engine initialized")
        self._initialize_knowledge_integration()
    
    def _initialize_knowledge_integration(self):
        """Initialize integration between traditional knowledge and neurosymbolic system"""
        
        # Transfer traditional knowledge base facts to symbolic reasoner
        traditional_kb = self.traditional_reasoner.knowledge_base
        
        for domain, content in traditional_kb.items():
            if "facts" in content:
                for fact in content["facts"]:
                    try:
                        # Convert natural language facts to symbolic format
                        self._add_fact_from_text(fact)
                    except Exception as e:
                        logger.warning(f"Could not convert fact '{fact}': {e}")
            
            if "rules" in content:
                for rule in content["rules"]:
                    try:
                        # Convert natural language rules to symbolic format
                        self._add_rule_from_text(rule)
                    except Exception as e:
                        logger.warning(f"Could not convert rule '{rule}': {e}")
        
        logger.info("🧠 Knowledge integration completed")
    
    def _add_fact_from_text(self, text: str):
        """Convert natural language fact to symbolic format"""
        # Simplified conversion - would benefit from more sophisticated NLP
        text = text.lower().strip()
        
        if " is " in text and " is not " not in text:
            parts = text.split(" is ", 1)
            if len(parts) == 2:
                subject = parts[0].strip()
                predicate = parts[1].strip()
                fact = f"is_a({subject}, {predicate})"
                self.symbolic_reasoner.add_fact(fact)
        elif " are " in text:
            parts = text.split(" are ", 1)
            if len(parts) == 2:
                subject = parts[0].strip()
                predicate = parts[1].strip()
                fact = f"is_a({subject}, {predicate})"
                self.symbolic_reasoner.add_fact(fact)
        elif " freezes at " in text:
            parts = text.split(" freezes at ", 1)
            if len(parts) == 2:
                subject = parts[0].strip()
                temperature = parts[1].strip()
                fact = f"freezes_at({subject}, {temperature})"
                self.symbolic_reasoner.add_fact(fact)
        elif " need " in text:
            if "humans need air to breathe" in text:
                self.symbolic_reasoner.add_fact("needs(humans, air)")
        elif "objects fall due to gravity" in text:
            self.symbolic_reasoner.add_fact("affects(gravity, objects)")
    
    def _add_rule_from_text(self, text: str):
        """Convert natural language rule to symbolic format"""
        text = text.lower().strip()
        
        if "if " in text and " then " in text:
            # Already in if-then format
            self.symbolic_reasoner.add_rule(text)
        elif " implies " in text:
            # Convert implies to if-then
            parts = text.split(" implies ", 1)
            if len(parts) == 2:
                condition = parts[0].strip()
                conclusion = parts[1].strip()
                rule = f"IF {condition} THEN {conclusion}"
                self.symbolic_reasoner.add_rule(rule)
    
    async def reason_about(self, 
                          problem: str, 
                          mode: Optional[IntegratedReasoningMode] = None,
                          context: Optional[Dict[str, Any]] = None) -> EnhancedReasoningResult:
        """Enhanced reasoning that combines traditional and neurosymbolic approaches"""
        
        start_time = datetime.now()
        mode = mode or self.default_mode
        
        logger.info(f"🧠 Enhanced reasoning about: '{problem}' with mode: {mode.value}")
        
        # Adaptive mode selection
        if mode == IntegratedReasoningMode.ADAPTIVE:
            mode = self._select_optimal_mode(problem, context)
        
        traditional_result = None
        neurosymbolic_result = None
        
        try:
            if mode == IntegratedReasoningMode.TRADITIONAL:
                traditional_result = await self._traditional_reasoning(problem, context)
                conclusion = traditional_result.final_conclusion
                confidence = traditional_result.overall_confidence
                explanation = [f"Traditional reasoning: {traditional_result.reasoning_path}"]
                
            elif mode == IntegratedReasoningMode.SYMBOLIC:
                neurosymbolic_result = await self._symbolic_reasoning(problem, context)
                conclusion = str(neurosymbolic_result.final_conclusion) if neurosymbolic_result else "No symbolic conclusion"
                confidence = neurosymbolic_result.confidence if neurosymbolic_result else 0.0
                explanation = neurosymbolic_result.explanation if neurosymbolic_result else ["No symbolic reasoning available"]
                
            elif mode == IntegratedReasoningMode.NEURAL_SYMBOLIC:
                neurosymbolic_result = await self._neural_symbolic_reasoning(problem, context)
                conclusion = str(neurosymbolic_result.final_conclusion) if neurosymbolic_result else "No neural-symbolic conclusion"
                confidence = neurosymbolic_result.confidence if neurosymbolic_result else 0.0
                explanation = neurosymbolic_result.explanation if neurosymbolic_result else ["No neural-symbolic reasoning available"]
                
            elif mode == IntegratedReasoningMode.COLLABORATIVE:
                traditional_result, neurosymbolic_result = await self._collaborative_reasoning(problem, context)
                conclusion, confidence, explanation = self._integrate_results(traditional_result, neurosymbolic_result)
                
            else:
                raise ValueError(f"Unsupported reasoning mode: {mode}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create enhanced result
            result = EnhancedReasoningResult(
                problem=problem,
                traditional_reasoning=traditional_result,
                neurosymbolic_reasoning=neurosymbolic_result,
                integrated_conclusion=conclusion,
                confidence=confidence,
                reasoning_mode=mode,
                processing_time=processing_time,
                explanation=explanation,
                metadata={
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                    "reasoning_id": str(uuid.uuid4())
                }
            )
            
            # Store in memory if available
            if self.memory_system:
                await self._store_reasoning_in_memory(result)
            
            # Update performance statistics
            self._update_performance_stats(result)
            
            # Store in reasoning history
            self.reasoning_history.append(result)
            
            logger.info(f"🧠 Enhanced reasoning completed - Confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced reasoning: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedReasoningResult(
                problem=problem,
                traditional_reasoning=traditional_result,
                neurosymbolic_reasoning=neurosymbolic_result,
                integrated_conclusion=f"Reasoning failed: {str(e)}",
                confidence=0.0,
                reasoning_mode=mode,
                processing_time=processing_time,
                explanation=[f"Error occurred during reasoning: {str(e)}"],
                metadata={"error": True, "error_message": str(e)}
            )
    
    def _select_optimal_mode(self, problem: str, context: Optional[Dict[str, Any]]) -> IntegratedReasoningMode:
        """Select the optimal reasoning mode based on problem characteristics"""
        
        problem_lower = problem.lower()
        
        # Heuristics for mode selection
        if any(word in problem_lower for word in ['is', 'are', 'who', 'what', 'why']):
            # Questions that benefit from symbolic logic
            return IntegratedReasoningMode.NEURAL_SYMBOLIC
        
        elif any(word in problem_lower for word in ['if', 'then', 'implies', 'because']):
            # Logical reasoning problems
            return IntegratedReasoningMode.SYMBOLIC
        
        elif any(word in problem_lower for word in ['creative', 'imagine', 'brainstorm', 'innovative']):
            # Creative problems benefit from traditional reasoning
            return IntegratedReasoningMode.TRADITIONAL
        
        elif len(problem.split()) > 20:
            # Complex problems benefit from collaboration
            return IntegratedReasoningMode.COLLABORATIVE
        
        else:
            # Default to neural-symbolic for general queries
            return IntegratedReasoningMode.NEURAL_SYMBOLIC
    
    async def _traditional_reasoning(self, problem: str, context: Optional[Dict[str, Any]]) -> ReasoningChain:
        """Use traditional brain reasoning"""
        reasoning_types = [ReasoningType.LOGICAL, ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE]
        return self.traditional_reasoner.reason_about(problem, reasoning_types)
    
    async def _symbolic_reasoning(self, problem: str, context: Optional[Dict[str, Any]]):
        """Use pure symbolic reasoning"""
        return await self.neural_symbolic_bridge.reason(problem, reasoning_mode=ReasoningMode.SYMBOLIC_ONLY)
    
    async def _neural_symbolic_reasoning(self, problem: str, context: Optional[Dict[str, Any]]):
        """Use neural-symbolic hybrid reasoning"""
        return await self.neural_symbolic_bridge.reason(problem, reasoning_mode=ReasoningMode.PARALLEL)
    
    async def _collaborative_reasoning(self, problem: str, context: Optional[Dict[str, Any]]):
        """Use both traditional and neurosymbolic reasoning"""
        # Run both approaches in parallel
        traditional_task = asyncio.create_task(self._traditional_reasoning(problem, context))
        neurosymbolic_task = asyncio.create_task(self._neural_symbolic_reasoning(problem, context))
        
        traditional_result = await traditional_task
        neurosymbolic_result = await neurosymbolic_task
        
        return traditional_result, neurosymbolic_result
    
    def _integrate_results(self, traditional_result: Optional[ReasoningChain], neurosymbolic_result: Optional[Any]):
        """Integrate results from both reasoning approaches"""
        
        if not traditional_result and not neurosymbolic_result:
            return "No reasoning results available", 0.0, ["Both reasoning approaches failed"]
        
        if traditional_result and not neurosymbolic_result:
            return traditional_result.final_conclusion, traditional_result.overall_confidence, \
                   [f"Traditional reasoning: {traditional_result.reasoning_path}"]
        
        if neurosymbolic_result and not traditional_result:
            return str(neurosymbolic_result.final_conclusion), neurosymbolic_result.confidence, \
                   neurosymbolic_result.explanation
        
        # Both approaches have results - integrate them
        traditional_confidence = traditional_result.overall_confidence
        neurosymbolic_confidence = neurosymbolic_result.confidence
        
        # Weight results by confidence
        if traditional_confidence > neurosymbolic_confidence:
            primary_conclusion = traditional_result.final_conclusion
            secondary_info = f"Neurosymbolic reasoning also suggests: {neurosymbolic_result.final_conclusion}"
        else:
            primary_conclusion = str(neurosymbolic_result.final_conclusion)
            secondary_info = f"Traditional reasoning also suggests: {traditional_result.final_conclusion}"
        
        # Combined confidence (weighted average)
        combined_confidence = (traditional_confidence + neurosymbolic_confidence) / 2
        
        # Agreement bonus
        if self._results_agree(traditional_result.final_conclusion, str(neurosymbolic_result.final_conclusion)):
            combined_confidence = min(1.0, combined_confidence * 1.2)
        
        explanation = [
            f"Traditional reasoning: {traditional_result.reasoning_path}",
            f"Neurosymbolic reasoning: {', '.join(neurosymbolic_result.explanation)}",
            f"Primary conclusion: {primary_conclusion}",
            secondary_info,
            f"Agreement bonus applied: {self._results_agree(traditional_result.final_conclusion, str(neurosymbolic_result.final_conclusion))}"
        ]
        
        return primary_conclusion, combined_confidence, explanation
    
    def _results_agree(self, traditional_conclusion: str, neurosymbolic_conclusion: str) -> bool:
        """Check if reasoning results agree with each other"""
        # Simple agreement check - could be more sophisticated
        traditional_words = set(traditional_conclusion.lower().split())
        neurosymbolic_words = set(neurosymbolic_conclusion.lower().split())
        
        # Check for significant word overlap
        overlap = len(traditional_words.intersection(neurosymbolic_words))
        total_unique_words = len(traditional_words.union(neurosymbolic_words))
        
        similarity = overlap / total_unique_words if total_unique_words > 0 else 0
        return similarity > 0.3  # 30% word overlap threshold
    
    async def _store_reasoning_in_memory(self, result: EnhancedReasoningResult):
        """Store reasoning result in memory system"""
        if self.memory_system:
            memory_content = f"Reasoned about: {result.problem} | Conclusion: {result.integrated_conclusion}"
            
            self.memory_system.store_memory(
                content=memory_content,
                memory_type="episodic",
                importance=result.confidence,
                tags=["reasoning", "enhanced", result.reasoning_mode.value],
                context={
                    "problem": result.problem,
                    "conclusion": result.integrated_conclusion,
                    "confidence": result.confidence,
                    "mode": result.reasoning_mode.value,
                    "processing_time": result.processing_time
                }
            )
    
    def _update_performance_stats(self, result: EnhancedReasoningResult):
        """Update performance statistics"""
        self.performance_stats["total_queries"] += 1
        
        if result.reasoning_mode == IntegratedReasoningMode.TRADITIONAL:
            self.performance_stats["traditional_only"] += 1
        elif result.reasoning_mode in [IntegratedReasoningMode.SYMBOLIC, IntegratedReasoningMode.NEURAL_SYMBOLIC]:
            self.performance_stats["neurosymbolic_only"] += 1
        elif result.reasoning_mode == IntegratedReasoningMode.COLLABORATIVE:
            self.performance_stats["collaborative"] += 1
        
        # Update averages
        total = self.performance_stats["total_queries"]
        self.performance_stats["average_confidence"] = (
            (self.performance_stats["average_confidence"] * (total - 1) + result.confidence) / total
        )
        self.performance_stats["average_processing_time"] = (
            (self.performance_stats["average_processing_time"] * (total - 1) + result.processing_time) / total
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_stats,
            "knowledge_base_facts": len(self.symbolic_reasoner.knowledge_base.facts),
            "knowledge_base_rules": len(self.symbolic_reasoner.knowledge_base.rules),
            "knowledge_graph_entities": len(self.knowledge_graph.entities),
            "knowledge_graph_relations": len(self.knowledge_graph.relations),
            "reasoning_history_size": len(self.reasoning_history)
        }
    
    async def process_natural_language(self, query: str) -> str:
        """Process natural language query and return human-readable response"""
        nl_response = await self.nl_interface.process_query(query)
        return nl_response.answer
    
    def add_knowledge(self, facts: List[str] = None, rules: List[str] = None):
        """Add knowledge to the system"""
        if facts:
            for fact in facts:
                try:
                    self.symbolic_reasoner.add_fact(fact)
                except Exception as e:
                    logger.warning(f"Could not add fact '{fact}': {e}")
        
        if rules:
            for rule in rules:
                try:
                    self.symbolic_reasoner.add_rule(rule)
                except Exception as e:
                    logger.warning(f"Could not add rule '{rule}': {e}")
    
    def clear_reasoning_history(self):
        """Clear reasoning history to free memory"""
        self.reasoning_history.clear()
        logger.info("🧠 Reasoning history cleared")

# Factory function for easy integration
async def create_enhanced_reasoning_engine(memory_system: Optional[MemorySystem] = None) -> NeurosymbolicReasoningEngine:
    """Create an enhanced reasoning engine with neurosymbolic capabilities"""
    engine = NeurosymbolicReasoningEngine(memory_system)
    return engine