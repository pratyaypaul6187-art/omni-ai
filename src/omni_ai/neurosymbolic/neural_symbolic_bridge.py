"""
🔹 NEURAL-SYMBOLIC BRIDGE
Hybrid AI reasoning system that combines neural networks with symbolic logic,
enabling seamless integration between pattern recognition and logical reasoning
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from structlog import get_logger

from .symbolic_reasoner import SymbolicReasoner, Predicate, Rule, InferenceResult
from .knowledge_graph import KnowledgeGraph, Entity, EntityType, RelationType

logger = get_logger()

class ReasoningMode(Enum):
    SYMBOLIC_ONLY = "symbolic_only"
    NEURAL_ONLY = "neural_only"
    SEQUENTIAL = "sequential"        # Neural then symbolic
    PARALLEL = "parallel"           # Both simultaneously
    CONSENSUS = "consensus"         # Both, require agreement
    ADAPTIVE = "adaptive"           # Choose based on input

class ConfidenceAggregation(Enum):
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    MULTIPLICATION = "multiplication"

@dataclass
class NeuralOutput:
    """Output from neural network processing"""
    predictions: Dict[str, float]
    embeddings: Optional[np.ndarray]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SymbolicOutput:
    """Output from symbolic reasoning"""
    conclusions: List[InferenceResult]
    facts_used: List[Predicate]
    rules_fired: List[Rule]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HybridResult:
    """Result from hybrid neural-symbolic reasoning"""
    final_conclusion: str
    confidence: float
    neural_output: Optional[NeuralOutput]
    symbolic_output: Optional[SymbolicOutput]
    reasoning_mode: ReasoningMode
    agreement_score: float
    processing_time: float
    explanation: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class NeuralSymbolicBridge:
    """🔹 Main hybrid reasoning system combining neural and symbolic approaches"""
    
    def __init__(self,
                 symbolic_reasoner: Optional[SymbolicReasoner] = None,
                 knowledge_graph: Optional[KnowledgeGraph] = None,
                 neural_processor: Optional[Callable] = None,
                 default_reasoning_mode: ReasoningMode = ReasoningMode.ADAPTIVE,
                 confidence_threshold: float = 0.7):
        
        # Initialize components
        self.symbolic_reasoner = symbolic_reasoner or SymbolicReasoner()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.neural_processor = neural_processor or self._fallback_neural_processor
        
        # Configuration
        self.default_reasoning_mode = default_reasoning_mode
        self.confidence_threshold = confidence_threshold
        
        # Performance metrics
        self.metrics = {
            "total_inferences": 0,
            "neural_only_count": 0,
            "symbolic_only_count": 0,
            "hybrid_count": 0,
            "agreement_rate": 0.0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "mode_usage": {mode.value: 0 for mode in ReasoningMode}
        }
        
        # Symbol grounding mappings
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_concepts: Dict[str, str] = {}  # Reverse mapping
        
        logger.info("🔹 Neural-Symbolic Bridge initialized")
        logger.info(f"🔹 Default reasoning mode: {default_reasoning_mode.value}")
    
    async def reason(self,
                    input_data: Any,
                    query: Optional[str] = None,
                    reasoning_mode: Optional[ReasoningMode] = None,
                    context: Optional[Dict[str, Any]] = None) -> HybridResult:
        """Main reasoning method combining neural and symbolic approaches"""
        
        start_time = datetime.now()
        mode = reasoning_mode or self.default_reasoning_mode
        
        # Adaptive mode selection
        if mode == ReasoningMode.ADAPTIVE:
            mode = self._select_reasoning_mode(input_data, query, context)
        
        logger.info(f"🧠 Starting hybrid reasoning with mode: {mode.value}")
        
        neural_output = None
        symbolic_output = None
        
        try:
            if mode == ReasoningMode.NEURAL_ONLY:
                neural_output = await self._process_neural(input_data, query, context)
                final_result = self._create_result_from_neural(neural_output, mode, start_time)
                
            elif mode == ReasoningMode.SYMBOLIC_ONLY:
                symbolic_output = await self._process_symbolic(input_data, query, context)
                final_result = self._create_result_from_symbolic(symbolic_output, mode, start_time)
                
            elif mode == ReasoningMode.SEQUENTIAL:
                # Neural first, then symbolic
                neural_output = await self._process_neural(input_data, query, context)
                
                # Use neural output to inform symbolic reasoning
                enhanced_context = self._enhance_context_with_neural(
                    context or {}, neural_output
                )
                symbolic_output = await self._process_symbolic(
                    input_data, query, enhanced_context
                )
                
                final_result = self._combine_outputs(
                    neural_output, symbolic_output, mode, start_time
                )
                
            elif mode == ReasoningMode.PARALLEL:
                # Run both simultaneously
                neural_task = self._process_neural(input_data, query, context)
                symbolic_task = self._process_symbolic(input_data, query, context)
                
                neural_output, symbolic_output = await asyncio.gather(
                    neural_task, symbolic_task
                )
                
                final_result = self._combine_outputs(
                    neural_output, symbolic_output, mode, start_time
                )
                
            elif mode == ReasoningMode.CONSENSUS:
                # Require agreement between neural and symbolic
                neural_output = await self._process_neural(input_data, query, context)
                symbolic_output = await self._process_symbolic(input_data, query, context)
                
                final_result = self._consensus_reasoning(
                    neural_output, symbolic_output, mode, start_time
                )
                
            else:
                raise ValueError(f"Unsupported reasoning mode: {mode}")
            
            # Update metrics
            self._update_metrics(final_result, mode)
            
            logger.info(f"🧠 Hybrid reasoning completed: {final_result.final_conclusion}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in hybrid reasoning: {e}")
            
            # Return fallback result
            processing_time = (datetime.now() - start_time).total_seconds()
            return HybridResult(
                final_conclusion="Error in reasoning process",
                confidence=0.0,
                neural_output=neural_output,
                symbolic_output=symbolic_output,
                reasoning_mode=mode,
                agreement_score=0.0,
                processing_time=processing_time,
                explanation=[f"Error occurred: {str(e)}"],
                metadata={"error": True, "error_message": str(e)}
            )
    
    async def _process_neural(self,
                             input_data: Any,
                             query: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None) -> NeuralOutput:
        """Process input through neural network"""
        
        start_time = datetime.now()
        
        try:
            # Call neural processor
            neural_result = await self.neural_processor(input_data, query, context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Ensure we have a proper NeuralOutput
            if isinstance(neural_result, NeuralOutput):
                neural_result.processing_time = processing_time
                return neural_result
            else:
                # Convert to NeuralOutput if needed
                return NeuralOutput(
                    predictions=neural_result if isinstance(neural_result, dict) else {"output": str(neural_result)},
                    embeddings=None,
                    confidence=0.8,  # Default confidence
                    processing_time=processing_time,
                    metadata={"fallback_conversion": True}
                )
                
        except Exception as e:
            logger.error(f"Neural processing error: {e}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            return NeuralOutput(
                predictions={"error": f"Neural processing failed: {str(e)}"},
                embeddings=None,
                confidence=0.1,
                processing_time=processing_time,
                metadata={"error": True}
            )
    
    async def _process_symbolic(self,
                               input_data: Any,
                               query: Optional[str] = None,
                               context: Optional[Dict[str, Any]] = None) -> SymbolicOutput:
        """Process input through symbolic reasoning"""
        
        start_time = datetime.now()
        
        try:
            # Extract facts from input data
            facts = self._extract_facts_from_input(input_data, context)
            
            # Add facts to knowledge base
            for fact_str in facts:
                try:
                    self.symbolic_reasoner.add_fact(fact_str)
                except Exception as fact_error:
                    logger.warning(f"Failed to add fact '{fact_str}': {fact_error}")
            
            # Perform reasoning
            if query:
                # Backward chaining for specific query
                conclusions = await self.symbolic_reasoner.backward_chain(query)
            else:
                # Forward chaining for general inference
                conclusions = await self.symbolic_reasoner.forward_chain(max_steps=10)
            
            # Calculate overall confidence
            overall_confidence = (
                sum(result.confidence for result in conclusions) / len(conclusions)
                if conclusions else 0.0
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SymbolicOutput(
                conclusions=conclusions,
                facts_used=list(self.symbolic_reasoner.knowledge_base.facts),
                rules_fired=[result.rule_applied for result in conclusions],
                confidence=overall_confidence,
                processing_time=processing_time,
                metadata={
                    "facts_added": len(facts),
                    "query_provided": query is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Symbolic processing error: {e}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            return SymbolicOutput(
                conclusions=[],
                facts_used=[],
                rules_fired=[],
                confidence=0.1,
                processing_time=processing_time,
                metadata={"error": True, "error_message": str(e)}
            )
    
    def _select_reasoning_mode(self,
                             input_data: Any,
                             query: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None) -> ReasoningMode:
        """Adaptively select reasoning mode based on input characteristics"""
        
        # Analyze input characteristics
        has_logical_structure = self._has_logical_structure(input_data, query)
        has_patterns = self._has_pattern_data(input_data)
        complexity = self._estimate_complexity(input_data, query)
        
        # Decision logic
        if has_logical_structure and not has_patterns:
            return ReasoningMode.SYMBOLIC_ONLY
        elif has_patterns and not has_logical_structure:
            return ReasoningMode.NEURAL_ONLY
        elif complexity < 0.3:
            return ReasoningMode.SEQUENTIAL
        elif complexity > 0.7:
            return ReasoningMode.CONSENSUS
        else:
            return ReasoningMode.PARALLEL
    
    def _has_logical_structure(self, input_data: Any, query: Optional[str] = None) -> bool:
        """Check if input has logical structure suitable for symbolic reasoning"""
        
        # Check for logical keywords
        logical_keywords = ["if", "then", "and", "or", "not", "implies", "because", "therefore"]
        
        text_content = str(input_data) + (str(query) if query else "")
        return any(keyword in text_content.lower() for keyword in logical_keywords)
    
    def _has_pattern_data(self, input_data: Any) -> bool:
        """Check if input has patterns suitable for neural processing"""
        
        # Check for numeric data, embeddings, or large text
        if isinstance(input_data, np.ndarray):
            return True
        elif isinstance(input_data, (list, tuple)) and len(input_data) > 10:
            return True
        elif isinstance(input_data, str) and len(input_data) > 200:
            return True
        elif isinstance(input_data, dict) and "embedding" in input_data:
            return True
        
        return False
    
    def _estimate_complexity(self, input_data: Any, query: Optional[str] = None) -> float:
        """Estimate complexity of the reasoning task (0-1 scale)"""
        
        complexity_score = 0.0
        
        # Input data complexity
        if isinstance(input_data, str):
            complexity_score += min(len(input_data) / 1000, 0.3)
        elif isinstance(input_data, (list, tuple)):
            complexity_score += min(len(input_data) / 100, 0.3)
        elif isinstance(input_data, dict):
            complexity_score += min(len(input_data) / 50, 0.3)
        
        # Query complexity
        if query:
            logical_operators = query.lower().count('and') + query.lower().count('or')
            complexity_score += min(logical_operators / 10, 0.3)
        
        # Knowledge base size complexity
        kb_size = len(self.symbolic_reasoner.knowledge_base.facts)
        complexity_score += min(kb_size / 1000, 0.4)
        
        return min(complexity_score, 1.0)
    
    def _extract_facts_from_input(self,
                                 input_data: Any,
                                 context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Extract facts from input data for symbolic reasoning"""
        
        facts = []
        
        # Extract from text input
        if isinstance(input_data, str):
            facts.extend(self._extract_facts_from_text(input_data))
        
        # Extract from structured input
        elif isinstance(input_data, dict):
            for key, value in input_data.items():
                if isinstance(value, str):
                    fact = f"has_property({key}, {value})"
                    facts.append(fact)
                elif isinstance(value, bool):
                    if value:
                        fact = f"is_true({key})"
                        facts.append(fact)
                elif isinstance(value, (int, float)):
                    fact = f"{key}(value) = {value}"
                    facts.append(fact)
        
        # Extract from context
        if context:
            for key, value in context.items():
                if key == "entities" and isinstance(value, list):
                    for entity in value:
                        fact = f"is_a({entity}, entity)"
                        facts.append(fact)
                elif key == "relations" and isinstance(value, list):
                    for relation in value:
                        if isinstance(relation, dict) and all(k in relation for k in ["source", "target", "type"]):
                            fact = f"{relation['type']}({relation['source']}, {relation['target']})"
                            facts.append(fact)
        
        return facts
    
    def _extract_facts_from_text(self, text: str) -> List[str]:
        """Extract facts from natural language text (simplified NLP)"""
        
        facts = []
        
        # Simple pattern matching for basic facts
        sentences = text.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            
            # Pattern: "X is a Y"
            if " is a " in sentence:
                parts = sentence.split(" is a ")
                if len(parts) == 2:
                    x, y = parts[0].strip(), parts[1].strip()
                    fact = f"is_a({x}, {y})"
                    facts.append(fact)
            
            # Pattern: "X has Y"
            elif " has " in sentence:
                parts = sentence.split(" has ")
                if len(parts) == 2:
                    x, y = parts[0].strip(), parts[1].strip()
                    fact = f"has_property({x}, {y})"
                    facts.append(fact)
            
            # Pattern: "X is Y" (property)
            elif " is " in sentence and " is a " not in sentence:
                parts = sentence.split(" is ")
                if len(parts) == 2:
                    x, y = parts[0].strip(), parts[1].strip()
                    fact = f"property({x}, {y})"
                    facts.append(fact)
        
        return facts
    
    def _enhance_context_with_neural(self,
                                   context: Dict[str, Any],
                                   neural_output: NeuralOutput) -> Dict[str, Any]:
        """Enhance context with neural network insights"""
        
        enhanced_context = context.copy()
        
        # Add neural predictions as context
        enhanced_context["neural_predictions"] = neural_output.predictions
        enhanced_context["neural_confidence"] = neural_output.confidence
        
        # Convert embeddings to conceptual hints
        if neural_output.embeddings is not None:
            similar_concepts = self._find_similar_concepts(neural_output.embeddings)
            enhanced_context["similar_concepts"] = similar_concepts
        
        return enhanced_context
    
    def _find_similar_concepts(self, embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """Find similar concepts based on embedding similarity"""
        
        similarities = []
        
        for concept, concept_embedding in self.concept_embeddings.items():
            similarity = self._cosine_similarity(embedding, concept_embedding)
            similarities.append((concept, similarity))
        
        # Sort by similarity and return top concepts
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in similarities[:top_k]]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if vec1.shape != vec2.shape:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _create_result_from_neural(self,
                                  neural_output: NeuralOutput,
                                  mode: ReasoningMode,
                                  start_time: datetime) -> HybridResult:
        """Create hybrid result from neural output only"""
        
        # Extract main conclusion from predictions
        if neural_output.predictions:
            main_prediction = max(neural_output.predictions.items(), 
                                key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
            conclusion = f"{main_prediction[0]}: {main_prediction[1]}"
        else:
            conclusion = "No neural predictions available"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return HybridResult(
            final_conclusion=conclusion,
            confidence=neural_output.confidence,
            neural_output=neural_output,
            symbolic_output=None,
            reasoning_mode=mode,
            agreement_score=1.0,  # Only neural, so perfect agreement
            processing_time=processing_time,
            explanation=[
                "Used neural network processing only",
                f"Neural confidence: {neural_output.confidence:.3f}",
                f"Processing time: {neural_output.processing_time:.3f}s"
            ]
        )
    
    def _create_result_from_symbolic(self,
                                   symbolic_output: SymbolicOutput,
                                   mode: ReasoningMode,
                                   start_time: datetime) -> HybridResult:
        """Create hybrid result from symbolic output only"""
        
        # Extract main conclusion from symbolic reasoning
        if symbolic_output.conclusions:
            main_conclusion = symbolic_output.conclusions[0]
            conclusion = str(main_conclusion.conclusion)
        else:
            conclusion = "No symbolic conclusions derived"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        explanation = [
            "Used symbolic reasoning only",
            f"Symbolic confidence: {symbolic_output.confidence:.3f}",
            f"Facts used: {len(symbolic_output.facts_used)}",
            f"Rules fired: {len(symbolic_output.rules_fired)}"
        ]
        
        return HybridResult(
            final_conclusion=conclusion,
            confidence=symbolic_output.confidence,
            neural_output=None,
            symbolic_output=symbolic_output,
            reasoning_mode=mode,
            agreement_score=1.0,  # Only symbolic, so perfect agreement
            processing_time=processing_time,
            explanation=explanation
        )
    
    def _combine_outputs(self,
                        neural_output: NeuralOutput,
                        symbolic_output: SymbolicOutput,
                        mode: ReasoningMode,
                        start_time: datetime) -> HybridResult:
        """Combine neural and symbolic outputs"""
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement(neural_output, symbolic_output)
        
        # Determine final conclusion based on confidences and agreement
        if neural_output.confidence > symbolic_output.confidence:
            primary_output = "neural"
            if neural_output.predictions:
                main_prediction = max(neural_output.predictions.items(), 
                                    key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                conclusion = f"{main_prediction[0]}: {main_prediction[1]}"
            else:
                conclusion = "Neural processing result"
        else:
            primary_output = "symbolic"
            if symbolic_output.conclusions:
                conclusion = str(symbolic_output.conclusions[0].conclusion)
            else:
                conclusion = "Symbolic reasoning result"
        
        # Combined confidence using weighted average
        combined_confidence = self._aggregate_confidence(
            [neural_output.confidence, symbolic_output.confidence],
            [0.6 if primary_output == "neural" else 0.4, 
             0.4 if primary_output == "neural" else 0.6],
            ConfidenceAggregation.WEIGHTED_AVERAGE
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        explanation = [
            f"Combined neural and symbolic reasoning ({mode.value})",
            f"Primary output: {primary_output}",
            f"Neural confidence: {neural_output.confidence:.3f}",
            f"Symbolic confidence: {symbolic_output.confidence:.3f}",
            f"Agreement score: {agreement_score:.3f}",
            f"Combined confidence: {combined_confidence:.3f}"
        ]
        
        return HybridResult(
            final_conclusion=conclusion,
            confidence=combined_confidence,
            neural_output=neural_output,
            symbolic_output=symbolic_output,
            reasoning_mode=mode,
            agreement_score=agreement_score,
            processing_time=processing_time,
            explanation=explanation
        )
    
    def _consensus_reasoning(self,
                           neural_output: NeuralOutput,
                           symbolic_output: SymbolicOutput,
                           mode: ReasoningMode,
                           start_time: datetime) -> HybridResult:
        """Perform consensus reasoning requiring agreement between neural and symbolic"""
        
        agreement_score = self._calculate_agreement(neural_output, symbolic_output)
        
        # Require high agreement for consensus
        if agreement_score < self.confidence_threshold:
            conclusion = "No consensus reached between neural and symbolic reasoning"
            confidence = min(neural_output.confidence, symbolic_output.confidence) * agreement_score
            explanation = [
                "Consensus reasoning failed - insufficient agreement",
                f"Agreement score: {agreement_score:.3f} < threshold: {self.confidence_threshold}",
                "Neural and symbolic outputs disagree significantly"
            ]
        else:
            # High agreement - combine outputs
            if neural_output.predictions and symbolic_output.conclusions:
                conclusion = "Consensus: Both neural and symbolic reasoning agree"
                confidence = (neural_output.confidence + symbolic_output.confidence) / 2 * agreement_score
                explanation = [
                    "Consensus achieved between neural and symbolic reasoning",
                    f"Agreement score: {agreement_score:.3f}",
                    f"Combined confidence: {confidence:.3f}"
                ]
            else:
                conclusion = "Partial consensus - limited outputs available"
                confidence = (neural_output.confidence + symbolic_output.confidence) / 2
                explanation = [
                    "Partial consensus due to limited outputs",
                    f"Agreement score: {agreement_score:.3f}"
                ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return HybridResult(
            final_conclusion=conclusion,
            confidence=confidence,
            neural_output=neural_output,
            symbolic_output=symbolic_output,
            reasoning_mode=mode,
            agreement_score=agreement_score,
            processing_time=processing_time,
            explanation=explanation
        )
    
    def _calculate_agreement(self,
                           neural_output: NeuralOutput,
                           symbolic_output: SymbolicOutput) -> float:
        """Calculate agreement score between neural and symbolic outputs"""
        
        # Simplified agreement calculation
        # In a full implementation, this would involve more sophisticated comparison
        
        agreement_factors = []
        
        # Confidence agreement
        confidence_diff = abs(neural_output.confidence - symbolic_output.confidence)
        confidence_agreement = 1.0 - confidence_diff
        agreement_factors.append(confidence_agreement)
        
        # Output consistency (simplified)
        if neural_output.predictions and symbolic_output.conclusions:
            # Check if both have high confidence outputs
            neural_high_conf = neural_output.confidence > 0.7
            symbolic_high_conf = symbolic_output.confidence > 0.7
            
            if neural_high_conf == symbolic_high_conf:
                agreement_factors.append(0.8)
            else:
                agreement_factors.append(0.3)
        else:
            # One or both have no outputs
            agreement_factors.append(0.5)
        
        # Return average agreement
        return sum(agreement_factors) / len(agreement_factors)
    
    def _aggregate_confidence(self,
                            confidences: List[float],
                            weights: Optional[List[float]] = None,
                            method: ConfidenceAggregation = ConfidenceAggregation.AVERAGE) -> float:
        """Aggregate multiple confidence scores"""
        
        if not confidences:
            return 0.0
        
        if method == ConfidenceAggregation.AVERAGE:
            return sum(confidences) / len(confidences)
        
        elif method == ConfidenceAggregation.WEIGHTED_AVERAGE:
            if weights and len(weights) == len(confidences):
                weighted_sum = sum(c * w for c, w in zip(confidences, weights))
                weight_sum = sum(weights)
                return weighted_sum / weight_sum if weight_sum > 0 else 0.0
            else:
                return sum(confidences) / len(confidences)
        
        elif method == ConfidenceAggregation.MAXIMUM:
            return max(confidences)
        
        elif method == ConfidenceAggregation.MINIMUM:
            return min(confidences)
        
        elif method == ConfidenceAggregation.MULTIPLICATION:
            result = 1.0
            for conf in confidences:
                result *= conf
            return result
        
        else:
            return sum(confidences) / len(confidences)
    
    async def _fallback_neural_processor(self,
                                       input_data: Any,
                                       query: Optional[str] = None,
                                       context: Optional[Dict[str, Any]] = None) -> NeuralOutput:
        """Fallback neural processor when no real neural network is available"""
        
        # Simple simulation of neural processing
        predictions = {}
        
        # Generate some mock predictions based on input
        if isinstance(input_data, str):
            predictions["text_classification"] = len(input_data) / 1000
            predictions["sentiment"] = 0.6  # Neutral sentiment
        elif isinstance(input_data, dict):
            predictions["structured_analysis"] = 0.7
            predictions["complexity"] = min(len(input_data) / 10, 1.0)
        else:
            predictions["general_analysis"] = 0.5
        
        # Generate random embedding
        embedding = np.random.rand(768).astype(np.float32)
        
        # Calculate confidence based on input characteristics
        confidence = 0.6 + np.random.rand() * 0.3  # 0.6-0.9 range
        
        return NeuralOutput(
            predictions=predictions,
            embeddings=embedding,
            confidence=confidence,
            processing_time=0.001,  # Very fast fallback
            metadata={"fallback": True, "simulated": True}
        )
    
    def _update_metrics(self, result: HybridResult, mode: ReasoningMode):
        """Update performance metrics"""
        
        self.metrics["total_inferences"] += 1
        self.metrics["mode_usage"][mode.value] += 1
        
        if result.neural_output and not result.symbolic_output:
            self.metrics["neural_only_count"] += 1
        elif result.symbolic_output and not result.neural_output:
            self.metrics["symbolic_only_count"] += 1
        else:
            self.metrics["hybrid_count"] += 1
        
        # Update running averages
        total = self.metrics["total_inferences"]
        
        current_avg_agreement = self.metrics["agreement_rate"]
        self.metrics["agreement_rate"] = (
            (current_avg_agreement * (total - 1) + result.agreement_score) / total
        )
        
        current_avg_confidence = self.metrics["average_confidence"]
        self.metrics["average_confidence"] = (
            (current_avg_confidence * (total - 1) + result.confidence) / total
        )
        
        current_avg_time = self.metrics["average_processing_time"]
        self.metrics["average_processing_time"] = (
            (current_avg_time * (total - 1) + result.processing_time) / total
        )
    
    def add_concept_grounding(self, concept: str, embedding: np.ndarray):
        """Add concept-to-embedding grounding for symbol grounding"""
        
        self.concept_embeddings[concept] = embedding
        
        # Create reverse mapping for nearest concepts
        concept_key = f"concept_{len(self.concept_embeddings)}"
        self.embedding_concepts[concept_key] = concept
        
        logger.debug(f"Added concept grounding: {concept}")
    
    def ground_symbols_to_embeddings(self, symbols: List[str]) -> Dict[str, np.ndarray]:
        """Ground symbolic concepts to neural embeddings"""
        
        groundings = {}
        
        for symbol in symbols:
            if symbol in self.concept_embeddings:
                groundings[symbol] = self.concept_embeddings[symbol]
            else:
                # Generate or infer embedding for unknown concept
                # This could involve more sophisticated methods in practice
                inferred_embedding = self._infer_embedding_for_concept(symbol)
                if inferred_embedding is not None:
                    groundings[symbol] = inferred_embedding
        
        return groundings
    
    def _infer_embedding_for_concept(self, concept: str) -> Optional[np.ndarray]:
        """Infer embedding for unknown concept using similarity to known concepts"""
        
        # Simple approach: find similar concept names
        best_match = None
        best_similarity = 0.0
        
        for known_concept in self.concept_embeddings.keys():
            # String similarity (simplified)
            similarity = self._string_similarity(concept, known_concept)
            
            if similarity > best_similarity and similarity > 0.3:
                best_similarity = similarity
                best_match = known_concept
        
        if best_match:
            # Return similar embedding with some noise
            base_embedding = self.concept_embeddings[best_match]
            noise = np.random.normal(0, 0.1, base_embedding.shape)
            return base_embedding + noise
        
        return None
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Simple string similarity measure"""
        
        # Jaccard similarity of character sets
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bridge statistics"""
        
        return {
            **self.metrics,
            "symbolic_reasoner_stats": self.symbolic_reasoner.get_statistics(),
            "knowledge_graph_stats": self.knowledge_graph.get_statistics(),
            "concept_groundings": len(self.concept_embeddings),
            "default_reasoning_mode": self.default_reasoning_mode.value,
            "confidence_threshold": self.confidence_threshold,
            "timestamp": datetime.now().isoformat()
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export bridge configuration"""
        
        return {
            "default_reasoning_mode": self.default_reasoning_mode.value,
            "confidence_threshold": self.confidence_threshold,
            "concept_groundings": list(self.concept_embeddings.keys()),
            "symbolic_knowledge_base": self.symbolic_reasoner.export_knowledge_base(),
            "knowledge_graph": self.knowledge_graph.export_graph(),
            "statistics": self.get_statistics(),
            "exported_at": datetime.now().isoformat()
        }