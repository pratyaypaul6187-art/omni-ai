"""
🔮 ADVANCED REASONING ENGINE
Sophisticated reasoning capabilities including temporal logic, probabilistic inference,
uncertainty handling, and explanation generation for neurosymbolic AI
"""

import asyncio
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from statistics import mean, stdev

from structlog import get_logger

from .symbolic_reasoner import SymbolicReasoner, Predicate, Rule, LogicalOperator
from .knowledge_graph import KnowledgeGraph, EntityType, RelationType

logger = get_logger()

class TemporalOperator(Enum):
    """Temporal logic operators"""
    ALWAYS = "always"           # □ (Box) - always true
    EVENTUALLY = "eventually"   # ◇ (Diamond) - eventually true
    NEXT = "next"              # X - true in next state
    UNTIL = "until"            # U - true until condition
    SINCE = "since"            # S - true since condition
    BEFORE = "before"          # < - temporal ordering
    AFTER = "after"            # > - temporal ordering
    DURING = "during"          # ⊆ - interval containment

class UncertaintyType(Enum):
    """Types of uncertainty handling"""
    PROBABILITY = "probability"         # Bayesian probability
    FUZZY = "fuzzy"                    # Fuzzy logic values [0,1]
    CONFIDENCE = "confidence"          # Confidence intervals
    POSSIBILITY = "possibility"        # Possibilistic logic
    DEMPSTER_SHAFER = "dempster_shafer" # Evidence theory

class ReasoningMode(Enum):
    """Advanced reasoning modes"""
    DEDUCTIVE = "deductive"           # Classical deductive reasoning
    INDUCTIVE = "inductive"           # Learning from examples
    ABDUCTIVE = "abductive"           # Best explanation inference
    ANALOGICAL = "analogical"         # Reasoning by analogy
    CAUSAL = "causal"                # Causal reasoning
    COUNTERFACTUAL = "counterfactual" # What-if scenarios
    MODAL = "modal"                   # Necessity and possibility

@dataclass
class TemporalFact:
    """A fact with temporal information"""
    predicate: str
    confidence: float
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    temporal_operator: Optional[TemporalOperator] = None
    
    def is_valid_at(self, time: datetime) -> bool:
        """Check if fact is valid at given time"""
        if self.valid_from and time < self.valid_from:
            return False
        if self.valid_until and time > self.valid_until:
            return False
        return True

@dataclass
class ProbabilisticFact:
    """A fact with probabilistic information"""
    predicate: str
    probability: float
    evidence: List[str] = field(default_factory=list)
    uncertainty_type: UncertaintyType = UncertaintyType.PROBABILITY
    
    def __post_init__(self):
        # Ensure probability is in valid range
        self.probability = max(0.0, min(1.0, self.probability))

@dataclass
class CausalRelation:
    """Represents a causal relationship"""
    cause: str
    effect: str
    strength: float  # Causal strength [0,1]
    mechanism: Optional[str] = None
    confounders: List[str] = field(default_factory=list)

@dataclass
class Explanation:
    """An explanation for a reasoning result"""
    conclusion: str
    reasoning_chain: List[str]
    confidence: float
    supporting_facts: List[str]
    assumptions: List[str] = field(default_factory=list)
    alternative_explanations: List[str] = field(default_factory=list)

@dataclass
class ReasoningResult:
    """Result of advanced reasoning operation"""
    query: str
    answer: Any
    confidence: float
    explanation: Explanation
    reasoning_mode: ReasoningMode
    computation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedReasoner:
    """🔮 Advanced Reasoning Engine with sophisticated inference capabilities"""
    
    def __init__(self, base_reasoner: Optional[SymbolicReasoner] = None):
        self.base_reasoner = base_reasoner or SymbolicReasoner()
        
        # Advanced knowledge storage
        self.temporal_facts: List[TemporalFact] = []
        self.probabilistic_facts: List[ProbabilisticFact] = []
        self.causal_relations: List[CausalRelation] = []
        
        # Reasoning state
        self.current_time = datetime.now()
        self.reasoning_history: List[ReasoningResult] = []
        
        # Configuration
        self.config = {
            "temporal_reasoning_enabled": True,
            "probabilistic_reasoning_enabled": True,
            "explanation_depth": 3,
            "confidence_threshold": 0.5,
            "max_inference_steps": 100,
            "uncertainty_propagation_method": "monte_carlo"
        }
        
        # Performance metrics
        self.metrics = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "average_confidence": 0.0,
            "reasoning_modes_used": defaultdict(int),
            "explanation_requests": 0
        }
        
        logger.info("🔮 Advanced Reasoning Engine initialized")
        
    async def add_temporal_fact(self, predicate: str, confidence: float = 1.0,
                              valid_from: Optional[datetime] = None,
                              valid_until: Optional[datetime] = None,
                              temporal_operator: Optional[TemporalOperator] = None):
        """Add a temporal fact to the knowledge base"""
        
        temporal_fact = TemporalFact(
            predicate=predicate,
            confidence=confidence,
            valid_from=valid_from,
            valid_until=valid_until,
            temporal_operator=temporal_operator
        )
        
        self.temporal_facts.append(temporal_fact)
        
        # Also add to base reasoner if currently valid
        if temporal_fact.is_valid_at(self.current_time):
            self.base_reasoner.add_fact(predicate, confidence)
        
        logger.debug(f"Added temporal fact: {predicate}")
    
    async def add_probabilistic_fact(self, predicate: str, probability: float,
                                   evidence: Optional[List[str]] = None,
                                   uncertainty_type: UncertaintyType = UncertaintyType.PROBABILITY):
        """Add a probabilistic fact to the knowledge base"""
        
        prob_fact = ProbabilisticFact(
            predicate=predicate,
            probability=probability,
            evidence=evidence or [],
            uncertainty_type=uncertainty_type
        )
        
        self.probabilistic_facts.append(prob_fact)
        
        # Add to base reasoner with probability as confidence
        self.base_reasoner.add_fact(predicate, probability)
        
        logger.debug(f"Added probabilistic fact: {predicate} (P={probability:.3f})")
    
    async def add_causal_relation(self, cause: str, effect: str, strength: float,
                                mechanism: Optional[str] = None,
                                confounders: Optional[List[str]] = None):
        """Add a causal relationship"""
        
        causal_rel = CausalRelation(
            cause=cause,
            effect=effect,
            strength=max(0.0, min(1.0, strength)),
            mechanism=mechanism,
            confounders=confounders or []
        )
        
        self.causal_relations.append(causal_rel)
        
        # Add corresponding rule to base reasoner
        rule_str = f"IF {cause} THEN {effect}"
        self.base_reasoner.add_rule(rule_str, priority=1, confidence=strength)
        
        logger.debug(f"Added causal relation: {cause} → {effect} (strength={strength:.3f})")
    
    async def temporal_query(self, query: str, at_time: Optional[datetime] = None) -> ReasoningResult:
        """Perform temporal reasoning query"""
        start_time = datetime.now()
        query_time = at_time or self.current_time
        
        # Filter temporal facts valid at query time
        valid_facts = [
            fact for fact in self.temporal_facts 
            if fact.is_valid_at(query_time)
        ]
        
        # Create temporary reasoner with valid facts
        temp_reasoner = SymbolicReasoner()
        for fact in valid_facts:
            temp_reasoner.add_fact(fact.predicate, fact.confidence)
        
        # Copy rules from base reasoner
        for rule in self.base_reasoner.knowledge_base.rules:
            temp_reasoner.knowledge_base.rules.append(rule)
        
        # Perform query
        base_result = await temp_reasoner.query(query)
        
        # Calculate confidence based on temporal validity
        confidence = self._calculate_temporal_confidence(base_result, query_time)
        
        # Generate explanation
        explanation = await self._generate_temporal_explanation(query, base_result, query_time, valid_facts)
        
        result = ReasoningResult(
            query=query,
            answer=base_result,
            confidence=confidence,
            explanation=explanation,
            reasoning_mode=ReasoningMode.DEDUCTIVE,
            computation_time=(datetime.now() - start_time).total_seconds(),
            metadata={"query_time": query_time.isoformat(), "valid_facts_count": len(valid_facts)}
        )
        
        self.reasoning_history.append(result)
        self._update_metrics(result)
        
        logger.info(f"🔮 Temporal query '{query}' at {query_time}: {len(base_result)} results")
        return result
    
    async def probabilistic_query(self, query: str, method: str = "bayesian") -> ReasoningResult:
        """Perform probabilistic reasoning query"""
        start_time = datetime.now()
        
        if method == "bayesian":
            result = await self._bayesian_inference(query)
        elif method == "fuzzy":
            result = await self._fuzzy_inference(query)
        elif method == "monte_carlo":
            result = await self._monte_carlo_inference(query)
        else:
            raise ValueError(f"Unknown probabilistic method: {method}")
        
        reasoning_result = ReasoningResult(
            query=query,
            answer=result["answer"],
            confidence=result["confidence"],
            explanation=result["explanation"],
            reasoning_mode=ReasoningMode.DEDUCTIVE,
            computation_time=(datetime.now() - start_time).total_seconds(),
            metadata={"method": method, "uncertainty_type": "probabilistic"}
        )
        
        self.reasoning_history.append(reasoning_result)
        self._update_metrics(reasoning_result)
        
        logger.info(f"🔮 Probabilistic query '{query}' using {method}: confidence={result['confidence']:.3f}")
        return reasoning_result
    
    async def causal_reasoning(self, query: str, intervention: Optional[Dict[str, bool]] = None) -> ReasoningResult:
        """Perform causal reasoning with optional interventions"""
        start_time = datetime.now()
        
        # Parse query to identify causal relationships
        causal_chain = self._identify_causal_chain(query)
        
        # Apply interventions if provided
        if intervention:
            modified_relations = self._apply_interventions(causal_chain, intervention)
        else:
            modified_relations = causal_chain
        
        # Calculate causal effect
        causal_effect = self._calculate_causal_effect(modified_relations)
        
        # Generate explanation
        explanation = await self._generate_causal_explanation(query, causal_chain, causal_effect, intervention)
        
        result = ReasoningResult(
            query=query,
            answer=causal_effect,
            confidence=self._calculate_causal_confidence(causal_chain),
            explanation=explanation,
            reasoning_mode=ReasoningMode.CAUSAL,
            computation_time=(datetime.now() - start_time).total_seconds(),
            metadata={"causal_chain_length": len(causal_chain), "intervention": intervention}
        )
        
        self.reasoning_history.append(result)
        self._update_metrics(result)
        
        logger.info(f"🔮 Causal reasoning '{query}': effect={causal_effect:.3f}")
        return result
    
    async def abductive_reasoning(self, observation: str, possible_causes: Optional[List[str]] = None) -> ReasoningResult:
        """Perform abductive reasoning to find best explanation"""
        start_time = datetime.now()
        
        # Identify possible causes if not provided
        if possible_causes is None:
            possible_causes = self._identify_possible_causes(observation)
        
        # Evaluate each possible cause
        explanations = []
        for cause in possible_causes:
            explanation_quality = await self._evaluate_explanation(cause, observation)
            explanations.append({
                "cause": cause,
                "quality": explanation_quality,
                "probability": self._get_cause_probability(cause)
            })
        
        # Sort by explanation quality
        explanations.sort(key=lambda x: x["quality"], reverse=True)
        
        # Select best explanation
        best_explanation = explanations[0] if explanations else None
        
        # Generate detailed explanation
        detailed_explanation = await self._generate_abductive_explanation(
            observation, explanations, best_explanation
        )
        
        result = ReasoningResult(
            query=f"explain: {observation}",
            answer=best_explanation,
            confidence=best_explanation["quality"] if best_explanation else 0.0,
            explanation=detailed_explanation,
            reasoning_mode=ReasoningMode.ABDUCTIVE,
            computation_time=(datetime.now() - start_time).total_seconds(),
            metadata={"candidate_causes": len(possible_causes), "evaluated_explanations": len(explanations)}
        )
        
        self.reasoning_history.append(result)
        self._update_metrics(result)
        
        logger.info(f"🔮 Abductive reasoning for '{observation}': best explanation found")
        return result
    
    async def analogical_reasoning(self, source_case: str, target_case: str, 
                                 similarity_threshold: float = 0.7) -> ReasoningResult:
        """Perform reasoning by analogy"""
        start_time = datetime.now()
        
        # Calculate structural similarity
        similarity = await self._calculate_structural_similarity(source_case, target_case)
        
        if similarity < similarity_threshold:
            logger.warning(f"Low similarity ({similarity:.3f}) between cases")
        
        # Transfer knowledge from source to target
        transferred_knowledge = await self._transfer_analogical_knowledge(source_case, target_case, similarity)
        
        # Generate explanation
        explanation = await self._generate_analogical_explanation(
            source_case, target_case, similarity, transferred_knowledge
        )
        
        result = ReasoningResult(
            query=f"analogy: {source_case} → {target_case}",
            answer=transferred_knowledge,
            confidence=similarity,
            explanation=explanation,
            reasoning_mode=ReasoningMode.ANALOGICAL,
            computation_time=(datetime.now() - start_time).total_seconds(),
            metadata={"similarity_score": similarity, "similarity_threshold": similarity_threshold}
        )
        
        self.reasoning_history.append(result)
        self._update_metrics(result)
        
        logger.info(f"🔮 Analogical reasoning: similarity={similarity:.3f}")
        return result
    
    async def counterfactual_reasoning(self, factual: str, counterfactual_condition: str) -> ReasoningResult:
        """Perform counterfactual reasoning (what-if analysis)"""
        start_time = datetime.now()
        
        # Create counterfactual world
        counterfactual_reasoner = await self._create_counterfactual_world(factual, counterfactual_condition)
        
        # Evaluate consequences in counterfactual world
        consequences = await self._evaluate_counterfactual_consequences(counterfactual_reasoner, factual)
        
        # Compare with actual world
        comparison = await self._compare_factual_counterfactual(factual, consequences)
        
        # Generate explanation
        explanation = await self._generate_counterfactual_explanation(
            factual, counterfactual_condition, consequences, comparison
        )
        
        result = ReasoningResult(
            query=f"what-if: {counterfactual_condition}",
            answer=comparison,
            confidence=self._calculate_counterfactual_confidence(consequences),
            explanation=explanation,
            reasoning_mode=ReasoningMode.COUNTERFACTUAL,
            computation_time=(datetime.now() - start_time).total_seconds(),
            metadata={"factual": factual, "counterfactual_condition": counterfactual_condition}
        )
        
        self.reasoning_history.append(result)
        self._update_metrics(result)
        
        logger.info(f"🔮 Counterfactual reasoning: {counterfactual_condition}")
        return result
    
    async def explain_reasoning(self, query: str, depth: int = 3) -> Explanation:
        """Generate detailed explanation for a reasoning process"""
        
        # Find relevant reasoning result
        relevant_result = self._find_relevant_reasoning_result(query)
        
        if not relevant_result:
            # Perform new reasoning if no history found
            result = await self.probabilistic_query(query)
            relevant_result = result
        
        # Generate detailed explanation
        explanation = await self._generate_detailed_explanation(relevant_result, depth)
        
        self.metrics["explanation_requests"] += 1
        
        logger.info(f"🔮 Generated explanation for '{query}' (depth={depth})")
        return explanation
    
    # Private helper methods
    
    def _calculate_temporal_confidence(self, base_result: List, query_time: datetime) -> float:
        """Calculate confidence based on temporal validity"""
        if not base_result:
            return 0.0
        
        # Weight by how many temporal facts support the result
        supporting_facts = [
            fact for fact in self.temporal_facts
            if fact.is_valid_at(query_time) and any(fact.predicate in str(r) for r in base_result)
        ]
        
        if not supporting_facts:
            return 0.5  # Default confidence for non-temporal facts
        
        # Average confidence of supporting facts
        return mean([fact.confidence for fact in supporting_facts])
    
    async def _generate_temporal_explanation(self, query: str, result: List, 
                                           query_time: datetime, valid_facts: List[TemporalFact]) -> Explanation:
        """Generate explanation for temporal reasoning"""
        
        reasoning_chain = [
            f"Temporal query at {query_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Found {len(valid_facts)} facts valid at query time",
            f"Applied temporal reasoning to derive {len(result)} results"
        ]
        
        supporting_facts = [fact.predicate for fact in valid_facts]
        
        return Explanation(
            conclusion=f"Query '{query}' has {len(result)} results at {query_time}",
            reasoning_chain=reasoning_chain,
            confidence=self._calculate_temporal_confidence(result, query_time),
            supporting_facts=supporting_facts
        )
    
    async def _bayesian_inference(self, query: str) -> Dict[str, Any]:
        """Perform Bayesian inference"""
        
        # Find relevant probabilistic facts
        relevant_facts = [
            fact for fact in self.probabilistic_facts
            if query.lower() in fact.predicate.lower()
        ]
        
        if not relevant_facts:
            return {
                "answer": [],
                "confidence": 0.0,
                "explanation": Explanation(
                    conclusion="No probabilistic facts found for query",
                    reasoning_chain=["No matching probabilistic facts"],
                    confidence=0.0,
                    supporting_facts=[]
                )
            }
        
        # Simple Bayesian calculation (can be extended)
        prior = relevant_facts[0].probability
        likelihood = 0.8  # Simplified likelihood
        evidence = 0.6    # Simplified evidence
        
        posterior = (likelihood * prior) / evidence
        posterior = max(0.0, min(1.0, posterior))  # Clamp to [0,1]
        
        explanation = Explanation(
            conclusion=f"Bayesian inference yields probability {posterior:.3f}",
            reasoning_chain=[
                f"Prior probability: {prior:.3f}",
                f"Likelihood: {likelihood:.3f}", 
                f"Evidence: {evidence:.3f}",
                f"Posterior = (likelihood × prior) / evidence = {posterior:.3f}"
            ],
            confidence=posterior,
            supporting_facts=[fact.predicate for fact in relevant_facts]
        )
        
        return {
            "answer": [{"predicate": query, "probability": posterior}],
            "confidence": posterior,
            "explanation": explanation
        }
    
    async def _fuzzy_inference(self, query: str) -> Dict[str, Any]:
        """Perform fuzzy logic inference"""
        
        # Find relevant facts and convert to fuzzy values
        fuzzy_facts = []
        for fact in self.probabilistic_facts:
            if fact.uncertainty_type == UncertaintyType.FUZZY:
                fuzzy_facts.append(fact)
        
        # Simple fuzzy inference (can be extended with proper fuzzy operators)
        if fuzzy_facts:
            # Use minimum t-norm for conjunction
            fuzzy_result = min([fact.probability for fact in fuzzy_facts])
        else:
            fuzzy_result = 0.0
        
        explanation = Explanation(
            conclusion=f"Fuzzy inference yields membership degree {fuzzy_result:.3f}",
            reasoning_chain=[
                f"Applied fuzzy logic operators to {len(fuzzy_facts)} facts",
                f"Used minimum t-norm for conjunction",
                f"Result: {fuzzy_result:.3f}"
            ],
            confidence=fuzzy_result,
            supporting_facts=[fact.predicate for fact in fuzzy_facts]
        )
        
        return {
            "answer": [{"predicate": query, "membership": fuzzy_result}],
            "confidence": fuzzy_result,
            "explanation": explanation
        }
    
    async def _monte_carlo_inference(self, query: str) -> Dict[str, Any]:
        """Perform Monte Carlo sampling for inference"""
        
        num_samples = 1000
        successful_samples = 0
        
        # Monte Carlo sampling
        for _ in range(num_samples):
            # Sample from probabilistic facts
            sampled_world = self._sample_world()
            
            # Check if query holds in sampled world
            if self._query_holds_in_world(query, sampled_world):
                successful_samples += 1
        
        probability = successful_samples / num_samples
        
        explanation = Explanation(
            conclusion=f"Monte Carlo estimation: {probability:.3f}",
            reasoning_chain=[
                f"Performed {num_samples} samples",
                f"Query held in {successful_samples} samples",
                f"Estimated probability: {probability:.3f}"
            ],
            confidence=probability,
            supporting_facts=[f"Based on {len(self.probabilistic_facts)} probabilistic facts"]
        )
        
        return {
            "answer": [{"predicate": query, "probability": probability}],
            "confidence": probability,
            "explanation": explanation
        }
    
    def _sample_world(self) -> Dict[str, bool]:
        """Sample a possible world from probabilistic facts"""
        world = {}
        for fact in self.probabilistic_facts:
            world[fact.predicate] = random.random() < fact.probability
        return world
    
    def _query_holds_in_world(self, query: str, world: Dict[str, bool]) -> bool:
        """Check if query holds in a sampled world"""
        # Simplified check - can be extended with proper logical evaluation
        return world.get(query, False)
    
    def _identify_causal_chain(self, query: str) -> List[CausalRelation]:
        """Identify causal chain relevant to query"""
        relevant_relations = []
        for relation in self.causal_relations:
            if query.lower() in relation.cause.lower() or query.lower() in relation.effect.lower():
                relevant_relations.append(relation)
        return relevant_relations
    
    def _apply_interventions(self, causal_chain: List[CausalRelation], 
                           intervention: Dict[str, bool]) -> List[CausalRelation]:
        """Apply interventions to causal chain"""
        # Simplified intervention application
        modified_chain = []
        for relation in causal_chain:
            if relation.cause not in intervention:
                modified_chain.append(relation)
        return modified_chain
    
    def _calculate_causal_effect(self, causal_chain: List[CausalRelation]) -> float:
        """Calculate overall causal effect"""
        if not causal_chain:
            return 0.0
        
        # Simplified calculation - product of strengths
        effect = 1.0
        for relation in causal_chain:
            effect *= relation.strength
        
        return effect
    
    def _calculate_causal_confidence(self, causal_chain: List[CausalRelation]) -> float:
        """Calculate confidence in causal reasoning"""
        if not causal_chain:
            return 0.0
        return mean([relation.strength for relation in causal_chain])
    
    async def _generate_causal_explanation(self, query: str, causal_chain: List[CausalRelation],
                                         causal_effect: float, intervention: Optional[Dict[str, bool]]) -> Explanation:
        """Generate explanation for causal reasoning"""
        
        reasoning_chain = [f"Identified {len(causal_chain)} causal relations"]
        
        for relation in causal_chain:
            reasoning_chain.append(f"{relation.cause} → {relation.effect} (strength: {relation.strength:.3f})")
        
        if intervention:
            reasoning_chain.append(f"Applied intervention: {intervention}")
        
        reasoning_chain.append(f"Calculated causal effect: {causal_effect:.3f}")
        
        return Explanation(
            conclusion=f"Causal effect: {causal_effect:.3f}",
            reasoning_chain=reasoning_chain,
            confidence=self._calculate_causal_confidence(causal_chain),
            supporting_facts=[f"{r.cause} → {r.effect}" for r in causal_chain]
        )
    
    def _identify_possible_causes(self, observation: str) -> List[str]:
        """Identify possible causes for an observation"""
        possible_causes = []
        
        # Check causal relations where observation is the effect
        for relation in self.causal_relations:
            if observation.lower() in relation.effect.lower():
                possible_causes.append(relation.cause)
        
        # Check rules in base reasoner
        for rule in self.base_reasoner.knowledge_base.rules:
            rule_str = str(rule)
            if observation.lower() in rule_str.lower():
                # Extract potential causes from rule conditions
                # Simplified extraction
                if "IF" in rule_str and "THEN" in rule_str:
                    try:
                        condition_part = rule_str.split("IF")[1].split("THEN")[0].strip()
                        possible_causes.append(condition_part)
                    except IndexError:
                        # Skip malformed rules
                        continue
        
        return list(set(possible_causes))  # Remove duplicates
    
    async def _evaluate_explanation(self, cause: str, observation: str) -> float:
        """Evaluate quality of an explanation"""
        
        # Check if there's a direct causal relation
        direct_causal = any(
            r.cause == cause and observation.lower() in r.effect.lower()
            for r in self.causal_relations
        )
        
        if direct_causal:
            return 0.9
        
        # Check if base reasoner can derive observation from cause
        temp_reasoner = SymbolicReasoner()
        temp_reasoner.add_fact(cause, 1.0)
        
        # Copy rules
        for rule in self.base_reasoner.knowledge_base.rules:
            temp_reasoner.knowledge_base.rules.append(rule)
        
        result = await temp_reasoner.query(observation)
        
        return 0.7 if result else 0.3
    
    def _get_cause_probability(self, cause: str) -> float:
        """Get prior probability of a cause"""
        for fact in self.probabilistic_facts:
            if cause.lower() in fact.predicate.lower():
                return fact.probability
        return 0.5  # Default prior
    
    async def _generate_abductive_explanation(self, observation: str, explanations: List[Dict],
                                            best_explanation: Optional[Dict]) -> Explanation:
        """Generate explanation for abductive reasoning"""
        
        if not best_explanation:
            return Explanation(
                conclusion="No suitable explanation found",
                reasoning_chain=["No candidate causes could explain the observation"],
                confidence=0.0,
                supporting_facts=[]
            )
        
        reasoning_chain = [
            f"Observation: {observation}",
            f"Evaluated {len(explanations)} possible causes",
            f"Best explanation: {best_explanation['cause']}",
            f"Explanation quality: {best_explanation['quality']:.3f}",
            f"Prior probability: {best_explanation['probability']:.3f}"
        ]
        
        # Add alternative explanations
        alternatives = [exp['cause'] for exp in explanations[1:4]]  # Top 3 alternatives
        
        return Explanation(
            conclusion=f"Best explanation for '{observation}': {best_explanation['cause']}",
            reasoning_chain=reasoning_chain,
            confidence=best_explanation['quality'],
            supporting_facts=[best_explanation['cause']],
            alternative_explanations=alternatives
        )
    
    async def _calculate_structural_similarity(self, source_case: str, target_case: str) -> float:
        """Calculate structural similarity between cases"""
        
        # Simplified similarity calculation based on shared predicates/terms
        source_terms = set(source_case.lower().split())
        target_terms = set(target_case.lower().split())
        
        if not source_terms or not target_terms:
            return 0.0
        
        # Jaccard similarity
        intersection = len(source_terms.intersection(target_terms))
        union = len(source_terms.union(target_terms))
        
        return intersection / union if union > 0 else 0.0
    
    async def _transfer_analogical_knowledge(self, source_case: str, target_case: str, similarity: float) -> Dict[str, Any]:
        """Transfer knowledge from source to target case"""
        
        # Find facts and rules related to source case
        source_facts = []
        source_rules = []
        
        for fact in self.base_reasoner.knowledge_base.facts:
            if any(term in str(fact).lower() for term in source_case.lower().split()):
                source_facts.append(str(fact))
        
        for rule in self.base_reasoner.knowledge_base.rules:
            if any(term in str(rule).lower() for term in source_case.lower().split()):
                source_rules.append(str(rule))
        
        # Transfer with similarity weighting
        transferred = {
            "facts": [(fact, similarity) for fact in source_facts],
            "rules": [(rule, similarity) for rule in source_rules],
            "similarity_score": similarity
        }
        
        return transferred
    
    async def _generate_analogical_explanation(self, source_case: str, target_case: str,
                                             similarity: float, transferred_knowledge: Dict) -> Explanation:
        """Generate explanation for analogical reasoning"""
        
        reasoning_chain = [
            f"Source case: {source_case}",
            f"Target case: {target_case}",
            f"Structural similarity: {similarity:.3f}",
            f"Transferred {len(transferred_knowledge['facts'])} facts",
            f"Transferred {len(transferred_knowledge['rules'])} rules"
        ]
        
        return Explanation(
            conclusion=f"Analogical transfer with similarity {similarity:.3f}",
            reasoning_chain=reasoning_chain,
            confidence=similarity,
            supporting_facts=[f"Source: {source_case}", f"Target: {target_case}"]
        )
    
    async def _create_counterfactual_world(self, factual: str, counterfactual_condition: str) -> SymbolicReasoner:
        """Create a counterfactual world with modified conditions"""
        
        # Create new reasoner for counterfactual world
        cf_reasoner = SymbolicReasoner()
        
        # Copy all facts except those contradicted by counterfactual condition
        for fact in self.base_reasoner.knowledge_base.facts:
            if not self._contradicts_condition(str(fact), counterfactual_condition):
                cf_reasoner.add_fact(str(fact), fact.confidence)
        
        # Add counterfactual condition
        cf_reasoner.add_fact(counterfactual_condition, 1.0)
        
        # Copy rules
        for rule in self.base_reasoner.knowledge_base.rules:
            cf_reasoner.knowledge_base.rules.append(rule)
        
        return cf_reasoner
    
    def _contradicts_condition(self, fact: str, condition: str) -> bool:
        """Check if fact contradicts counterfactual condition"""
        # Simplified contradiction check
        # In practice, this would need more sophisticated logical analysis
        return False
    
    async def _evaluate_counterfactual_consequences(self, cf_reasoner: SymbolicReasoner, original_factual: str) -> List[str]:
        """Evaluate consequences in counterfactual world"""
        
        # Query the counterfactual world for various consequences
        consequences = []
        
        # Query for the original factual statement
        result = await cf_reasoner.query(original_factual)
        consequences.extend([str(r) for r in result])
        
        return consequences
    
    async def _compare_factual_counterfactual(self, factual: str, cf_consequences: List[str]) -> Dict[str, Any]:
        """Compare factual and counterfactual outcomes"""
        
        # Get factual consequences
        factual_result = await self.base_reasoner.query(factual)
        
        comparison = {
            "factual_holds": len(factual_result) > 0,
            "counterfactual_holds": len(cf_consequences) > 0,
            "difference": len(factual_result) - len(cf_consequences),
            "factual_consequences": [str(r) for r in factual_result],
            "counterfactual_consequences": cf_consequences
        }
        
        return comparison
    
    def _calculate_counterfactual_confidence(self, consequences: List[str]) -> float:
        """Calculate confidence in counterfactual reasoning"""
        # Simplified confidence calculation
        return 0.8 if consequences else 0.2
    
    async def _generate_counterfactual_explanation(self, factual: str, counterfactual_condition: str,
                                                 consequences: List[str], comparison: Dict) -> Explanation:
        """Generate explanation for counterfactual reasoning"""
        
        reasoning_chain = [
            f"Factual: {factual}",
            f"Counterfactual condition: {counterfactual_condition}",
            f"In factual world: {len(comparison['factual_consequences'])} consequences",
            f"In counterfactual world: {len(consequences)} consequences",
            f"Difference: {comparison['difference']}"
        ]
        
        return Explanation(
            conclusion=f"Counterfactual analysis: {counterfactual_condition}",
            reasoning_chain=reasoning_chain,
            confidence=self._calculate_counterfactual_confidence(consequences),
            supporting_facts=[factual, counterfactual_condition]
        )
    
    def _find_relevant_reasoning_result(self, query: str) -> Optional[ReasoningResult]:
        """Find relevant reasoning result from history"""
        for result in reversed(self.reasoning_history):
            if query.lower() in result.query.lower():
                return result
        return None
    
    async def _generate_detailed_explanation(self, result: ReasoningResult, depth: int) -> Explanation:
        """Generate detailed explanation with specified depth"""
        
        reasoning_chain = result.explanation.reasoning_chain.copy()
        
        # Add more detail based on depth
        if depth > 1:
            reasoning_chain.extend([
                f"Reasoning mode: {result.reasoning_mode.value}",
                f"Computation time: {result.computation_time:.3f} seconds",
                f"Metadata: {result.metadata}"
            ])
        
        if depth > 2:
            # Add supporting evidence detail
            reasoning_chain.extend([
                f"Supporting facts: {len(result.explanation.supporting_facts)}",
                f"Assumptions made: {len(result.explanation.assumptions)}",
                f"Alternative explanations: {len(result.explanation.alternative_explanations)}"
            ])
        
        return Explanation(
            conclusion=result.explanation.conclusion,
            reasoning_chain=reasoning_chain,
            confidence=result.confidence,
            supporting_facts=result.explanation.supporting_facts,
            assumptions=result.explanation.assumptions,
            alternative_explanations=result.explanation.alternative_explanations
        )
    
    def _update_metrics(self, result: ReasoningResult):
        """Update performance metrics"""
        self.metrics["total_inferences"] += 1
        
        if result.confidence >= self.config["confidence_threshold"]:
            self.metrics["successful_inferences"] += 1
        
        # Update running average confidence
        current_avg = self.metrics["average_confidence"]
        total = self.metrics["total_inferences"]
        self.metrics["average_confidence"] = (current_avg * (total - 1) + result.confidence) / total
        
        self.metrics["reasoning_modes_used"][result.reasoning_mode.value] += 1
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reasoning statistics"""
        
        success_rate = (self.metrics["successful_inferences"] / 
                       max(1, self.metrics["total_inferences"]))
        
        return {
            "total_inferences": self.metrics["total_inferences"],
            "successful_inferences": self.metrics["successful_inferences"],
            "success_rate": success_rate,
            "average_confidence": self.metrics["average_confidence"],
            "reasoning_modes_used": dict(self.metrics["reasoning_modes_used"]),
            "explanation_requests": self.metrics["explanation_requests"],
            "temporal_facts": len(self.temporal_facts),
            "probabilistic_facts": len(self.probabilistic_facts),
            "causal_relations": len(self.causal_relations),
            "reasoning_history_length": len(self.reasoning_history)
        }


# Factory function
def create_advanced_reasoner(base_reasoner: Optional[SymbolicReasoner] = None) -> AdvancedReasoner:
    """Create an advanced reasoning engine"""
    return AdvancedReasoner(base_reasoner)