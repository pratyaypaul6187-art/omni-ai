"""
🧠 REASONING ENGINE
Advanced logical reasoning and problem-solving capabilities
"""

import re
import json
import time
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

from structlog import get_logger

logger = get_logger()

class ReasoningType(Enum):
    DEDUCTIVE = "deductive"    # General to specific
    INDUCTIVE = "inductive"    # Specific to general
    ABDUCTIVE = "abductive"    # Best explanation
    ANALOGICAL = "analogical"  # Pattern matching
    CAUSAL = "causal"         # Cause and effect
    LOGICAL = "logical"       # Formal logic
    CREATIVE = "creative"     # Outside-the-box thinking

@dataclass
class ReasoningStep:
    """A single step in a reasoning chain"""
    step_id: str
    reasoning_type: ReasoningType
    premise: str
    conclusion: str
    confidence: float  # 0.0 to 1.0
    evidence: List[str]
    assumptions: List[str]
    timestamp: datetime

@dataclass
class ReasoningChain:
    """A complete chain of reasoning"""
    chain_id: str
    problem: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    reasoning_path: str
    started_at: datetime
    completed_at: datetime

class ReasoningEngine:
    """🧠 Advanced reasoning and logical thinking engine"""
    
    def __init__(self):
        self.reasoning_chains = {}  # Store reasoning histories
        self.knowledge_base = {}    # Facts and rules
        self.inference_rules = []   # Logical inference rules
        self.pattern_library = {}   # Known patterns for analogical reasoning
        self.causal_models = {}     # Cause-effect relationships
        
        # Reasoning statistics
        self.reasoning_stats = defaultdict(int)
        self.successful_patterns = defaultdict(float)
        
        self._initialize_knowledge_base()
        self._initialize_inference_rules()
        
        logger.info("🧠 Reasoning engine initialized")
    
    def _initialize_knowledge_base(self):
        """Initialize basic knowledge base"""
        self.knowledge_base = {
            # Basic facts
            "mathematics": {
                "rules": [
                    "If a > b and b > c, then a > c",
                    "If x = y and y = z, then x = z",
                    "The sum of angles in a triangle is 180 degrees"
                ]
            },
            "logic": {
                "rules": [
                    "If P implies Q and P is true, then Q is true",
                    "If P implies Q and Q is false, then P is false",
                    "If not P or Q, and P is true, then Q is true"
                ]
            },
            "common_sense": {
                "facts": [
                    "Fire is hot",
                    "Water freezes at 0°C",
                    "Humans need air to breathe",
                    "Objects fall due to gravity"
                ]
            }
        }
    
    def _initialize_inference_rules(self):
        """Initialize logical inference rules"""
        self.inference_rules = [
            {
                "name": "modus_ponens",
                "pattern": r"If (.+) then (.+)\. (.+)\.",
                "rule": "If P then Q. P. Therefore Q."
            },
            {
                "name": "modus_tollens", 
                "pattern": r"If (.+) then (.+)\. Not (.+)\.",
                "rule": "If P then Q. Not Q. Therefore not P."
            },
            {
                "name": "hypothetical_syllogism",
                "pattern": r"If (.+) then (.+)\. If (.+) then (.+)\.",
                "rule": "If P then Q. If Q then R. Therefore if P then R."
            },
            {
                "name": "disjunctive_syllogism",
                "pattern": r"(.+) or (.+)\. Not (.+)\.",
                "rule": "P or Q. Not P. Therefore Q."
            }
        ]
    
    def reason_about(self, problem: str, reasoning_types: List[ReasoningType] = None) -> ReasoningChain:
        """🧠 Main reasoning function - think through a problem"""
        
        if reasoning_types is None:
            reasoning_types = [ReasoningType.LOGICAL, ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE]
        
        chain_id = f"reasoning_{int(time.time())}"
        reasoning_chain = ReasoningChain(
            chain_id=chain_id,
            problem=problem,
            steps=[],
            final_conclusion="",
            overall_confidence=0.0,
            reasoning_path="",
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        logger.info(f"🧠 Starting reasoning about: {problem}")
        
        # Apply different reasoning approaches
        for reasoning_type in reasoning_types:
            try:
                if reasoning_type == ReasoningType.DEDUCTIVE:
                    steps = self._deductive_reasoning(problem)
                elif reasoning_type == ReasoningType.INDUCTIVE:
                    steps = self._inductive_reasoning(problem)
                elif reasoning_type == ReasoningType.ABDUCTIVE:
                    steps = self._abductive_reasoning(problem)
                elif reasoning_type == ReasoningType.ANALOGICAL:
                    steps = self._analogical_reasoning(problem)
                elif reasoning_type == ReasoningType.CAUSAL:
                    steps = self._causal_reasoning(problem)
                elif reasoning_type == ReasoningType.LOGICAL:
                    steps = self._logical_reasoning(problem)
                elif reasoning_type == ReasoningType.CREATIVE:
                    steps = self._creative_reasoning(problem)
                else:
                    steps = []
                
                reasoning_chain.steps.extend(steps)
                
            except Exception as e:
                logger.error(f"Error in {reasoning_type.value} reasoning: {e}")
        
        # Synthesize final conclusion
        reasoning_chain.final_conclusion = self._synthesize_conclusion(reasoning_chain.steps)
        reasoning_chain.overall_confidence = self._calculate_overall_confidence(reasoning_chain.steps)
        reasoning_chain.reasoning_path = self._generate_reasoning_path(reasoning_chain.steps)
        reasoning_chain.completed_at = datetime.now()
        
        # Store reasoning chain
        self.reasoning_chains[chain_id] = reasoning_chain
        
        # Update statistics
        self.reasoning_stats["total_reasoning_sessions"] += 1
        for step in reasoning_chain.steps:
            self.reasoning_stats[f"{step.reasoning_type.value}_steps"] += 1
        
        logger.info(f"🧠 Completed reasoning: {chain_id}")
        return reasoning_chain
    
    def _deductive_reasoning(self, problem: str) -> List[ReasoningStep]:
        """Deductive reasoning: General principles to specific conclusions"""
        steps = []
        
        # Look for general rules that apply to this problem
        relevant_rules = self._find_relevant_rules(problem)
        
        for rule in relevant_rules:
            if self._rule_applies(rule, problem):
                step = ReasoningStep(
                    step_id=f"deductive_{len(steps)}",
                    reasoning_type=ReasoningType.DEDUCTIVE,
                    premise=f"General rule: {rule}",
                    conclusion=self._apply_rule_to_problem(rule, problem),
                    confidence=0.8,
                    evidence=[rule],
                    assumptions=[f"Rule '{rule}' is valid and applicable"],
                    timestamp=datetime.now()
                )
                steps.append(step)
        
        return steps
    
    def _inductive_reasoning(self, problem: str) -> List[ReasoningStep]:
        """Inductive reasoning: Specific observations to general patterns"""
        steps = []
        
        # Look for patterns in the problem
        patterns = self._identify_patterns(problem)
        
        for pattern in patterns:
            step = ReasoningStep(
                step_id=f"inductive_{len(steps)}",
                reasoning_type=ReasoningType.INDUCTIVE,
                premise=f"Observed pattern: {pattern}",
                conclusion=self._generalize_from_pattern(pattern, problem),
                confidence=0.6,  # Inductive reasoning is less certain
                evidence=[f"Pattern: {pattern}"],
                assumptions=["Pattern is representative of broader trend"],
                timestamp=datetime.now()
            )
            steps.append(step)
        
        return steps
    
    def _abductive_reasoning(self, problem: str) -> List[ReasoningStep]:
        """Abductive reasoning: Best explanation for observations"""
        steps = []
        
        # Generate possible explanations
        explanations = self._generate_explanations(problem)
        
        # Rank explanations by plausibility
        ranked_explanations = self._rank_explanations(explanations, problem)
        
        for i, (explanation, score) in enumerate(ranked_explanations[:3]):  # Top 3
            step = ReasoningStep(
                step_id=f"abductive_{i}",
                reasoning_type=ReasoningType.ABDUCTIVE,
                premise=f"Observation: {problem}",
                conclusion=f"Best explanation: {explanation}",
                confidence=score,
                evidence=[f"Explanation plausibility: {score:.2f}"],
                assumptions=["Available explanations are comprehensive"],
                timestamp=datetime.now()
            )
            steps.append(step)
        
        return steps
    
    def _analogical_reasoning(self, problem: str) -> List[ReasoningStep]:
        """Analogical reasoning: Reasoning by similarity to known cases"""
        steps = []
        
        # Find similar problems in pattern library
        similar_problems = self._find_analogous_problems(problem)
        
        for similar_problem, similarity_score in similar_problems:
            if similarity_score > 0.3:  # Threshold for useful analogy
                step = ReasoningStep(
                    step_id=f"analogical_{len(steps)}",
                    reasoning_type=ReasoningType.ANALOGICAL,
                    premise=f"Similar problem: {similar_problem['problem']}",
                    conclusion=f"By analogy: {self._adapt_solution(similar_problem['solution'], problem)}",
                    confidence=similarity_score,
                    evidence=[f"Similarity score: {similarity_score:.2f}"],
                    assumptions=["Analogous situations have similar solutions"],
                    timestamp=datetime.now()
                )
                steps.append(step)
        
        return steps
    
    def _causal_reasoning(self, problem: str) -> List[ReasoningStep]:
        """Causal reasoning: Understanding cause-effect relationships"""
        steps = []
        
        # Identify potential causes and effects
        causes = self._identify_causes(problem)
        effects = self._identify_effects(problem)
        
        for cause in causes:
            for effect in effects:
                if self._has_causal_relationship(cause, effect):
                    step = ReasoningStep(
                        step_id=f"causal_{len(steps)}",
                        reasoning_type=ReasoningType.CAUSAL,
                        premise=f"Cause: {cause}",
                        conclusion=f"Effect: {effect}",
                        confidence=0.7,
                        evidence=[f"Causal link between {cause} and {effect}"],
                        assumptions=["Causal relationships are consistent"],
                        timestamp=datetime.now()
                    )
                    steps.append(step)
        
        return steps
    
    def _logical_reasoning(self, problem: str) -> List[ReasoningStep]:
        """Formal logical reasoning using inference rules"""
        steps = []
        
        # Apply inference rules
        for rule in self.inference_rules:
            matches = re.findall(rule["pattern"], problem, re.IGNORECASE)
            
            if matches:
                for match in matches:
                    step = ReasoningStep(
                        step_id=f"logical_{len(steps)}",
                        reasoning_type=ReasoningType.LOGICAL,
                        premise=f"Premises: {match}",
                        conclusion=self._apply_inference_rule(rule, match),
                        confidence=0.9,  # Logical reasoning is highly certain
                        evidence=[f"Inference rule: {rule['name']}"],
                        assumptions=["Logical rules are valid"],
                        timestamp=datetime.now()
                    )
                    steps.append(step)
        
        return steps
    
    def _creative_reasoning(self, problem: str) -> List[ReasoningStep]:
        """Creative reasoning: Thinking outside the box"""
        steps = []
        
        # Generate creative associations
        creative_ideas = self._generate_creative_ideas(problem)
        
        for idea in creative_ideas:
            step = ReasoningStep(
                step_id=f"creative_{len(steps)}",
                reasoning_type=ReasoningType.CREATIVE,
                premise=f"Creative association: {idea['association']}",
                conclusion=f"Novel insight: {idea['insight']}",
                confidence=0.4,  # Creative reasoning is speculative
                evidence=[f"Creative connection: {idea['connection']}"],
                assumptions=["Creative leaps can yield valuable insights"],
                timestamp=datetime.now()
            )
            steps.append(step)
        
        return steps
    
    def _find_relevant_rules(self, problem: str) -> List[str]:
        """Find rules relevant to the problem"""
        relevant = []
        problem_lower = problem.lower()
        
        for domain, content in self.knowledge_base.items():
            if "rules" in content:
                for rule in content["rules"]:
                    rule_keywords = set(rule.lower().split())
                    problem_keywords = set(problem_lower.split())
                    
                    if len(rule_keywords & problem_keywords) > 0:
                        relevant.append(rule)
        
        return relevant
    
    def _rule_applies(self, rule: str, problem: str) -> bool:
        """Check if a rule applies to the problem"""
        # Simple keyword matching (can be enhanced)
        rule_words = set(rule.lower().split())
        problem_words = set(problem.lower().split())
        return len(rule_words & problem_words) >= 2
    
    def _apply_rule_to_problem(self, rule: str, problem: str) -> str:
        """Apply a rule to generate a conclusion"""
        return f"Applying '{rule}' to '{problem}' suggests a logical conclusion based on established principles"
    
    def _identify_patterns(self, problem: str) -> List[str]:
        """Identify patterns in the problem"""
        patterns = []
        
        # Look for numerical patterns
        numbers = re.findall(r'\d+', problem)
        if len(numbers) > 1:
            patterns.append(f"Numerical sequence: {numbers}")
        
        # Look for repetitive words
        words = problem.lower().split()
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        repeated = [word for word, count in word_counts.items() if count > 1]
        if repeated:
            patterns.append(f"Repeated terms: {repeated}")
        
        # Look for conditional structures
        if re.search(r'\bif\b.*\bthen\b', problem, re.IGNORECASE):
            patterns.append("Conditional logic structure")
        
        return patterns
    
    def _generalize_from_pattern(self, pattern: str, problem: str) -> str:
        """Generate a generalization from observed pattern"""
        return f"Based on the pattern '{pattern}', we can infer a general principle that applies beyond this specific case"
    
    def _generate_explanations(self, problem: str) -> List[str]:
        """Generate possible explanations for the problem"""
        explanations = []
        
        # Simple explanation generation based on keywords
        if "why" in problem.lower():
            explanations.extend([
                "Causal mechanism at work",
                "Underlying system behavior", 
                "Natural consequence of conditions",
                "Result of multiple factors interacting"
            ])
        elif "how" in problem.lower():
            explanations.extend([
                "Sequential process unfolding",
                "Mechanism operating step by step",
                "System following established rules",
                "Emergent behavior from simple interactions"
            ])
        else:
            explanations.extend([
                "Direct causation",
                "Indirect causation through intermediaries",
                "Correlation without causation",
                "Random occurrence",
                "Systematic pattern"
            ])
        
        return explanations
    
    def _rank_explanations(self, explanations: List[str], problem: str) -> List[Tuple[str, float]]:
        """Rank explanations by plausibility"""
        ranked = []
        
        for explanation in explanations:
            # Simple scoring based on keyword matching and explanation type
            score = 0.5  # Base score
            
            explanation_lower = explanation.lower()
            problem_lower = problem.lower()
            
            # Boost score for relevant explanations
            if "causal" in explanation_lower and ("why" in problem_lower or "because" in problem_lower):
                score += 0.3
            if "process" in explanation_lower and ("how" in problem_lower):
                score += 0.3
            if "system" in explanation_lower and ("behavior" in problem_lower or "pattern" in problem_lower):
                score += 0.2
            
            ranked.append((explanation, score))
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    def _find_analogous_problems(self, problem: str) -> List[Tuple[Dict, float]]:
        """Find analogous problems in pattern library"""
        # For now, return some example analogies
        # In practice, this would search a comprehensive database
        analogies = [
            ({"problem": "Water flowing through pipes", "solution": "Apply pressure and clear blockages"}, 0.6),
            ({"problem": "Traffic flow optimization", "solution": "Identify bottlenecks and alternate routes"}, 0.5),
            ({"problem": "Information processing", "solution": "Break into smaller chunks and process sequentially"}, 0.4)
        ]
        
        return analogies
    
    def _adapt_solution(self, solution: str, problem: str) -> str:
        """Adapt a solution from analogous problem to current problem"""
        return f"Adapting solution '{solution}' to current context: {problem[:50]}..."
    
    def _identify_causes(self, problem: str) -> List[str]:
        """Identify potential causes mentioned in the problem"""
        causes = []
        
        # Look for causal keywords
        causal_patterns = [
            r'because of ([^,.!?]+)',
            r'due to ([^,.!?]+)', 
            r'caused by ([^,.!?]+)',
            r'results from ([^,.!?]+)'
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, problem, re.IGNORECASE)
            causes.extend(matches)
        
        return causes
    
    def _identify_effects(self, problem: str) -> List[str]:
        """Identify potential effects mentioned in the problem"""
        effects = []
        
        # Look for effect keywords
        effect_patterns = [
            r'results in ([^,.!?]+)',
            r'leads to ([^,.!?]+)',
            r'causes ([^,.!?]+)',
            r'produces ([^,.!?]+)'
        ]
        
        for pattern in effect_patterns:
            matches = re.findall(pattern, problem, re.IGNORECASE)
            effects.extend(matches)
        
        return effects
    
    def _has_causal_relationship(self, cause: str, effect: str) -> bool:
        """Check if there's a known causal relationship"""
        # Simple check - in practice this would use a causal knowledge base
        return len(cause.split()) > 0 and len(effect.split()) > 0
    
    def _apply_inference_rule(self, rule: Dict, premises: Tuple) -> str:
        """Apply logical inference rule to premises"""
        rule_name = rule["name"]
        
        if rule_name == "modus_ponens":
            return f"Therefore: {premises[1] if len(premises) > 1 else 'conclusion follows'}"
        elif rule_name == "modus_tollens":
            return f"Therefore: not {premises[0] if len(premises) > 0 else 'antecedent'}"
        else:
            return f"By {rule_name}: logical conclusion follows from premises"
    
    def _generate_creative_ideas(self, problem: str) -> List[Dict[str, str]]:
        """Generate creative ideas and associations"""
        ideas = []
        
        # Random associations (in practice, use more sophisticated methods)
        creative_associations = [
            {"association": "biological systems", "insight": "Self-organizing principles", "connection": "natural efficiency"},
            {"association": "musical harmony", "insight": "Balance and rhythm", "connection": "synchronized patterns"},
            {"association": "architectural design", "insight": "Form follows function", "connection": "structural integrity"},
            {"association": "ecosystem dynamics", "insight": "Interdependent relationships", "connection": "emergent stability"}
        ]
        
        return creative_associations[:2]  # Return a few creative ideas
    
    def _synthesize_conclusion(self, steps: List[ReasoningStep]) -> str:
        """Synthesize final conclusion from all reasoning steps"""
        if not steps:
            return "No conclusion could be reached from available reasoning."
        
        # Weight conclusions by confidence and reasoning type
        weighted_conclusions = []
        
        for step in steps:
            weight = step.confidence
            
            # Boost weight for certain reasoning types
            if step.reasoning_type == ReasoningType.LOGICAL:
                weight *= 1.2
            elif step.reasoning_type == ReasoningType.DEDUCTIVE:
                weight *= 1.1
            elif step.reasoning_type == ReasoningType.CREATIVE:
                weight *= 0.8  # Creative reasoning gets lower weight in synthesis
            
            weighted_conclusions.append((step.conclusion, weight))
        
        # Find the highest confidence conclusion
        best_conclusion = max(weighted_conclusions, key=lambda x: x[1])
        
        return f"Based on {len(steps)} reasoning steps, the most likely conclusion is: {best_conclusion[0]}"
    
    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence from all steps"""
        if not steps:
            return 0.0
        
        # Average confidence weighted by reasoning type reliability
        total_weight = 0
        weighted_confidence = 0
        
        for step in steps:
            # Weight by reasoning type reliability
            type_weight = {
                ReasoningType.LOGICAL: 1.0,
                ReasoningType.DEDUCTIVE: 0.9,
                ReasoningType.ABDUCTIVE: 0.7,
                ReasoningType.INDUCTIVE: 0.6,
                ReasoningType.CAUSAL: 0.8,
                ReasoningType.ANALOGICAL: 0.5,
                ReasoningType.CREATIVE: 0.3
            }.get(step.reasoning_type, 0.5)
            
            weighted_confidence += step.confidence * type_weight
            total_weight += type_weight
        
        return min(1.0, weighted_confidence / total_weight if total_weight > 0 else 0.0)
    
    def _generate_reasoning_path(self, steps: List[ReasoningStep]) -> str:
        """Generate human-readable reasoning path"""
        if not steps:
            return "No reasoning path available."
        
        path_parts = []
        for i, step in enumerate(steps, 1):
            path_parts.append(f"{i}. {step.reasoning_type.value.title()}: {step.premise} → {step.conclusion}")
        
        return "\n".join(path_parts)
    
    def self_evaluate_reasoning(self, chain_id: str) -> Dict[str, Any]:
        """🧠 Self-evaluate the quality of reasoning"""
        if chain_id not in self.reasoning_chains:
            return {"error": "Reasoning chain not found"}
        
        chain = self.reasoning_chains[chain_id]
        
        evaluation = {
            "chain_id": chain_id,
            "problem": chain.problem,
            "total_steps": len(chain.steps),
            "reasoning_types_used": list(set(step.reasoning_type.value for step in chain.steps)),
            "overall_confidence": chain.overall_confidence,
            "reasoning_duration": (chain.completed_at - chain.started_at).total_seconds(),
            "step_analysis": []
        }
        
        # Analyze each step
        for step in chain.steps:
            step_eval = {
                "step_id": step.step_id,
                "type": step.reasoning_type.value,
                "confidence": step.confidence,
                "evidence_quality": len(step.evidence),
                "assumption_count": len(step.assumptions),
                "premise_clarity": len(step.premise.split()) > 3,  # Simple clarity check
                "conclusion_specificity": len(step.conclusion.split()) > 5
            }
            evaluation["step_analysis"].append(step_eval)
        
        # Overall quality assessment
        avg_confidence = sum(step.confidence for step in chain.steps) / len(chain.steps)
        type_diversity = len(set(step.reasoning_type for step in chain.steps))
        
        evaluation["quality_metrics"] = {
            "average_step_confidence": avg_confidence,
            "reasoning_diversity": type_diversity,
            "logical_consistency": self._check_logical_consistency(chain.steps),
            "evidence_strength": sum(len(step.evidence) for step in chain.steps) / len(chain.steps)
        }
        
        return evaluation
    
    def _check_logical_consistency(self, steps: List[ReasoningStep]) -> float:
        """Check logical consistency across reasoning steps"""
        # Simple consistency check - look for contradictions
        conclusions = [step.conclusion.lower() for step in steps]
        
        # Check for obvious contradictions (very basic)
        contradiction_count = 0
        total_pairs = 0
        
        for i, conc1 in enumerate(conclusions):
            for j, conc2 in enumerate(conclusions[i+1:], i+1):
                total_pairs += 1
                
                # Look for contradictory keywords
                if ("not" in conc1 and "not" not in conc2) or ("not" not in conc1 and "not" in conc2):
                    # Check if they're about the same topic
                    words1 = set(conc1.replace("not ", "").split())
                    words2 = set(conc2.replace("not ", "").split())
                    
                    if len(words1 & words2) > 2:  # Significant overlap
                        contradiction_count += 1
        
        if total_pairs == 0:
            return 1.0
        
        consistency_score = 1.0 - (contradiction_count / total_pairs)
        return max(0.0, consistency_score)
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """🧠 Get reasoning engine statistics"""
        return {
            "total_reasoning_sessions": self.reasoning_stats["total_reasoning_sessions"],
            "reasoning_type_usage": {
                reasoning_type.value: self.reasoning_stats[f"{reasoning_type.value}_steps"]
                for reasoning_type in ReasoningType
            },
            "average_steps_per_session": sum(len(chain.steps) for chain in self.reasoning_chains.values()) / max(1, len(self.reasoning_chains)),
            "knowledge_base_size": sum(len(content.get("rules", [])) + len(content.get("facts", [])) for content in self.knowledge_base.values()),
            "successful_patterns": dict(self.successful_patterns)
        }