"""
🔹 SYMBOLIC REASONING ENGINE
Advanced symbolic reasoning with rule-based inference, logical operations,
and integration with neural networks for hybrid neurosymbolic AI
"""

import asyncio
import re
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from structlog import get_logger

logger = get_logger()

class LogicalOperator(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    IFF = "IFF"  # If and only if
    EXISTS = "EXISTS"
    FORALL = "FORALL"

class PredicateType(Enum):
    BINARY = "binary"      # is_a(X, Y)
    UNARY = "unary"       # alive(X)
    TERNARY = "ternary"   # between(X, Y, Z)
    FUNCTIONAL = "functional"  # age(X) = 25

@dataclass
class Predicate:
    """Represents a logical predicate"""
    name: str
    args: List[str]
    predicate_type: PredicateType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.predicate_type == PredicateType.FUNCTIONAL:
            return f"{self.name}({', '.join(self.args[:-1])}) = {self.args[-1]}"
        return f"{self.name}({', '.join(self.args)})"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Predicate) and 
                self.name == other.name and 
                self.args == other.args)
    
    def __hash__(self) -> int:
        return hash((self.name, tuple(self.args)))

@dataclass
class LogicalExpression:
    """Represents a complex logical expression"""
    operator: LogicalOperator
    operands: List[Union['LogicalExpression', Predicate]]
    confidence: float = 1.0
    
    def __str__(self) -> str:
        if self.operator == LogicalOperator.NOT:
            return f"NOT({self.operands[0]})"
        elif self.operator in [LogicalOperator.AND, LogicalOperator.OR]:
            op_str = f" {self.operator.value} "
            return f"({op_str.join(str(op) for op in self.operands)})"
        elif self.operator == LogicalOperator.IMPLIES:
            return f"({self.operands[0]} → {self.operands[1]})"
        elif self.operator == LogicalOperator.IFF:
            return f"({self.operands[0]} ↔ {self.operands[1]})"
        elif self.operator in [LogicalOperator.EXISTS, LogicalOperator.FORALL]:
            var = self.operands[0] if isinstance(self.operands[0], str) else "X"
            expr = self.operands[1] if len(self.operands) > 1 else self.operands[0]
            symbol = "∃" if self.operator == LogicalOperator.EXISTS else "∀"
            return f"{symbol}{var}.{expr}"
        return str(self.operands)

@dataclass
class Rule:
    """Represents an inference rule"""
    id: str
    premises: List[Union[Predicate, LogicalExpression]]
    conclusion: Union[Predicate, LogicalExpression]
    confidence: float = 1.0
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        premises_str = " ∧ ".join(str(p) for p in self.premises)
        return f"{premises_str} → {self.conclusion}"
    
    def matches_pattern(self, facts: List[Predicate], 
                       variable_bindings: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
        """Check if this rule matches given facts and return variable bindings"""
        bindings = variable_bindings or {}
        
        for premise in self.premises:
            if isinstance(premise, Predicate):
                match_found = False
                for fact in facts:
                    new_bindings = self._unify_predicate(premise, fact, bindings.copy())
                    if new_bindings is not None:
                        bindings.update(new_bindings)
                        match_found = True
                        break
                if not match_found:
                    return None
            elif isinstance(premise, LogicalExpression):
                # Handle complex expressions
                if not self._evaluate_expression(premise, facts, bindings):
                    return None
        
        return bindings

    def _unify_predicate(self, pattern: Predicate, fact: Predicate, 
                        bindings: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Unify a predicate pattern with a fact"""
        if pattern.name != fact.name or len(pattern.args) != len(fact.args):
            return None
        
        new_bindings = bindings.copy()
        
        for pattern_arg, fact_arg in zip(pattern.args, fact.args):
            if self._is_variable(pattern_arg):
                if pattern_arg in new_bindings:
                    if new_bindings[pattern_arg] != fact_arg:
                        return None
                else:
                    new_bindings[pattern_arg] = fact_arg
            elif pattern_arg != fact_arg:
                return None
        
        return new_bindings
    
    def _is_variable(self, term: str) -> bool:
        """Check if a term is a variable (starts with uppercase or ?)"""
        return term.startswith('?') or (term.isalpha() and term[0].isupper())
    
    def _evaluate_expression(self, expr: LogicalExpression, 
                           facts: List[Predicate], 
                           bindings: Dict[str, str]) -> bool:
        """Evaluate a logical expression against facts"""
        # Simplified evaluation - would need more sophisticated implementation
        if expr.operator == LogicalOperator.AND:
            return all(self._evaluate_operand(op, facts, bindings) for op in expr.operands)
        elif expr.operator == LogicalOperator.OR:
            return any(self._evaluate_operand(op, facts, bindings) for op in expr.operands)
        elif expr.operator == LogicalOperator.NOT:
            return not self._evaluate_operand(expr.operands[0], facts, bindings)
        return False
    
    def _evaluate_operand(self, operand: Union[Predicate, LogicalExpression], 
                         facts: List[Predicate], bindings: Dict[str, str]) -> bool:
        """Evaluate a single operand"""
        if isinstance(operand, Predicate):
            # Substitute variables and check if predicate exists in facts
            substituted = self._substitute_variables(operand, bindings)
            # Try to find matching facts with variable unification
            for fact in facts:
                if self._predicates_unify(substituted, fact):
                    return True
            return False
        elif isinstance(operand, LogicalExpression):
            return self._evaluate_expression(operand, facts, bindings)
        return False
    
    def _predicates_unify(self, pred1: Predicate, pred2: Predicate) -> bool:
        """Check if two predicates can unify (ignoring variables)"""
        if pred1.name != pred2.name or len(pred1.args) != len(pred2.args):
            return False
        
        # Check if all non-variable arguments match
        for arg1, arg2 in zip(pred1.args, pred2.args):
            if not self._is_variable(arg1) and not self._is_variable(arg2):
                if arg1 != arg2:
                    return False
        return True
    
    def _substitute_variables(self, predicate: Predicate, bindings: Dict[str, str]) -> Predicate:
        """Substitute variables in a predicate with their bindings"""
        new_args = []
        for arg in predicate.args:
            if self._is_variable(arg) and arg in bindings:
                new_args.append(bindings[arg])
            else:
                new_args.append(arg)
        
        return Predicate(
            name=predicate.name,
            args=new_args,
            predicate_type=predicate.predicate_type,
            confidence=predicate.confidence,
            metadata=predicate.metadata
        )

@dataclass
class InferenceResult:
    """Result of symbolic inference"""
    conclusion: Union[Predicate, LogicalExpression]
    confidence: float
    rule_applied: Rule
    variable_bindings: Dict[str, str]
    inference_chain: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class KnowledgeBase:
    """Knowledge base for storing facts and rules"""
    
    def __init__(self):
        self.facts: Set[Predicate] = set()
        self.rules: List[Rule] = []
        self.predicates: Dict[str, List[Predicate]] = {}  # Index by predicate name
        self.metadata = {
            "created_at": datetime.now(),
            "facts_count": 0,
            "rules_count": 0,
            "last_updated": datetime.now()
        }
    
    def add_fact(self, fact: Predicate):
        """Add a fact to the knowledge base"""
        if fact not in self.facts:
            self.facts.add(fact)
            if fact.name not in self.predicates:
                self.predicates[fact.name] = []
            self.predicates[fact.name].append(fact)
            self.metadata["facts_count"] += 1
            self.metadata["last_updated"] = datetime.now()
            logger.debug(f"Added fact: {fact}")
    
    def add_rule(self, rule: Rule):
        """Add a rule to the knowledge base"""
        self.rules.append(rule)
        # Sort by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        self.metadata["rules_count"] += 1
        self.metadata["last_updated"] = datetime.now()
        logger.debug(f"Added rule: {rule}")
    
    def remove_fact(self, fact: Predicate) -> bool:
        """Remove a fact from the knowledge base"""
        if fact in self.facts:
            self.facts.remove(fact)
            if fact.name in self.predicates:
                self.predicates[fact.name] = [f for f in self.predicates[fact.name] if f != fact]
            self.metadata["facts_count"] -= 1
            self.metadata["last_updated"] = datetime.now()
            return True
        return False
    
    def get_facts_by_predicate(self, predicate_name: str) -> List[Predicate]:
        """Get all facts with a specific predicate name"""
        return self.predicates.get(predicate_name, [])
    
    def query_facts(self, pattern: Predicate) -> List[Tuple[Predicate, Dict[str, str]]]:
        """Query facts that match a pattern and return bindings"""
        results = []
        relevant_facts = self.get_facts_by_predicate(pattern.name)
        
        for fact in relevant_facts:
            bindings = self._unify_predicate(pattern, fact)
            if bindings is not None:
                results.append((fact, bindings))
        
        return results
    
    def _unify_predicate(self, pattern: Predicate, fact: Predicate) -> Optional[Dict[str, str]]:
        """Unify a pattern with a fact"""
        if pattern.name != fact.name or len(pattern.args) != len(fact.args):
            return None
        
        bindings = {}
        for pattern_arg, fact_arg in zip(pattern.args, fact.args):
            if self._is_variable(pattern_arg):
                if pattern_arg in bindings:
                    if bindings[pattern_arg] != fact_arg:
                        return None
                else:
                    bindings[pattern_arg] = fact_arg
            elif pattern_arg != fact_arg:
                return None
        
        return bindings
    
    def _is_variable(self, term: str) -> bool:
        """Check if a term is a variable"""
        return term.startswith('?') or (term.isalpha() and term[0].isupper())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        predicate_counts = {name: len(facts) for name, facts in self.predicates.items()}
        rule_priorities = [rule.priority for rule in self.rules]
        
        return {
            **self.metadata,
            "predicate_types": len(self.predicates),
            "predicate_counts": predicate_counts,
            "average_rule_priority": np.mean(rule_priorities) if rule_priorities else 0,
            "max_rule_priority": max(rule_priorities) if rule_priorities else 0,
            "min_rule_priority": min(rule_priorities) if rule_priorities else 0
        }

class SymbolicReasoner:
    """🔹 Main symbolic reasoning engine with forward and backward chaining"""
    
    def __init__(self, max_inference_steps: int = 100):
        self.knowledge_base = KnowledgeBase()
        self.max_inference_steps = max_inference_steps
        self.inference_history: List[InferenceResult] = []
        
        # Performance metrics
        self.metrics = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "average_inference_time": 0.0,
            "max_inference_steps_used": 0,
            "rules_fired": 0
        }
        
        logger.info("🔹 Symbolic Reasoner initialized")
    
    def add_fact(self, fact_str: str, confidence: float = 1.0) -> Predicate:
        """Add a fact from string representation"""
        fact = self.parse_predicate(fact_str, confidence)
        self.knowledge_base.add_fact(fact)
        return fact
    
    def add_rule(self, rule_str: str, priority: int = 1, confidence: float = 1.0) -> Rule:
        """Add a rule from string representation"""
        rule = self.parse_rule(rule_str, priority, confidence)
        self.knowledge_base.add_rule(rule)
        return rule
    
    def parse_predicate(self, predicate_str: str, confidence: float = 1.0) -> Predicate:
        """Parse a predicate from string format: predicate(arg1, arg2, ...)"""
        # Handle functional predicates: age(john) = 25
        if ' = ' in predicate_str:
            left, right = predicate_str.split(' = ', 1)
            match = re.match(r'(\w+)\((.*?)\)', left.strip())
            if match:
                name, args_str = match.groups()
                args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                args.append(right.strip())
                return Predicate(name, args, PredicateType.FUNCTIONAL, confidence)
        
        # Regular predicates: is_a(john, human)
        match = re.match(r'(\w+)\((.*?)\)', predicate_str.strip())
        if not match:
            raise ValueError(f"Invalid predicate format: {predicate_str}")
        
        name, args_str = match.groups()
        args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
        
        # Determine predicate type based on argument count
        if len(args) == 1:
            pred_type = PredicateType.UNARY
        elif len(args) == 2:
            pred_type = PredicateType.BINARY
        elif len(args) == 3:
            pred_type = PredicateType.TERNARY
        else:
            pred_type = PredicateType.BINARY  # Default
        
        return Predicate(name, args, pred_type, confidence)
    
    def parse_rule(self, rule_str: str, priority: int = 1, confidence: float = 1.0) -> Rule:
        """Parse a rule from string format: IF ... THEN ..."""
        rule_str = rule_str.strip()
        
        # Handle different rule formats
        if ' THEN ' in rule_str:
            premises_str, conclusion_str = rule_str.split(' THEN ', 1)
            premises_str = premises_str.replace('IF ', '').strip()
        elif ' → ' in rule_str:
            premises_str, conclusion_str = rule_str.split(' → ', 1)
        elif ' implies ' in rule_str:
            premises_str, conclusion_str = rule_str.split(' implies ', 1)
        else:
            raise ValueError(f"Invalid rule format: {rule_str}")
        
        # Parse premises (connected by AND/OR)
        premises = []
        if ' AND ' in premises_str:
            premise_strs = premises_str.split(' AND ')
        elif ' OR ' in premises_str:
            # For now, convert OR to multiple rules (would need more sophisticated handling)
            premise_strs = premises_str.split(' OR ')
        else:
            premise_strs = [premises_str]
        
        for premise_str in premise_strs:
            premise_str = premise_str.strip()
            if premise_str:
                # Handle negated premises
                if premise_str.startswith('NOT '):
                    # Create a logical expression for negation
                    positive_premise_str = premise_str[4:].strip()  # Remove "NOT "
                    positive_predicate = self.parse_predicate(positive_premise_str)
                    negated_expr = LogicalExpression(
                        operator=LogicalOperator.NOT,
                        operands=[positive_predicate],
                        confidence=1.0
                    )
                    premises.append(negated_expr)
                else:
                    premises.append(self.parse_predicate(premise_str))
        
        # Parse conclusion
        conclusion = self.parse_predicate(conclusion_str.strip())
        
        rule_id = str(uuid.uuid4())
        return Rule(rule_id, premises, conclusion, confidence, priority)
    
    async def forward_chain(self, max_steps: Optional[int] = None) -> List[InferenceResult]:
        """Perform forward chaining inference"""
        max_steps = max_steps or self.max_inference_steps
        start_time = datetime.now()
        
        new_facts = []
        step_count = 0
        
        logger.info("🔍 Starting forward chaining inference")
        
        for step in range(max_steps):
            step_count += 1
            facts_added_this_step = False
            
            for rule in self.knowledge_base.rules:
                # Check if rule can be applied
                bindings = rule.matches_pattern(list(self.knowledge_base.facts))
                
                if bindings is not None:
                    # Apply rule to generate new conclusion
                    conclusion = self._apply_bindings(rule.conclusion, bindings)
                    
                    # Check if conclusion is new
                    if conclusion not in self.knowledge_base.facts:
                        # Calculate confidence
                        confidence = rule.confidence
                        for premise in rule.premises:
                            if isinstance(premise, Predicate):
                                # Find matching fact confidence
                                for fact in self.knowledge_base.facts:
                                    if self._predicates_match(premise, fact, bindings):
                                        confidence *= fact.confidence
                                        break
                        
                        # Add new fact
                        conclusion.confidence = min(confidence, 1.0)
                        self.knowledge_base.add_fact(conclusion)
                        
                        # Record inference
                        inference_result = InferenceResult(
                            conclusion=conclusion,
                            confidence=conclusion.confidence,
                            rule_applied=rule,
                            variable_bindings=bindings,
                            inference_chain=[f"Step {step + 1}: Applied rule {rule.id}"],
                            metadata={
                                "step": step + 1,
                                "forward_chaining": True,
                                "inference_time": (datetime.now() - start_time).total_seconds()
                            }
                        )
                        
                        new_facts.append(inference_result)
                        self.inference_history.append(inference_result)
                        facts_added_this_step = True
                        self.metrics["rules_fired"] += 1
                        
                        logger.debug(f"Forward chain step {step + 1}: {conclusion}")
            
            # Stop if no new facts were derived
            if not facts_added_this_step:
                logger.info(f"🔍 Forward chaining completed after {step + 1} steps")
                break
        
        # Update metrics
        inference_time = (datetime.now() - start_time).total_seconds()
        self.metrics["total_inferences"] += 1
        self.metrics["successful_inferences"] += len(new_facts)
        self.metrics["max_inference_steps_used"] = max(
            self.metrics["max_inference_steps_used"], step_count
        )
        self._update_average_inference_time(inference_time)
        
        logger.info(f"🔍 Forward chaining derived {len(new_facts)} new facts")
        return new_facts
    
    async def backward_chain(self, goal: Union[str, Predicate], 
                           max_depth: int = 10) -> List[InferenceResult]:
        """Perform backward chaining to prove a goal"""
        if isinstance(goal, str):
            goal = self.parse_predicate(goal)
        
        start_time = datetime.now()
        logger.info(f"🎯 Starting backward chaining for goal: {goal}")
        
        proof_chain = []
        visited = set()
        
        def prove_goal(current_goal: Predicate, depth: int = 0) -> bool:
            if depth > max_depth:
                return False
            
            goal_key = str(current_goal)
            if goal_key in visited:
                return False
            visited.add(goal_key)
            
            # Check if goal is already a known fact
            for fact in self.knowledge_base.facts:
                if self._predicates_match(current_goal, fact):
                    proof_chain.append(f"Goal {current_goal} proven by fact")
                    return True
            
            # Try to prove goal using rules
            for rule in self.knowledge_base.rules:
                # Check if rule conclusion matches goal
                bindings = self._unify_predicate(rule.conclusion, current_goal)
                if bindings is not None:
                    # Try to prove all premises
                    all_premises_proven = True
                    subgoals = []
                    
                    for premise in rule.premises:
                        if isinstance(premise, Predicate):
                            subgoal = self._apply_bindings(premise, bindings)
                            subgoals.append(subgoal)
                            
                            if not prove_goal(subgoal, depth + 1):
                                all_premises_proven = False
                                break
                    
                    if all_premises_proven:
                        proof_chain.append(f"Goal {current_goal} proven by rule {rule.id}")
                        
                        # Record successful inference
                        inference_result = InferenceResult(
                            conclusion=current_goal,
                            confidence=rule.confidence,
                            rule_applied=rule,
                            variable_bindings=bindings,
                            inference_chain=proof_chain.copy(),
                            metadata={
                                "backward_chaining": True,
                                "depth": depth,
                                "subgoals": [str(sg) for sg in subgoals],
                                "inference_time": (datetime.now() - start_time).total_seconds()
                            }
                        )
                        
                        self.inference_history.append(inference_result)
                        return True
            
            return False
        
        # Attempt to prove the goal
        success = prove_goal(goal)
        
        # Update metrics
        inference_time = (datetime.now() - start_time).total_seconds()
        self.metrics["total_inferences"] += 1
        if success:
            self.metrics["successful_inferences"] += 1
        else:
            self.metrics["failed_inferences"] += 1
        self._update_average_inference_time(inference_time)
        
        if success:
            logger.info(f"🎯 Backward chaining successfully proved: {goal}")
            return [self.inference_history[-1]]  # Return the last successful inference
        else:
            logger.info(f"🎯 Backward chaining failed to prove: {goal}")
            return []
    
    async def query(self, query_str: str) -> List[Dict[str, Any]]:
        """Query the knowledge base with a pattern"""
        pattern = self.parse_predicate(query_str)
        results = []
        
        # Direct fact matching
        for fact_result in self.knowledge_base.query_facts(pattern):
            fact, bindings = fact_result
            results.append({
                "fact": str(fact),
                "bindings": bindings,
                "confidence": fact.confidence,
                "type": "direct_fact"
            })
        
        # Inference-based results (forward chaining with specific goal)
        # This would involve more sophisticated goal-directed reasoning
        
        logger.info(f"🔍 Query '{query_str}' returned {len(results)} results")
        return results
    
    def _apply_bindings(self, predicate: Predicate, bindings: Dict[str, str]) -> Predicate:
        """Apply variable bindings to a predicate"""
        new_args = []
        for arg in predicate.args:
            if self._is_variable(arg) and arg in bindings:
                new_args.append(bindings[arg])
            else:
                new_args.append(arg)
        
        return Predicate(
            name=predicate.name,
            args=new_args,
            predicate_type=predicate.predicate_type,
            confidence=predicate.confidence,
            metadata=predicate.metadata
        )
    
    def _predicates_match(self, p1: Predicate, p2: Predicate, 
                         bindings: Optional[Dict[str, str]] = None) -> bool:
        """Check if two predicates match (considering variable bindings)"""
        if p1.name != p2.name or len(p1.args) != len(p2.args):
            return False
        
        if bindings:
            p1_substituted = self._apply_bindings(p1, bindings)
            return p1_substituted == p2
        
        return p1 == p2
    
    def _unify_predicate(self, pattern: Predicate, fact: Predicate) -> Optional[Dict[str, str]]:
        """Unify a predicate pattern with a fact"""
        if pattern.name != fact.name or len(pattern.args) != len(fact.args):
            return None
        
        bindings = {}
        for pattern_arg, fact_arg in zip(pattern.args, fact.args):
            if self._is_variable(pattern_arg):
                if pattern_arg in bindings:
                    if bindings[pattern_arg] != fact_arg:
                        return None
                else:
                    bindings[pattern_arg] = fact_arg
            elif pattern_arg != fact_arg:
                return None
        
        return bindings
    
    def _is_variable(self, term: str) -> bool:
        """Check if a term is a variable"""
        return term.startswith('?') or (term.isalpha() and term[0].isupper())
    
    def _update_average_inference_time(self, inference_time: float):
        """Update average inference time metric"""
        current_avg = self.metrics["average_inference_time"]
        total_inferences = self.metrics["total_inferences"]
        
        if total_inferences == 1:
            self.metrics["average_inference_time"] = inference_time
        else:
            self.metrics["average_inference_time"] = (
                (current_avg * (total_inferences - 1) + inference_time) / total_inferences
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reasoner statistics"""
        kb_stats = self.knowledge_base.get_statistics()
        
        return {
            "reasoner_metrics": self.metrics,
            "knowledge_base": kb_stats,
            "inference_history_size": len(self.inference_history),
            "recent_inferences": [
                {
                    "conclusion": str(inf.conclusion),
                    "confidence": inf.confidence,
                    "rule_id": inf.rule_applied.id
                }
                for inf in self.inference_history[-5:]  # Last 5 inferences
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_inference_history(self):
        """Clear inference history (useful for memory management)"""
        self.inference_history.clear()
        logger.info("🧹 Cleared inference history")
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """Export knowledge base in a serializable format"""
        return {
            "facts": [
                {
                    "predicate": str(fact),
                    "confidence": fact.confidence,
                    "metadata": fact.metadata
                }
                for fact in self.knowledge_base.facts
            ],
            "rules": [
                {
                    "id": rule.id,
                    "rule": str(rule),
                    "confidence": rule.confidence,
                    "priority": rule.priority,
                    "metadata": rule.metadata
                }
                for rule in self.knowledge_base.rules
            ],
            "statistics": self.get_statistics(),
            "exported_at": datetime.now().isoformat()
        }