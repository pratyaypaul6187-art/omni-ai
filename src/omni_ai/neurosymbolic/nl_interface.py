"""
🔸 NATURAL LANGUAGE INTERFACE FOR NEUROSYMBOLIC AI
Converts natural language queries to symbolic logic and provides human-readable responses
"""

import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from structlog import get_logger

from .symbolic_reasoner import SymbolicReasoner, Predicate, Rule
from .knowledge_graph import KnowledgeGraph
from .neural_symbolic_bridge import NeuralSymbolicBridge, ReasoningMode

logger = get_logger()

class QueryType(Enum):
    FACT_QUERY = "fact_query"          # "Is Socrates mortal?"
    VARIABLE_QUERY = "variable_query"   # "Who is mortal?"
    RULE_ADDITION = "rule_addition"     # "All humans are mortal"
    FACT_ADDITION = "fact_addition"     # "Socrates is a human"
    EXPLANATION = "explanation"         # "Why is Socrates mortal?"
    COMPLEX_REASONING = "complex_reasoning"  # "What can we infer about..."

@dataclass
class ParsedQuery:
    """Represents a parsed natural language query"""
    query_type: QueryType
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    variables: List[str] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 1.0
    symbolic_representation: Optional[str] = None

@dataclass
class NLResponse:
    """Natural language response with reasoning details"""
    answer: str
    confidence: float
    reasoning_chain: List[str]
    symbolic_results: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class NaturalLanguageInterface:
    """🔸 Natural Language Interface for Neurosymbolic AI"""
    
    def __init__(self, neurosymbolic_bridge: NeuralSymbolicBridge):
        self.bridge = neurosymbolic_bridge
        self.reasoner = neurosymbolic_bridge.symbolic_reasoner
        self.knowledge_graph = neurosymbolic_bridge.knowledge_graph
        
        # Language patterns for parsing
        self.patterns = self._initialize_patterns()
        
        # Response templates
        self.response_templates = self._initialize_templates()
        
        logger.info("🔸 Natural Language Interface initialized")
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for natural language parsing"""
        return {
            "is_questions": [
                r"is (\w+) (?:a |an )?(\w+)\??",
                r"does (\w+) have (\w+)\??",
                r"can (\w+) (\w+)\??"
            ],
            "who_questions": [
                r"who (?:is|are) (?:a |an )?(\w+)\??",
                r"who (?:has|have) (\w+)\??",
                r"who can (\w+)\??"
            ],
            "what_questions": [
                r"what (?:is|are) (\w+)\??",
                r"what (?:can|could) we (?:infer|conclude) about (\w+)\??",
                r"what (?:properties|attributes) (?:does|do) (\w+) have\??"
            ],
            "all_statements": [
                r"all (\w+) (?:are|is) (\w+)",
                r"every (\w+) (?:has|have) (\w+)",
                r"if (\w+) (?:is|are) (?:a |an )?(\w+) then (?:they|it) (?:is|are) (\w+)"
            ],
            "fact_statements": [
                r"(\w+) (?:is|are) (?:a |an )?(\w+)",
                r"(\w+) (?:has|have) (\w+)",
                r"(\w+) can (\w+)"
            ],
            "why_questions": [
                r"why (?:is|are) (\w+) (?:a |an )?(\w+)\??",
                r"explain why (\w+) (?:is|are) (\w+)\??",
                r"how do we know (?:that )?(\w+) (?:is|are) (\w+)\??"
            ]
        }
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates"""
        return {
            "positive_fact": [
                "Yes, {subject} is {predicate}.",
                "That's correct, {subject} is {predicate}.",
                "Indeed, {subject} is {predicate}."
            ],
            "negative_fact": [
                "No, {subject} is not {predicate}.",
                "That's not established in the knowledge base.",
                "I cannot confirm that {subject} is {predicate}."
            ],
            "variable_results": [
                "I found {count} results: {results}",
                "Here are the matches: {results}",
                "The following satisfy your query: {results}"
            ],
            "no_results": [
                "I couldn't find any results for that query.",
                "No matches were found in the knowledge base.",
                "The query didn't return any results."
            ],
            "explanation": [
                "Based on the rule '{rule}' and the facts {facts}, we can conclude that {conclusion}.",
                "This follows from {rule} given that {facts}.",
                "The reasoning is: {facts} + {rule} → {conclusion}"
            ]
        }
    
    async def process_query(self, text: str, reasoning_mode: ReasoningMode = ReasoningMode.ADAPTIVE) -> NLResponse:
        """Process a natural language query and return a response"""
        logger.info(f"🔸 Processing query: '{text}'")
        
        try:
            # Parse the natural language query
            parsed = self.parse_query(text)
            
            # Process based on query type
            if parsed.query_type == QueryType.FACT_QUERY:
                return await self._handle_fact_query(parsed, reasoning_mode)
            elif parsed.query_type == QueryType.VARIABLE_QUERY:
                return await self._handle_variable_query(parsed, reasoning_mode)
            elif parsed.query_type == QueryType.RULE_ADDITION:
                return await self._handle_rule_addition(parsed)
            elif parsed.query_type == QueryType.FACT_ADDITION:
                return await self._handle_fact_addition(parsed)
            elif parsed.query_type == QueryType.EXPLANATION:
                return await self._handle_explanation_query(parsed)
            elif parsed.query_type == QueryType.COMPLEX_REASONING:
                return await self._handle_complex_reasoning(parsed, reasoning_mode)
            else:
                return self._create_error_response(f"Could not understand query type: {text}")
                
        except Exception as e:
            logger.error(f"Error processing query '{text}': {e}")
            return self._create_error_response(f"Error processing query: {str(e)}")
    
    def parse_query(self, text: str) -> ParsedQuery:
        """Parse natural language query into structured format"""
        text = text.lower().strip()
        
        # Check for different query types
        for pattern_group, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return self._extract_query_info(pattern_group, match, text)
        
        # Fallback to complex reasoning if no pattern matches
        return ParsedQuery(
            query_type=QueryType.COMPLEX_REASONING,
            raw_text=text,
            confidence=0.5
        )
    
    def _extract_query_info(self, pattern_group: str, match: re.Match, text: str) -> ParsedQuery:
        """Extract structured information from regex match"""
        groups = match.groups()
        
        if pattern_group == "is_questions":
            return ParsedQuery(
                query_type=QueryType.FACT_QUERY,
                subject=groups[0],
                predicate=groups[1] if len(groups) > 1 else groups[0],
                raw_text=text,
                symbolic_representation=f"{groups[1] if len(groups) > 1 else 'property'}({groups[0]})",
                confidence=0.9
            )
        elif pattern_group == "who_questions":
            return ParsedQuery(
                query_type=QueryType.VARIABLE_QUERY,
                predicate=groups[0],
                variables=["X"],
                raw_text=text,
                symbolic_representation=f"{groups[0]}(X)",
                confidence=0.9
            )
        elif pattern_group == "what_questions":
            return ParsedQuery(
                query_type=QueryType.COMPLEX_REASONING,
                subject=groups[0],
                raw_text=text,
                confidence=0.8
            )
        elif pattern_group == "all_statements":
            if len(groups) >= 3:  # if-then rule
                return ParsedQuery(
                    query_type=QueryType.RULE_ADDITION,
                    subject=groups[0],
                    predicate=groups[1],
                    object=groups[2],
                    raw_text=text,
                    symbolic_representation=f"IF is_a(X, {groups[0]}) THEN {groups[2]}(X)",
                    confidence=0.9
                )
            else:  # simple universal statement
                return ParsedQuery(
                    query_type=QueryType.RULE_ADDITION,
                    subject=groups[0],
                    predicate=groups[1],
                    raw_text=text,
                    symbolic_representation=f"IF is_a(X, {groups[0]}) THEN {groups[1]}(X)",
                    confidence=0.9
                )
        elif pattern_group == "fact_statements":
            return ParsedQuery(
                query_type=QueryType.FACT_ADDITION,
                subject=groups[0],
                predicate=groups[1],
                raw_text=text,
                symbolic_representation=f"is_a({groups[0]}, {groups[1]})",
                confidence=0.9
            )
        elif pattern_group == "why_questions":
            return ParsedQuery(
                query_type=QueryType.EXPLANATION,
                subject=groups[0],
                predicate=groups[1],
                raw_text=text,
                confidence=0.9
            )
        
        return ParsedQuery(query_type=QueryType.COMPLEX_REASONING, raw_text=text, confidence=0.5)
    
    async def _handle_fact_query(self, parsed: ParsedQuery, reasoning_mode: ReasoningMode) -> NLResponse:
        """Handle yes/no fact queries"""
        if parsed.symbolic_representation:
            # Try to prove the fact using symbolic reasoning
            goal_predicate = parsed.symbolic_representation
            
            # First check direct facts
            pattern = self.reasoner.parse_predicate(goal_predicate)
            query_results = self.reasoner.knowledge_base.query_facts(pattern)
            
            if query_results:
                return NLResponse(
                    answer=f"Yes, {parsed.subject} is {parsed.predicate}.",
                    confidence=1.0,
                    reasoning_chain=["Found direct fact in knowledge base"],
                    symbolic_results=query_results
                )
            
            # Try backward chaining
            try:
                inference_result = await self.reasoner.backward_chain(goal_predicate)
                if inference_result:
                    return NLResponse(
                        answer=f"Yes, {parsed.subject} is {parsed.predicate}.",
                        confidence=inference_result.confidence,
                        reasoning_chain=inference_result.inference_chain,
                        symbolic_results=[inference_result]
                    )
            except:
                pass
            
            # Use hybrid reasoning as fallback
            hybrid_result = await self.bridge.reason(parsed.raw_text, query=goal_predicate, reasoning_mode=reasoning_mode)
            
            if hybrid_result and hybrid_result.confidence > 0.5:
                return NLResponse(
                    answer=f"Based on available evidence, {parsed.subject} is likely {parsed.predicate}.",
                    confidence=hybrid_result.confidence,
                    reasoning_chain=[f"Hybrid reasoning result: {hybrid_result.final_conclusion}"],
                    metadata={"hybrid_reasoning": True}
                )
            
            return NLResponse(
                answer=f"I cannot confirm that {parsed.subject} is {parsed.predicate}.",
                confidence=0.0,
                reasoning_chain=["No supporting evidence found"]
            )
    
    async def _handle_variable_query(self, parsed: ParsedQuery, reasoning_mode: ReasoningMode) -> NLResponse:
        """Handle queries with variables (who/what questions)"""
        if parsed.symbolic_representation:
            pattern = self.reasoner.parse_predicate(parsed.symbolic_representation)
            query_results = self.reasoner.knowledge_base.query_facts(pattern)
            
            if query_results:
                results = []
                for result, bindings in query_results:
                    if 'X' in bindings:
                        results.append(bindings['X'])
                
                if results:
                    return NLResponse(
                        answer=f"I found {len(results)} results: {', '.join(results)}",
                        confidence=0.9,
                        reasoning_chain=[f"Direct query returned {len(results)} matches"],
                        symbolic_results=query_results
                    )
        
        return NLResponse(
            answer="I couldn't find any results for that query.",
            confidence=0.0,
            reasoning_chain=["No matches found in knowledge base"]
        )
    
    async def _handle_rule_addition(self, parsed: ParsedQuery) -> NLResponse:
        """Handle rule addition statements"""
        if parsed.symbolic_representation:
            try:
                rule = self.reasoner.add_rule(parsed.symbolic_representation, confidence=parsed.confidence)
                return NLResponse(
                    answer="I've added that rule to my knowledge base.",
                    confidence=1.0,
                    reasoning_chain=[f"Added rule: {rule}"],
                    symbolic_results=[rule]
                )
            except Exception as e:
                return self._create_error_response(f"Could not add rule: {e}")
        
        return self._create_error_response("Could not parse the rule statement.")
    
    async def _handle_fact_addition(self, parsed: ParsedQuery) -> NLResponse:
        """Handle fact addition statements"""
        if parsed.symbolic_representation:
            try:
                fact = self.reasoner.add_fact(parsed.symbolic_representation, confidence=parsed.confidence)
                return NLResponse(
                    answer="I've added that fact to my knowledge base.",
                    confidence=1.0,
                    reasoning_chain=[f"Added fact: {fact}"],
                    symbolic_results=[fact]
                )
            except Exception as e:
                return self._create_error_response(f"Could not add fact: {e}")
        
        return self._create_error_response("Could not parse the fact statement.")
    
    async def _handle_explanation_query(self, parsed: ParsedQuery) -> NLResponse:
        """Handle explanation queries (why questions)"""
        goal = f"{parsed.predicate}({parsed.subject})"
        
        try:
            # Try to find reasoning chain through backward chaining
            inference_result = await self.reasoner.backward_chain(goal)
            
            if inference_result:
                explanation = f"Because {parsed.subject} satisfies the rule '{inference_result.rule_applied}', " \
                            f"we can conclude that {parsed.subject} is {parsed.predicate}."
                
                return NLResponse(
                    answer=explanation,
                    confidence=inference_result.confidence,
                    reasoning_chain=inference_result.inference_chain,
                    symbolic_results=[inference_result]
                )
        except:
            pass
        
        return NLResponse(
            answer=f"I don't have enough information to explain why {parsed.subject} is {parsed.predicate}.",
            confidence=0.0,
            reasoning_chain=["No reasoning chain found"]
        )
    
    async def _handle_complex_reasoning(self, parsed: ParsedQuery, reasoning_mode: ReasoningMode) -> NLResponse:
        """Handle complex reasoning queries"""
        hybrid_result = await self.bridge.reason(parsed.raw_text, reasoning_mode=reasoning_mode)
        
        if hybrid_result:
            return NLResponse(
                answer=f"Based on my reasoning: {hybrid_result.final_conclusion}",
                confidence=hybrid_result.confidence,
                reasoning_chain=[f"Hybrid reasoning with {reasoning_mode.value} mode"],
                metadata={
                    "reasoning_mode": reasoning_mode.value,
                    "processing_time": hybrid_result.processing_time
                }
            )
        
        return NLResponse(
            answer="I couldn't process that complex query.",
            confidence=0.0,
            reasoning_chain=["Complex reasoning failed"]
        )
    
    def _create_error_response(self, message: str) -> NLResponse:
        """Create an error response"""
        return NLResponse(
            answer=f"Error: {message}",
            confidence=0.0,
            reasoning_chain=[f"Error: {message}"]
        )
    
    def add_knowledge_from_text(self, text: str) -> Dict[str, Any]:
        """Extract and add knowledge from free text"""
        logger.info(f"🔸 Extracting knowledge from text: '{text[:100]}...'")
        
        sentences = text.split('.')
        added_facts = []
        added_rules = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            try:
                parsed = self.parse_query(sentence)
                
                if parsed.query_type == QueryType.FACT_ADDITION:
                    fact = self.reasoner.add_fact(parsed.symbolic_representation)
                    added_facts.append(str(fact))
                elif parsed.query_type == QueryType.RULE_ADDITION:
                    rule = self.reasoner.add_rule(parsed.symbolic_representation)
                    added_rules.append(str(rule))
                    
            except Exception as e:
                logger.warning(f"Could not parse sentence: '{sentence}' - {e}")
        
        return {
            "facts_added": len(added_facts),
            "rules_added": len(added_rules),
            "facts": added_facts,
            "rules": added_rules
        }

async def create_nl_interface(knowledge_base_path: Optional[str] = None) -> NaturalLanguageInterface:
    """Factory function to create a natural language interface with optional knowledge base"""
    reasoner = SymbolicReasoner()
    knowledge_graph = KnowledgeGraph()
    bridge = NeuralSymbolicBridge(reasoner, knowledge_graph)
    
    interface = NaturalLanguageInterface(bridge)
    
    if knowledge_base_path:
        # Load knowledge base if provided
        logger.info(f"🔸 Loading knowledge base from {knowledge_base_path}")
        # Implementation would load from file
    
    return interface