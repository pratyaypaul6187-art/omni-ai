"""
🔹 NEUROSYMBOLIC AI FRAMEWORK
Advanced hybrid reasoning combining symbolic logic with neural networks
"""

from .symbolic_reasoner import (
    SymbolicReasoner,
    Predicate,
    LogicalExpression,
    Rule,
    InferenceResult,
    KnowledgeBase,
    LogicalOperator,
    PredicateType
)

from .knowledge_graph import (
    KnowledgeGraph,
    Entity,
    Relation,
    EntityType,
    RelationType
)

from .neural_symbolic_bridge import (
    NeuralSymbolicBridge,
    NeuralOutput,
    SymbolicOutput,
    HybridResult,
    ReasoningMode,
    ConfidenceAggregation
)

from .nl_interface import (
    NaturalLanguageInterface,
    QueryType,
    ParsedQuery,
    NLResponse,
    create_nl_interface
)

from .knowledge_manager import (
    KnowledgeBaseManager,
    KnowledgeBaseContent,
    KnowledgeBaseMetadata,
    KnowledgeBaseTemplate,
    DomainType,
    KnowledgeFormat,
    create_knowledge_manager,
    get_domain_template
)

from .advanced_reasoner import (
    AdvancedReasoner,
    TemporalOperator,
    UncertaintyType,
    ReasoningMode,
    TemporalFact,
    ProbabilisticFact,
    CausalRelation,
    Explanation,
    ReasoningResult,
    create_advanced_reasoner
)

__all__ = [
    # Symbolic Reasoner
    'SymbolicReasoner',
    'Predicate',
    'LogicalExpression', 
    'Rule',
    'InferenceResult',
    'KnowledgeBase',
    'LogicalOperator',
    'PredicateType',
    
    # Knowledge Graph
    'KnowledgeGraph',
    'Entity',
    'Relation',
    'EntityType',
    'RelationType',
    
    # Neural-Symbolic Bridge
    'NeuralSymbolicBridge',
    'NeuralOutput',
    'SymbolicOutput',
    'HybridResult',
    'ReasoningMode',
    'ConfidenceAggregation',
    
    # Natural Language Interface
    'NaturalLanguageInterface',
    'QueryType',
    'ParsedQuery',
    'NLResponse',
    'create_nl_interface',
    
    # Knowledge Base Management
    'KnowledgeBaseManager',
    'KnowledgeBaseContent',
    'KnowledgeBaseMetadata',
    'KnowledgeBaseTemplate',
    'DomainType',
    'KnowledgeFormat',
    'create_knowledge_manager',
    'get_domain_template',
    
    # Advanced Reasoning
    'AdvancedReasoner',
    'TemporalOperator',
    'UncertaintyType',
    'ReasoningMode',
    'TemporalFact',
    'ProbabilisticFact',
    'CausalRelation',
    'Explanation',
    'ReasoningResult',
    'create_advanced_reasoner'
]

__version__ = "1.0.0"