"""
🔹 NEUROSYMBOLIC AI FRAMEWORK TESTS
Comprehensive testing and demonstration of hybrid neural-symbolic reasoning
"""

import asyncio
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omni_ai.neurosymbolic import (
    SymbolicReasoner,
    KnowledgeGraph,
    NeuralSymbolicBridge,
    EntityType,
    RelationType,
    ReasoningMode,
    Predicate,
    PredicateType
)

async def test_symbolic_reasoner():
    """Test the symbolic reasoning engine"""
    print("🔹 Testing Symbolic Reasoner")
    print("=" * 40)
    
    reasoner = SymbolicReasoner()
    
    # Add facts
    print("📝 Adding facts...")
    reasoner.add_fact("is_a(socrates, human)")
    reasoner.add_fact("is_a(plato, human)")
    reasoner.add_fact("is_a(aristotle, human)")
    reasoner.add_fact("mortal(socrates)")
    reasoner.add_fact("philosopher(socrates)")
    reasoner.add_fact("philosopher(plato)")
    
    # Add rules
    print("📝 Adding rules...")
    reasoner.add_rule("IF is_a(X, human) THEN mortal(X)", priority=5, confidence=0.9)
    reasoner.add_rule("IF mortal(X) AND philosopher(X) THEN wise(X)", priority=3, confidence=0.8)
    reasoner.add_rule("IF is_a(X, human) AND philosopher(X) THEN teacher(X)", priority=2, confidence=0.7)
    
    # Test forward chaining
    print("\n🔍 Testing forward chaining...")
    new_facts = await reasoner.forward_chain(max_steps=5)
    
    print(f"✅ Derived {len(new_facts)} new facts:")
    for result in new_facts:
        print(f"  - {result.conclusion} (confidence: {result.confidence:.3f})")
    
    # Test backward chaining
    print("\n🎯 Testing backward chaining...")
    goal_results = await reasoner.backward_chain("wise(plato)")
    
    if goal_results:
        print("✅ Successfully proved goal 'wise(plato)'")
        for result in goal_results:
            print(f"  - Used rule: {result.rule_applied}")
    else:
        print("❌ Failed to prove goal 'wise(plato)'")
    
    # Test queries
    print("\n🔍 Testing queries...")
    query_results = await reasoner.query("mortal(X)")
    
    print(f"✅ Query 'mortal(X)' returned {len(query_results)} results:")
    for result in query_results:
        print(f"  - {result['fact']} | bindings: {result['bindings']}")
    
    # Get statistics
    stats = reasoner.get_statistics()
    print(f"\n📊 Reasoner Statistics:")
    print(f"  - Total facts: {stats['knowledge_base']['facts_count']}")
    print(f"  - Total rules: {stats['knowledge_base']['rules_count']}")
    print(f"  - Rules fired: {stats['reasoner_metrics']['rules_fired']}")
    print(f"  - Successful inferences: {stats['reasoner_metrics']['successful_inferences']}")
    
    return reasoner

async def test_knowledge_graph():
    """Test the knowledge graph system"""
    print("\n🔹 Testing Knowledge Graph")
    print("=" * 40)
    
    kg = KnowledgeGraph(enable_embeddings=True)
    
    # Add entities
    print("📝 Adding entities...")
    kg.add_entity("Socrates", EntityType.PERSON, {"birth_year": -470, "death_year": -399})
    kg.add_entity("Plato", EntityType.PERSON, {"birth_year": -428, "death_year": -348})
    kg.add_entity("Aristotle", EntityType.PERSON, {"birth_year": -384, "death_year": -322})
    kg.add_entity("Philosophy", EntityType.CONCEPT, {"definition": "love of wisdom"})
    kg.add_entity("Athens", EntityType.LOCATION, {"country": "Greece"})
    
    # Add relations
    print("📝 Adding relations...")
    kg.add_relation("Socrates", "Philosophy", RelationType.ASSOCIATED_WITH)
    kg.add_relation("Plato", "Socrates", RelationType.FRIEND_OF, bidirectional=True)
    kg.add_relation("Aristotle", "Plato", RelationType.ASSOCIATED_WITH)  # Student-teacher
    kg.add_relation("Socrates", "Athens", RelationType.LOCATED_IN)
    kg.add_relation("Plato", "Athens", RelationType.LOCATED_IN)
    
    # Test entity retrieval
    print("\n🔍 Testing entity retrieval...")
    socrates = kg.get_entity_by_name("Socrates")
    if socrates:
        print(f"✅ Found entity: {socrates.name} ({socrates.entity_type.value})")
        print(f"  - Properties: {socrates.properties}")
    
    # Test neighbors
    print("\n🔍 Testing neighbor search...")
    neighbors = kg.get_neighbors("Plato")
    print(f"✅ Plato has {len(neighbors)} neighbors:")
    for neighbor_entity, relation in neighbors:
        print(f"  - {neighbor_entity.name} via {relation.relation_type.value}")
    
    # Test shortest path
    print("\n🔍 Testing shortest path...")
    path = kg.shortest_path("Socrates", "Aristotle")
    if path:
        print("✅ Found path from Socrates to Aristotle:")
        for entity, relation in path:
            if relation:
                print(f"  - {entity.name} (via {relation.relation_type.value})")
            else:
                print(f"  - {entity.name} (start)")
    else:
        print("❌ No path found from Socrates to Aristotle")
    
    # Test pattern queries
    print("\n🔍 Testing pattern queries...")
    pattern_results = kg.query_by_pattern("Plato --friend_of-> ?")
    print(f"✅ Pattern 'Plato --friend_of-> ?' returned {len(pattern_results)} results:")
    for result in pattern_results:
        print(f"  - {result['source']} {result['relation']} {result['target']}")
    
    # Test subgraph extraction
    print("\n🔍 Testing subgraph extraction...")
    subgraph = kg.get_subgraph("Plato", depth=2)
    print(f"✅ Extracted subgraph around Plato:")
    print(f"  - Entities: {subgraph['entity_count']}")
    print(f"  - Relations: {subgraph['relation_count']}")
    
    # Get statistics
    stats = kg.get_statistics()
    print(f"\n📊 Knowledge Graph Statistics:")
    print(f"  - Total entities: {stats['total_entities']}")
    print(f"  - Total relations: {stats['total_relations']}")
    print(f"  - Graph density: {stats['graph_density']:.4f}")
    print(f"  - Connected components: {stats['graph_components']}")
    
    return kg

async def test_neural_symbolic_bridge():
    """Test the neural-symbolic bridge"""
    print("\n🔹 Testing Neural-Symbolic Bridge")
    print("=" * 40)
    
    # Initialize components
    reasoner = SymbolicReasoner()
    kg = KnowledgeGraph()
    
    # Add some knowledge
    reasoner.add_fact("is_a(socrates, human)")
    reasoner.add_fact("philosopher(socrates)")
    reasoner.add_rule("IF is_a(X, human) AND philosopher(X) THEN wise(X)")
    
    kg.add_entity("Socrates", EntityType.PERSON)
    kg.add_entity("Wisdom", EntityType.CONCEPT)
    kg.add_relation("Socrates", "Wisdom", RelationType.ASSOCIATED_WITH)
    
    # Create bridge
    bridge = NeuralSymbolicBridge(
        symbolic_reasoner=reasoner,
        knowledge_graph=kg,
        default_reasoning_mode=ReasoningMode.ADAPTIVE
    )
    
    # Add concept grounding
    print("📝 Adding concept groundings...")
    wisdom_embedding = np.random.rand(768).astype(np.float32)
    human_embedding = np.random.rand(768).astype(np.float32)
    
    bridge.add_concept_grounding("wisdom", wisdom_embedding)
    bridge.add_concept_grounding("human", human_embedding)
    
    # Test different reasoning modes
    reasoning_modes = [
        ReasoningMode.SYMBOLIC_ONLY,
        ReasoningMode.NEURAL_ONLY,
        ReasoningMode.PARALLEL,
        ReasoningMode.SEQUENTIAL,
        ReasoningMode.ADAPTIVE
    ]
    
    input_data = "Socrates is a human and a philosopher. Therefore, he should be wise."
    query = "wise(socrates)"
    
    results = {}
    
    for mode in reasoning_modes:
        print(f"\n🧠 Testing {mode.value} reasoning...")
        
        result = await bridge.reason(
            input_data=input_data,
            query=query,
            reasoning_mode=mode
        )
        
        results[mode.value] = result
        
        print(f"✅ Result: {result.final_conclusion}")
        print(f"  - Confidence: {result.confidence:.3f}")
        print(f"  - Agreement: {result.agreement_score:.3f}")
        print(f"  - Processing time: {result.processing_time:.3f}s")
        print(f"  - Mode used: {result.reasoning_mode.value}")
    
    # Test consensus reasoning with disagreement
    print("\n🧠 Testing consensus reasoning with potential disagreement...")
    
    # Create scenario where neural and symbolic might disagree
    ambiguous_input = "The situation is complex and unclear."
    consensus_result = await bridge.reason(
        input_data=ambiguous_input,
        reasoning_mode=ReasoningMode.CONSENSUS
    )
    
    print(f"✅ Consensus result: {consensus_result.final_conclusion}")
    print(f"  - Agreement score: {consensus_result.agreement_score:.3f}")
    print(f"  - Explanation: {consensus_result.explanation[0] if consensus_result.explanation else 'No explanation'}")
    
    # Get comprehensive statistics
    stats = bridge.get_statistics()
    print(f"\n📊 Bridge Statistics:")
    print(f"  - Total inferences: {stats['total_inferences']}")
    print(f"  - Neural only: {stats['neural_only_count']}")
    print(f"  - Symbolic only: {stats['symbolic_only_count']}")
    print(f"  - Hybrid: {stats['hybrid_count']}")
    print(f"  - Average agreement: {stats['agreement_rate']:.3f}")
    print(f"  - Average confidence: {stats['average_confidence']:.3f}")
    print(f"  - Concept groundings: {stats['concept_groundings']}")
    
    return bridge, results

async def demonstrate_complex_reasoning():
    """Demonstrate complex reasoning scenario"""
    print("\n🔹 Complex Reasoning Demonstration")
    print("=" * 50)
    
    # Create a more complex knowledge scenario
    reasoner = SymbolicReasoner()
    kg = KnowledgeGraph()
    
    # Medical knowledge example
    print("📝 Building medical knowledge base...")
    
    # Add facts about diseases and symptoms
    reasoner.add_fact("has_symptom(flu, fever)")
    reasoner.add_fact("has_symptom(flu, cough)")
    reasoner.add_fact("has_symptom(cold, cough)")
    reasoner.add_fact("has_symptom(cold, runny_nose)")
    reasoner.add_fact("has_symptom(covid, fever)")
    reasoner.add_fact("has_symptom(covid, cough)")
    reasoner.add_fact("has_symptom(covid, loss_of_taste)")
    
    # Add diagnostic rules
    reasoner.add_rule("IF has_symptom(X, fever) AND has_symptom(X, cough) AND has_symptom(X, loss_of_taste) THEN likely_diagnosis(X, covid)", priority=10, confidence=0.85)
    reasoner.add_rule("IF has_symptom(X, fever) AND has_symptom(X, cough) AND NOT has_symptom(X, loss_of_taste) THEN likely_diagnosis(X, flu)", priority=8, confidence=0.75)
    reasoner.add_rule("IF has_symptom(X, cough) AND has_symptom(X, runny_nose) AND NOT has_symptom(X, fever) THEN likely_diagnosis(X, cold)", priority=6, confidence=0.70)
    
    # Build knowledge graph
    kg.add_entity("Flu", EntityType.CONCEPT, {"type": "viral_infection"})
    kg.add_entity("Cold", EntityType.CONCEPT, {"type": "viral_infection"})
    kg.add_entity("COVID", EntityType.CONCEPT, {"type": "viral_infection"})
    kg.add_entity("Fever", EntityType.CONCEPT, {"type": "symptom"})
    kg.add_entity("Cough", EntityType.CONCEPT, {"type": "symptom"})
    
    kg.add_relation("Flu", "Fever", RelationType.HAS_PROPERTY)
    kg.add_relation("COVID", "Fever", RelationType.HAS_PROPERTY)
    kg.add_relation("Flu", "Cold", RelationType.SIMILAR_TO)
    
    # Create bridge
    bridge = NeuralSymbolicBridge(reasoner, kg)
    
    # Test diagnostic reasoning
    print("\n🩺 Testing diagnostic reasoning...")
    
    patient_cases = [
        {
            "name": "Patient A",
            "symptoms": "Patient has fever, cough, and loss of taste",
            "facts": ["has_symptom(patient_a, fever)", "has_symptom(patient_a, cough)", "has_symptom(patient_a, loss_of_taste)"]
        },
        {
            "name": "Patient B", 
            "symptoms": "Patient has cough and runny nose but no fever",
            "facts": ["has_symptom(patient_b, cough)", "has_symptom(patient_b, runny_nose)"]
        }
    ]
    
    for case in patient_cases:
        print(f"\n👤 Analyzing {case['name']}:")
        print(f"   Symptoms: {case['symptoms']}")
        
        # Add patient facts
        for fact in case['facts']:
            reasoner.add_fact(fact)
        
        # Perform reasoning
        result = await bridge.reason(
            input_data=case['symptoms'],
            reasoning_mode=ReasoningMode.PARALLEL
        )
        
        print(f"   🔍 Diagnosis: {result.final_conclusion}")
        print(f"   📊 Confidence: {result.confidence:.3f}")
        
        # Show reasoning chain
        if result.symbolic_output and result.symbolic_output.conclusions:
            print(f"   🧠 Reasoning chain:")
            for conclusion in result.symbolic_output.conclusions:
                print(f"      - {conclusion.conclusion}")
    
    # Test forward chaining to discover all possible diagnoses
    print(f"\n🔍 Discovering all possible diagnoses...")
    new_facts = await reasoner.forward_chain(max_steps=10)
    
    diagnoses = [fact for fact in new_facts if "likely_diagnosis" in str(fact.conclusion)]
    print(f"✅ Found {len(diagnoses)} diagnostic conclusions:")
    for diagnosis in diagnoses:
        print(f"  - {diagnosis.conclusion} (confidence: {diagnosis.confidence:.3f})")
    
    return bridge

async def benchmark_performance():
    """Benchmark performance of different reasoning modes"""
    print("\n🔹 Performance Benchmark")
    print("=" * 40)
    
    # Create reasonably sized knowledge base
    reasoner = SymbolicReasoner()
    kg = KnowledgeGraph()
    
    # Add many facts and rules for realistic benchmark
    print("📝 Building large knowledge base...")
    
    # Add hierarchical knowledge
    categories = ["animal", "plant", "mineral", "concept", "person"]
    instances = ["dog", "cat", "tree", "flower", "rock", "crystal", "wisdom", "knowledge", "socrates", "plato"]
    
    for instance in instances:
        category = np.random.choice(categories)
        reasoner.add_fact(f"is_a({instance}, {category})")
        kg.add_entity(instance.capitalize(), EntityType.CONCEPT)
    
    # Add rules
    reasoning_rules = [
        "IF is_a(X, animal) THEN living(X)",
        "IF is_a(X, plant) THEN living(X)",
        "IF living(X) THEN needs_energy(X)",
        "IF is_a(X, person) AND wise(X) THEN teacher(X)",
        "IF is_a(X, concept) THEN abstract(X)"
    ]
    
    for rule in reasoning_rules:
        reasoner.add_rule(rule, confidence=0.8)
    
    bridge = NeuralSymbolicBridge(reasoner, kg)
    
    # Benchmark different modes
    test_inputs = [
        "What can we infer about living things?",
        "Are there any teachers in the knowledge base?",
        "What abstract concepts do we know about?"
    ]
    
    modes_to_test = [
        ReasoningMode.SYMBOLIC_ONLY,
        ReasoningMode.NEURAL_ONLY,
        ReasoningMode.PARALLEL,
        ReasoningMode.ADAPTIVE
    ]
    
    benchmark_results = {}
    
    for mode in modes_to_test:
        print(f"\n⏱️ Benchmarking {mode.value}...")
        mode_times = []
        mode_confidences = []
        
        for input_text in test_inputs:
            start_time = datetime.now()
            
            result = await bridge.reason(
                input_data=input_text,
                reasoning_mode=mode
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            mode_times.append(processing_time)
            mode_confidences.append(result.confidence)
            
            print(f"  ✅ Input: '{input_text[:30]}...'")
            print(f"     Time: {processing_time:.3f}s | Confidence: {result.confidence:.3f}")
        
        avg_time = sum(mode_times) / len(mode_times)
        avg_confidence = sum(mode_confidences) / len(mode_confidences)
        
        benchmark_results[mode.value] = {
            "avg_time": avg_time,
            "avg_confidence": avg_confidence,
            "total_tests": len(test_inputs)
        }
        
        print(f"  📊 Average time: {avg_time:.3f}s | Average confidence: {avg_confidence:.3f}")
    
    # Summary
    print(f"\n📊 Benchmark Summary:")
    fastest_mode = min(benchmark_results.keys(), key=lambda x: benchmark_results[x]["avg_time"])
    most_confident_mode = max(benchmark_results.keys(), key=lambda x: benchmark_results[x]["avg_confidence"])
    
    print(f"  🏃 Fastest mode: {fastest_mode} ({benchmark_results[fastest_mode]['avg_time']:.3f}s)")
    print(f"  🎯 Most confident: {most_confident_mode} ({benchmark_results[most_confident_mode]['avg_confidence']:.3f})")
    
    return benchmark_results

async def main():
    """Main test execution"""
    print("🔹 NEUROSYMBOLIC AI FRAMEWORK TESTING")
    print("=" * 60)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all tests
        reasoner = await test_symbolic_reasoner()
        kg = await test_knowledge_graph()
        bridge, results = await test_neural_symbolic_bridge()
        
        # Complex demonstration
        await demonstrate_complex_reasoning()
        
        # Performance benchmark
        benchmark_results = await benchmark_performance()
        
        # Final summary
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\n📊 Final Statistics:")
        reasoner_stats = reasoner.get_statistics()
        kg_stats = kg.get_statistics()
        bridge_stats = bridge.get_statistics()
        
        print(f"  Symbolic Reasoner:")
        print(f"    - Facts: {reasoner_stats['knowledge_base']['facts_count']}")
        print(f"    - Rules: {reasoner_stats['knowledge_base']['rules_count']}")
        print(f"    - Inferences: {reasoner_stats['reasoner_metrics']['total_inferences']}")
        
        print(f"  Knowledge Graph:")
        print(f"    - Entities: {kg_stats['total_entities']}")
        print(f"    - Relations: {kg_stats['total_relations']}")
        print(f"    - Density: {kg_stats['graph_density']:.4f}")
        
        print(f"  Neural-Symbolic Bridge:")
        print(f"    - Total inferences: {bridge_stats['total_inferences']}")
        print(f"    - Hybrid inferences: {bridge_stats['hybrid_count']}")
        print(f"    - Average agreement: {bridge_stats['agreement_rate']:.3f}")
        
        print(f"\n✅ Neurosymbolic AI Framework is fully operational!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(main())