#!/usr/bin/env python3
"""
🔮 ADVANCED REASONING ENGINE DEMO
Comprehensive demonstration of temporal logic, probabilistic inference, causal reasoning,
abductive reasoning, analogical reasoning, and counterfactual reasoning capabilities
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omni_ai.neurosymbolic.advanced_reasoner import (
    AdvancedReasoner, TemporalOperator, UncertaintyType, ReasoningMode,
    create_advanced_reasoner
)
from omni_ai.neurosymbolic.symbolic_reasoner import SymbolicReasoner

def print_section(title: str, emoji: str = "🔮"):
    """Print a formatted section header"""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n📌 {title}")
    print("-" * (len(title) + 4))

def print_result(result, show_details: bool = True):
    """Print reasoning result in a formatted way"""
    print(f"🎯 Query: {result.query}")
    print(f"📊 Answer: {result.answer}")
    print(f"🎲 Confidence: {result.confidence:.3f}")
    print(f"🧠 Mode: {result.reasoning_mode.value}")
    print(f"⏱️  Time: {result.computation_time:.3f}s")
    
    if show_details:
        print(f"💡 Explanation: {result.explanation.conclusion}")
        print("🔗 Reasoning Chain:")
        for i, step in enumerate(result.explanation.reasoning_chain, 1):
            print(f"   {i}. {step}")
        
        if result.explanation.supporting_facts:
            print(f"📚 Supporting Facts: {', '.join(result.explanation.supporting_facts)}")

async def demo_temporal_reasoning():
    """Demonstrate temporal reasoning capabilities"""
    print_section("TEMPORAL REASONING", "⏰")
    
    reasoner = create_advanced_reasoner()
    
    # Add temporal facts
    print_subsection("Adding Temporal Facts")
    
    # Morning facts
    now = datetime.now()
    morning = now.replace(hour=8, minute=0)
    evening = now.replace(hour=18, minute=0)
    
    await reasoner.add_temporal_fact(
        "sun_is_up(daytime)", confidence=1.0,
        valid_from=morning,
        valid_until=evening,
        temporal_operator=TemporalOperator.DURING
    )
    
    await reasoner.add_temporal_fact(
        "people_commuting(rush_hour)", confidence=0.9,
        valid_from=morning.replace(hour=7),
        valid_until=morning.replace(hour=10)
    )
    
    await reasoner.add_temporal_fact(
        "traffic_heavy(city)", confidence=0.8,
        valid_from=morning.replace(hour=7, minute=30),
        valid_until=morning.replace(hour=9, minute=30)
    )
    
    print(f"✅ Added {len(reasoner.temporal_facts)} temporal facts")
    
    # Query at different times
    print_subsection("Temporal Queries")
    
    # Query during morning rush hour
    result = await reasoner.temporal_query("traffic_heavy(city)", at_time=morning.replace(hour=8))
    print_result(result)
    
    # Query in the afternoon
    result = await reasoner.temporal_query("traffic_heavy(city)", at_time=now.replace(hour=15))
    print_result(result)
    
    return reasoner

async def demo_probabilistic_reasoning(reasoner):
    """Demonstrate probabilistic reasoning capabilities"""
    print_section("PROBABILISTIC REASONING", "🎲")
    
    # Add probabilistic facts
    print_subsection("Adding Probabilistic Facts")
    
    await reasoner.add_probabilistic_fact("rain_today(weather)", 0.3, evidence=["weather_forecast"])
    await reasoner.add_probabilistic_fact("umbrella_needed(person)", 0.7, evidence=["rain_today"])
    await reasoner.add_probabilistic_fact("traffic_jam(city)", 0.6, evidence=["rain_today", "rush_hour"])
    
    # Fuzzy facts
    await reasoner.add_probabilistic_fact("weather_nice(today)", 0.8, uncertainty_type=UncertaintyType.FUZZY)
    await reasoner.add_probabilistic_fact("mood_good(person)", 0.6, uncertainty_type=UncertaintyType.FUZZY)
    
    print(f"✅ Added {len(reasoner.probabilistic_facts)} probabilistic facts")
    
    # Test different inference methods
    print_subsection("Bayesian Inference")
    result = await reasoner.probabilistic_query("rain_today(weather)", method="bayesian")
    print_result(result)
    
    print_subsection("Fuzzy Logic Inference")
    result = await reasoner.probabilistic_query("weather_nice(today)", method="fuzzy")
    print_result(result)
    
    print_subsection("Monte Carlo Inference")
    result = await reasoner.probabilistic_query("umbrella_needed(person)", method="monte_carlo")
    print_result(result)
    
    return reasoner

async def demo_causal_reasoning(reasoner):
    """Demonstrate causal reasoning capabilities"""
    print_section("CAUSAL REASONING", "🔗")
    
    # Add causal relations
    print_subsection("Adding Causal Relations")
    
    await reasoner.add_causal_relation("rain(weather)", "wet_streets(road)", 0.9, mechanism="water_accumulation")
    await reasoner.add_causal_relation("wet_streets(road)", "slippery_roads(surface)", 0.8, mechanism="reduced_friction")
    await reasoner.add_causal_relation("slippery_roads(surface)", "accidents(traffic)", 0.4, mechanism="loss_of_control")
    await reasoner.add_causal_relation("studying(student)", "good_grades(academic)", 0.7, mechanism="knowledge_acquisition")
    await reasoner.add_causal_relation("exercise(person)", "good_health(physical)", 0.8, mechanism="physical_fitness")
    
    print(f"✅ Added {len(reasoner.causal_relations)} causal relations")
    
    # Causal reasoning without intervention
    print_subsection("Basic Causal Reasoning")
    result = await reasoner.causal_reasoning("accidents(traffic)")
    print_result(result)
    
    # Causal reasoning with intervention
    print_subsection("Causal Reasoning with Intervention")
    intervention = {"rain(weather)": False}  # Intervening to prevent rain
    result = await reasoner.causal_reasoning("accidents(traffic)", intervention=intervention)
    print_result(result)
    
    return reasoner

async def demo_abductive_reasoning(reasoner):
    """Demonstrate abductive reasoning (finding best explanations)"""
    print_section("ABDUCTIVE REASONING", "🕵️")
    
    # Add some domain knowledge
    reasoner.base_reasoner.add_fact("virus_present(body)", 0.8)
    reasoner.base_reasoner.add_fact("bacteria_present(body)", 0.3)
    reasoner.base_reasoner.add_rule("IF virus_present(body) THEN fever(symptom)", 1, 0.9)
    reasoner.base_reasoner.add_rule("IF bacteria_present(body) THEN fever(symptom)", 1, 0.7)
    reasoner.base_reasoner.add_rule("IF virus_present(body) THEN fatigue(symptom)", 1, 0.8)
    reasoner.base_reasoner.add_rule("IF exercise_overdone(activity) THEN fatigue(symptom)", 1, 0.6)
    
    print_subsection("Finding Best Explanation")
    
    # Find best explanation for observed fever
    result = await reasoner.abductive_reasoning("fever(symptom)")
    print_result(result)
    
    # Find best explanation for observed fatigue
    result = await reasoner.abductive_reasoning("fatigue(symptom)")
    print_result(result)
    
    return reasoner

async def demo_analogical_reasoning(reasoner):
    """Demonstrate analogical reasoning"""
    print_section("ANALOGICAL REASONING", "🔄")
    
    # Add knowledge about source domain
    reasoner.base_reasoner.add_fact("has_nucleus(atom)", 1.0)
    reasoner.base_reasoner.add_fact("orbit(electrons, nucleus)", 0.9)
    reasoner.base_reasoner.add_fact("has_sun(solar_system)", 1.0)
    reasoner.base_reasoner.add_fact("orbit(planets, sun)", 1.0)
    
    reasoner.base_reasoner.add_rule("IF has_nucleus(atom) THEN stable_structure(atom)", 1, 0.8)
    reasoner.base_reasoner.add_rule("IF orbit(electrons, nucleus) THEN energy_levels(electrons)", 1, 0.7)
    
    print_subsection("Analogical Transfer")
    
    # Analogical reasoning from atomic model to solar system
    result = await reasoner.analogical_reasoning(
        "atom with nucleus and electrons",
        "solar system with sun and planets",
        similarity_threshold=0.3
    )
    print_result(result)
    
    return reasoner

async def demo_counterfactual_reasoning(reasoner):
    """Demonstrate counterfactual reasoning (what-if analysis)"""
    print_section("COUNTERFACTUAL REASONING", "❓")
    
    # Add knowledge about a scenario
    reasoner.base_reasoner.add_fact("studied_hard(student)", 1.0)
    reasoner.base_reasoner.add_fact("passed_exam(student)", 1.0)
    reasoner.base_reasoner.add_rule("IF studied_hard(student) THEN passed_exam(student)", 1, 0.9)
    reasoner.base_reasoner.add_rule("IF not_studied(student) THEN failed_exam(student)", 1, 0.8)
    
    print_subsection("What-If Analysis")
    
    # Counterfactual: What if the person didn't study?
    result = await reasoner.counterfactual_reasoning(
        "passed_exam(student)",
        "not_studied(student)"
    )
    print_result(result)
    
    return reasoner

async def demo_explanation_generation(reasoner):
    """Demonstrate explanation generation capabilities"""
    print_section("EXPLANATION GENERATION", "💡")
    
    print_subsection("Basic Explanation")
    explanation = await reasoner.explain_reasoning("rain_today(weather)", depth=1)
    print(f"🧩 Conclusion: {explanation.conclusion}")
    print(f"🎯 Confidence: {explanation.confidence:.3f}")
    print("📝 Reasoning Chain:")
    for i, step in enumerate(explanation.reasoning_chain, 1):
        print(f"   {i}. {step}")
    
    print_subsection("Detailed Explanation")
    explanation = await reasoner.explain_reasoning("accidents(traffic)", depth=3)
    print(f"🧩 Conclusion: {explanation.conclusion}")
    print(f"🎯 Confidence: {explanation.confidence:.3f}")
    print("📝 Reasoning Chain:")
    for i, step in enumerate(explanation.reasoning_chain, 1):
        print(f"   {i}. {step}")
    
    if explanation.alternative_explanations:
        print(f"🔀 Alternatives: {', '.join(explanation.alternative_explanations)}")
    
    return reasoner

async def demo_integrated_reasoning(reasoner):
    """Demonstrate integration of multiple reasoning types"""
    print_section("INTEGRATED REASONING SCENARIO", "🧩")
    
    print_subsection("Medical Diagnosis Scenario")
    
    # Set up a medical scenario with temporal, probabilistic, and causal elements
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    
    # Temporal facts about symptoms
    await reasoner.add_temporal_fact("fever(patient)", confidence=0.9, valid_from=yesterday)
    await reasoner.add_temporal_fact("headache(patient)", confidence=0.8, valid_from=yesterday)
    await reasoner.add_temporal_fact("fatigue(patient)", confidence=0.7, valid_from=yesterday)
    
    # Probabilistic facts about conditions
    await reasoner.add_probabilistic_fact("flu(patient)", 0.6, evidence=["fever", "headache", "fatigue"])
    await reasoner.add_probabilistic_fact("cold(patient)", 0.4, evidence=["headache", "fatigue"])
    await reasoner.add_probabilistic_fact("stress(patient)", 0.3, evidence=["headache", "fatigue"])
    
    # Causal relationships
    await reasoner.add_causal_relation("flu(patient)", "fever(patient)", 0.9)
    await reasoner.add_causal_relation("flu(patient)", "headache(patient)", 0.7)
    await reasoner.add_causal_relation("flu(patient)", "fatigue(patient)", 0.8)
    
    print("🏥 Medical scenario setup complete")
    
    # Temporal query: Are symptoms still present?
    result = await reasoner.temporal_query("fever(patient)", at_time=now)
    print(f"\n🌡️  Fever currently present: {len(result.answer) > 0}")
    
    # Probabilistic diagnosis
    result = await reasoner.probabilistic_query("flu(patient)", method="bayesian")
    print(f"\n🦠 Flu probability: {result.confidence:.3f}")
    
    # Abductive reasoning: What's the best explanation for symptoms?
    result = await reasoner.abductive_reasoning("fever(patient)")
    print(f"\n🔍 Best explanation for fever: {result.answer['cause'] if result.answer else 'None'}")
    
    # Causal reasoning: If we treat the flu, what happens?
    intervention = {"flu(patient)": False}  # Treatment eliminates flu
    result = await reasoner.causal_reasoning("fever(patient)", intervention=intervention)
    print(f"\n💊 Effect of treating flu on fever: {result.answer:.3f}")
    
    return reasoner

def demo_reasoning_statistics(reasoner):
    """Display comprehensive reasoning statistics"""
    print_section("REASONING STATISTICS", "📊")
    
    stats = reasoner.get_reasoning_statistics()
    
    print(f"🔢 Total Inferences: {stats['total_inferences']}")
    print(f"✅ Successful Inferences: {stats['successful_inferences']}")
    print(f"📈 Success Rate: {stats['success_rate']:.1%}")
    print(f"🎯 Average Confidence: {stats['average_confidence']:.3f}")
    
    print_subsection("Reasoning Modes Used")
    for mode, count in stats['reasoning_modes_used'].items():
        print(f"   {mode}: {count} inferences")
    
    print_subsection("Knowledge Base Statistics")
    print(f"   Temporal Facts: {stats['temporal_facts']}")
    print(f"   Probabilistic Facts: {stats['probabilistic_facts']}")
    print(f"   Causal Relations: {stats['causal_relations']}")
    print(f"   Reasoning History: {stats['reasoning_history_length']} entries")
    print(f"   Explanation Requests: {stats['explanation_requests']}")

async def main():
    """Run the complete advanced reasoning demonstration"""
    print("🔮 OMNI AI - ADVANCED REASONING ENGINE DEMO")
    print("=" * 50)
    print("This demo showcases sophisticated reasoning capabilities including:")
    print("• ⏰ Temporal Logic & Time-based Reasoning")
    print("• 🎲 Probabilistic Inference & Uncertainty")
    print("• 🔗 Causal Reasoning & Interventions")
    print("• 🕵️ Abductive Reasoning & Best Explanations")
    print("• 🔄 Analogical Reasoning & Knowledge Transfer")
    print("• ❓ Counterfactual Reasoning & What-If Analysis")
    print("• 💡 Automated Explanation Generation")
    print("• 🧩 Integrated Multi-Modal Reasoning")
    
    try:
        # Demo each reasoning capability
        reasoner = await demo_temporal_reasoning()
        reasoner = await demo_probabilistic_reasoning(reasoner)
        reasoner = await demo_causal_reasoning(reasoner)
        reasoner = await demo_abductive_reasoning(reasoner)
        reasoner = await demo_analogical_reasoning(reasoner)
        reasoner = await demo_counterfactual_reasoning(reasoner)
        reasoner = await demo_explanation_generation(reasoner)
        reasoner = await demo_integrated_reasoning(reasoner)
        
        # Show comprehensive statistics
        demo_reasoning_statistics(reasoner)
        
        print_section("DEMO COMPLETED SUCCESSFULLY", "🎉")
        print("The Advanced Reasoning Engine is fully operational!")
        print("All advanced reasoning capabilities have been demonstrated:")
        print("✅ Temporal reasoning with time-based facts and queries")
        print("✅ Probabilistic inference with Bayesian, fuzzy, and Monte Carlo methods")
        print("✅ Causal reasoning with interventional analysis")
        print("✅ Abductive reasoning for finding best explanations")
        print("✅ Analogical reasoning for knowledge transfer")
        print("✅ Counterfactual reasoning for what-if scenarios")
        print("✅ Automated explanation generation with variable depth")
        print("✅ Integrated multi-modal reasoning scenarios")
        print("✅ Comprehensive performance metrics and statistics")
        
        print(f"\n🧠 The system processed {reasoner.get_reasoning_statistics()['total_inferences']} inferences")
        print(f"🎯 with {reasoner.get_reasoning_statistics()['success_rate']:.1%} success rate")
        print(f"⚡ Average confidence: {reasoner.get_reasoning_statistics()['average_confidence']:.3f}")
        
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())