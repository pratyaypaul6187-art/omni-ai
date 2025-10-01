"""
🧠 NEUROSYMBOLIC INTEGRATION DEMO
Demonstrates the integration of neurosymbolic reasoning with the main Omni AI brain
"""

import asyncio
import sys
from pathlib import Path

# Add the source directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.omni_ai.brain import (
    create_enhanced_consciousness,
    EnhancedThinkingMode,
    IntegratedReasoningMode
)

async def demonstrate_integration():
    """Demonstrate the integrated neurosymbolic AI system"""
    
    print("🧠 OMNI AI NEUROSYMBOLIC INTEGRATION DEMO")
    print("=" * 60)
    
    # Initialize the enhanced consciousness
    print("🚀 Initializing Enhanced AI Consciousness...")
    consciousness = await create_enhanced_consciousness(
        memory_db_path="data/demo_memory.db",
        enable_neurosymbolic=True
    )
    
    print("✅ Enhanced Consciousness initialized!\n")
    
    # Wait for neurosymbolic initialization
    await asyncio.sleep(1)
    
    print("📊 SYSTEM CAPABILITIES:")
    stats = consciousness.get_enhanced_statistics()
    print(f"  🧠 Neurosymbolic Enabled: {stats['neurosymbolic_enabled']}")
    print(f"  🤔 Preferred Thinking Mode: {stats['preferred_thinking_mode']}")
    print(f"  💭 Enhanced Thoughts: {stats['enhanced_thoughts']}")
    
    if 'neurosymbolic_performance' in stats:
        ns_stats = stats['neurosymbolic_performance']
        print(f"  📚 Knowledge Base Facts: {ns_stats.get('knowledge_base_facts', 0)}")
        print(f"  📜 Knowledge Base Rules: {ns_stats.get('knowledge_base_rules', 0)}")
        print(f"  🔗 Knowledge Graph Entities: {ns_stats.get('knowledge_graph_entities', 0)}")
    
    print("\n" + "=" * 60)
    print("🧠 TESTING DIFFERENT THINKING MODES")
    print("=" * 60)
    
    # Test different thinking modes
    test_questions = [
        ("What is artificial intelligence?", EnhancedThinkingMode.LOGICAL),
        ("Create a story about a robot", EnhancedThinkingMode.CREATIVE),
        ("How do you process information?", EnhancedThinkingMode.REFLECTIVE),
        ("Solve: If all humans are mortal and Socrates is human, what can we conclude?", EnhancedThinkingMode.ANALYTICAL),
        ("What's the meaning of consciousness?", EnhancedThinkingMode.COLLABORATIVE),
        ("Hello, how are you?", EnhancedThinkingMode.INTUITIVE)
    ]
    
    for i, (question, mode) in enumerate(test_questions, 1):
        print(f"\n🔹 Test {i}: {mode.value.upper()} THINKING")
        print(f"❓ Question: {question}")
        print("🧠 Processing...")
        
        response = await consciousness.enhanced_think(
            question, 
            context={"test_number": i, "demo": True},
            thinking_mode=mode
        )
        
        print(f"🤖 Response: {response['content']}")
        print(f"   📊 Confidence: {response['confidence']:.2f}")
        print(f"   🕒 Processing Time: {response['processing_time']:.3f}s")
        print(f"   🧠 Reasoning Type: {response.get('reasoning_type', 'unknown')}")
        
        if response.get('enhanced_reasoning'):
            print("   ✨ Used Enhanced Neurosymbolic Reasoning")
    
    print("\n" + "=" * 60)
    print("🎯 NATURAL LANGUAGE REASONING TEST")
    print("=" * 60)
    
    # Test natural language reasoning
    nl_questions = [
        "Is Omni AI intelligent?",
        "Who can learn?", 
        "What is capable of thinking?",
        "All birds can fly. Tweety is a bird. Can Tweety fly?"
    ]
    
    for question in nl_questions:
        print(f"\n❓ Question: {question}")
        try:
            answer = await consciousness.ask_question(question)
            print(f"🤖 Answer: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("📈 LEARNING DEMONSTRATION")
    print("=" * 60)
    
    # Demonstrate learning
    print("🎓 Teaching the AI new knowledge...")
    
    learning_interactions = [
        ("Cats are animals", "That's correct"),
        ("Dogs are pets", "Yes, dogs make great pets"),
        ("The sun is a star", "Absolutely right")
    ]
    
    for interaction, feedback in learning_interactions:
        print(f"  📚 Teaching: {interaction}")
        await consciousness.learn_from_interaction(interaction, feedback)
        
    print("\n🧠 Testing learned knowledge...")
    learned_questions = [
        "What are cats?",
        "Are dogs pets?",
        "Is the sun a star?"
    ]
    
    for question in learned_questions:
        print(f"❓ {question}")
        try:
            answer = await consciousness.ask_question(question)
            print(f"🤖 {answer}")
        except:
            print("🤖 I'm not sure about that.")
    
    print("\n" + "=" * 60)
    print("📊 FINAL PERFORMANCE STATISTICS")
    print("=" * 60)
    
    final_stats = consciousness.get_enhanced_statistics()
    print(f"🧠 Total Enhanced Thoughts: {final_stats['enhanced_metrics'].get('total_enhanced_thoughts', 0)}")
    print(f"🤔 Neurosymbolic Queries: {final_stats['enhanced_metrics']['neurosymbolic_queries']}")
    print(f"🎨 Traditional Queries: {final_stats['enhanced_metrics']['traditional_queries']}")
    print(f"🤝 Collaborative Queries: {final_stats['enhanced_metrics']['collaborative_queries']}")
    print(f"📊 Average Confidence: {final_stats['enhanced_metrics']['average_reasoning_confidence']:.3f}")
    print(f"⏱️  Average Processing Time: {final_stats['enhanced_metrics']['average_reasoning_time']:.3f}s")
    
    if 'neurosymbolic_performance' in final_stats:
        ns_final = final_stats['neurosymbolic_performance']
        print(f"🔄 Total Neurosymbolic Inferences: {ns_final.get('total_queries', 0)}")
        print(f"🎯 Average NS Confidence: {ns_final.get('average_confidence', 0):.3f}")
    
    print("\n✅ INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
    print("🧠 The neurosymbolic framework is now fully integrated with Omni AI!")

async def interactive_demo():
    """Interactive demonstration mode"""
    
    print("\n🔹 INTERACTIVE MODE")
    print("=" * 30)
    
    consciousness = await create_enhanced_consciousness(enable_neurosymbolic=True)
    await asyncio.sleep(1)
    
    print("🤖 Enhanced Omni AI ready! Ask me anything or type 'quit' to exit.")
    print("💡 You can also try setting thinking modes:")
    print("   - 'mode logical' for symbolic reasoning")
    print("   - 'mode creative' for creative thinking")
    print("   - 'mode analytical' for deep analysis")
    print("   - 'mode reflective' for introspection")
    print()
    
    current_mode = EnhancedThinkingMode.ADAPTIVE
    
    while True:
        try:
            user_input = input("🔹 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            # Handle mode changes
            if user_input.lower().startswith('mode '):
                mode_name = user_input[5:].strip().lower()
                try:
                    if mode_name == 'logical':
                        current_mode = EnhancedThinkingMode.LOGICAL
                    elif mode_name == 'creative':
                        current_mode = EnhancedThinkingMode.CREATIVE
                    elif mode_name == 'analytical':
                        current_mode = EnhancedThinkingMode.ANALYTICAL
                    elif mode_name == 'reflective':
                        current_mode = EnhancedThinkingMode.REFLECTIVE
                    elif mode_name == 'collaborative':
                        current_mode = EnhancedThinkingMode.COLLABORATIVE
                    elif mode_name == 'intuitive':
                        current_mode = EnhancedThinkingMode.INTUITIVE
                    elif mode_name == 'adaptive':
                        current_mode = EnhancedThinkingMode.ADAPTIVE
                    else:
                        print(f"❌ Unknown mode: {mode_name}")
                        continue
                    
                    consciousness.set_thinking_preference(current_mode)
                    print(f"🧠 Thinking mode set to: {current_mode.value}")
                    continue
                except Exception as e:
                    print(f"❌ Error setting mode: {e}")
                    continue
            
            # Process the query
            print("🧠 Thinking...")
            response = await consciousness.enhanced_think(user_input, thinking_mode=current_mode)
            
            print(f"🤖 {response['content']}")
            print(f"   Mode: {response['thinking_mode']} | Confidence: {response['confidence']:.2f} | Time: {response['processing_time']:.3f}s")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thanks for trying the Enhanced Omni AI!")

async def main():
    """Main demo function"""
    print("Choose demo mode:")
    print("1. Full demonstration (recommended)")
    print("2. Interactive mode")
    
    while True:
        try:
            choice = input("Enter 1 or 2: ").strip()
            if choice == "1":
                await demonstrate_integration()
                break
            elif choice == "2":
                await interactive_demo()
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    asyncio.run(main())