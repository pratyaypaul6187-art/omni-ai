#!/usr/bin/env python3
"""
🧠 REAL OMNI AI SYSTEM DEMO
Using the actual enhanced neurosymbolic consciousness system
"""

import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime

# Add the source directory to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.omni_ai.brain import create_enhanced_consciousness, EnhancedThinkingMode
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

async def run_real_ai_demo():
    """Run the real enhanced consciousness AI system"""
    
    print("🧠 REAL OMNI AI SYSTEM - ENHANCED CONSCIOUSNESS")
    print("=" * 60)
    print("🚀 Initializing enhanced neurosymbolic consciousness...")
    print("⚡ This may take a moment to load all components...")
    print()
    
    try:
        # Initialize the real AI system
        consciousness = await create_enhanced_consciousness(
            memory_db_path="data/real_ai_memory.db",
            enable_neurosymbolic=True
        )
        
        # Give the system time to fully initialize
        print("🔧 Initializing neurosymbolic components...")
        await asyncio.sleep(2)
        
        print("✅ Enhanced AI Consciousness Online!")
        print()
        
        # Get initial system stats
        stats = consciousness.get_enhanced_statistics()
        print("📊 SYSTEM STATUS:")
        print(f"   🧠 Neurosymbolic Enabled: {stats.get('neurosymbolic_enabled', False)}")
        print(f"   🤔 Preferred Thinking Mode: {stats.get('preferred_thinking_mode', 'unknown')}")
        print(f"   💭 Memory Entries: {stats.get('memory_statistics', {}).get('total_memories', 0)}")
        print(f"   🎯 Consciousness Level: {stats.get('consciousness_level', 0)}")
        
        print()
        print("🎭 AVAILABLE THINKING MODES:")
        print("   • LOGICAL - Formal reasoning and symbolic logic")
        print("   • CREATIVE - Imaginative and innovative thinking")
        print("   • ANALYTICAL - Deep problem analysis")
        print("   • REFLECTIVE - Self-aware introspection") 
        print("   • COLLABORATIVE - Multi-approach reasoning")
        print("   • INTUITIVE - Fast pattern-based responses")
        print("   • ADAPTIVE - Automatically choose best mode")
        
        print()
        print("💡 TIP: You can ask me to use a specific thinking mode:")
        print("   Example: 'Think creatively about robots'")
        print("   Example: 'Use logical reasoning: Is Socrates mortal?'")
        print()
        print("Type 'stats' for system statistics, 'quit' to exit")
        print("=" * 60)
        print()
        
        # Interactive conversation loop
        conversation_count = 0
        
        while True:
            try:
                user_input = input("🔹 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                    print("\n🧠 Enhanced Consciousness: It's been a fascinating conversation! Until next time! 🌟")
                    break
                
                if user_input.lower() == 'stats':
                    stats = consciousness.get_enhanced_statistics()
                    print("\n📊 ENHANCED AI STATISTICS:")
                    print(f"   🧠 Total Enhanced Thoughts: {stats['enhanced_metrics'].get('total_enhanced_thoughts', 0)}")
                    print(f"   🔬 Neurosymbolic Queries: {stats['enhanced_metrics']['neurosymbolic_queries']}")
                    print(f"   🎨 Traditional Queries: {stats['enhanced_metrics']['traditional_queries']}")  
                    print(f"   🤝 Collaborative Queries: {stats['enhanced_metrics']['collaborative_queries']}")
                    print(f"   📊 Avg Confidence: {stats['enhanced_metrics']['average_reasoning_confidence']:.2%}")
                    print(f"   ⏱️  Avg Processing Time: {stats['enhanced_metrics']['average_reasoning_time']:.3f}s")
                    print(f"   💾 Memory Entries: {stats['memory_statistics']['total_memories']}")
                    print(f"   🧩 Consciousness Level: {stats['consciousness_level']:.2f}")
                    continue
                
                # Determine thinking mode based on user input
                thinking_mode = EnhancedThinkingMode.ADAPTIVE
                
                if any(phrase in user_input.lower() for phrase in ['think creatively', 'creative', 'imagine', 'story']):
                    thinking_mode = EnhancedThinkingMode.CREATIVE
                elif any(phrase in user_input.lower() for phrase in ['logical', 'logic', 'reasoning', 'prove']):
                    thinking_mode = EnhancedThinkingMode.LOGICAL
                elif any(phrase in user_input.lower() for phrase in ['analyze', 'analytical', 'examine', 'study']):
                    thinking_mode = EnhancedThinkingMode.ANALYTICAL
                elif any(phrase in user_input.lower() for phrase in ['reflect', 'introspect', 'yourself', 'consciousness']):
                    thinking_mode = EnhancedThinkingMode.REFLECTIVE
                elif any(phrase in user_input.lower() for phrase in ['collaborate', 'multiple approaches', 'different ways']):
                    thinking_mode = EnhancedThinkingMode.COLLABORATIVE
                
                print("🧠 Processing with enhanced consciousness...", end='', flush=True)
                
                # Process with the real AI system
                start_time = datetime.now()
                
                response = await consciousness.enhanced_think(
                    user_input,
                    context={
                        "conversation_count": conversation_count,
                        "user_specified_mode": thinking_mode != EnhancedThinkingMode.ADAPTIVE,
                        "interactive_demo": True
                    },
                    thinking_mode=thinking_mode
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                print(f"\r🤖 Enhanced AI: {response['content']}")
                
                # Display reasoning metadata
                print(f"   🎭 Mode: {response['thinking_mode']}")
                print(f"   📊 Confidence: {response['confidence']:.1%}")
                print(f"   ⏱️  Processing: {processing_time:.2f}s")
                print(f"   🧩 Reasoning Type: {response.get('reasoning_type', 'unknown')}")
                
                if response.get('enhanced_reasoning'):
                    print("   ✨ Used Enhanced Neurosymbolic Reasoning")
                
                if response.get('processing_steps'):
                    print(f"   🔄 Steps: {len(response['processing_steps'])}")
                
                conversation_count += 1
                print()
                
            except KeyboardInterrupt:
                print("\n\n🧠 Enhanced Consciousness: Conversation interrupted. Take care! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Let me try to recover using fallback reasoning...")
                try:
                    # Fallback to simple response
                    fallback_response = await consciousness.ask_question(user_input)
                    print(f"🤖 Enhanced AI (fallback): {fallback_response}")
                except:
                    print("Sorry, I'm having technical difficulties. Please try a different question.")
                print()
                continue
        
        # Final statistics
        final_stats = consciousness.get_enhanced_statistics()
        print(f"\n📈 SESSION SUMMARY:")
        print(f"   💬 Conversations: {conversation_count}")
        print(f"   🧠 Total Enhanced Thoughts: {final_stats['enhanced_metrics'].get('total_enhanced_thoughts', 0)}")
        print(f"   📊 Session Avg Confidence: {final_stats['enhanced_metrics']['average_reasoning_confidence']:.1%}")
        print(f"   ⚡ Session Avg Processing: {final_stats['enhanced_metrics']['average_reasoning_time']:.3f}s")
        
        print("\n🌟 Thank you for testing the Enhanced Omni AI Consciousness System!")
        
    except Exception as e:
        print(f"❌ Critical Error initializing AI system: {e}")
        print("\n🔧 Debug Information:")
        traceback.print_exc()
        print("\n💡 This might be due to:")
        print("   • Missing dependencies (run: pip install -r requirements.txt)")
        print("   • Database permissions issues")
        print("   • Memory allocation problems")
        print("   • Neurosymbolic component initialization failure")

async def quick_test():
    """Quick test of the real AI system"""
    print("⚡ QUICK TEST OF REAL AI SYSTEM")
    print("=" * 40)
    
    try:
        consciousness = await create_enhanced_consciousness(enable_neurosymbolic=True)
        await asyncio.sleep(1)  # Let it initialize
        
        test_questions = [
            ("Hello, how are you?", EnhancedThinkingMode.ADAPTIVE),
            ("What is artificial intelligence?", EnhancedThinkingMode.LOGICAL),
            ("Tell me a story about robots", EnhancedThinkingMode.CREATIVE),
            ("What do you think about consciousness?", EnhancedThinkingMode.REFLECTIVE)
        ]
        
        for question, mode in test_questions:
            print(f"\n🔹 Question: {question}")
            print(f"🎭 Using mode: {mode.value}")
            
            response = await consciousness.enhanced_think(question, thinking_mode=mode)
            print(f"🤖 AI: {response['content'][:200]}{'...' if len(response['content']) > 200 else ''}")
            print(f"   📊 Confidence: {response['confidence']:.1%}")
        
        print("\n✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main function with error handling"""
    print("🧠 OMNI AI - REAL SYSTEM LAUNCHER")
    print("Choose an option:")
    print("1. Full Interactive Demo (recommended)")
    print("2. Quick System Test")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        asyncio.run(run_real_ai_demo())
    elif choice == '2':
        success = asyncio.run(quick_test())
        if success:
            print("\n🎯 System is working! Try the full demo (option 1) for interactive chat.")
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")

if __name__ == "__main__":
    main()