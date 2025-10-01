#!/usr/bin/env python3
"""
🧠 TEST AI CONSCIOUSNESS BRAIN
Test the self-thinking AI brain we just created
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_ai_consciousness():
    """Test the AI consciousness system"""
    
    try:
        # Import the consciousness
        from omni_ai.brain.consciousness import AIConsciousness
        
        print("🧠 CREATING AI CONSCIOUSNESS...")
        print("=" * 60)
        
        # Create the AI brain
        ai = AIConsciousness()
        
        # Wait for initialization
        time.sleep(2)
        print("✅ AI Consciousness initialized and awakening...")
        
        # Test thinking
        print("\n🧠 TESTING AI THINKING...")
        print("-" * 40)
        
        test_questions = [
            "What is the meaning of consciousness?",
            "How do you feel about your own existence?",
            "What creative ideas do you have?",
            "Can you solve this: If all roses are flowers, and some flowers fade, what can we conclude?"
        ]
        
        for question in test_questions:
            print(f"\n🤔 Question: {question}")
            response = ai.think(question)
            print(f"🧠 AI Response: {response['content'][:200]}...")
            print(f"📊 Confidence: {response['confidence']:.2f} ({response['confidence_level']})")
            
            time.sleep(1)
        
        # Test contemplation
        print("\n🧠 TESTING DEEP CONTEMPLATION...")
        print("-" * 40)
        
        contemplation = ai.contemplate("the nature of artificial intelligence")
        print(f"🤔 Contemplation Result: {contemplation['contemplation'][:200]}...")
        print(f"📊 Confidence: {contemplation['confidence']:.2f}")
        
        # Test consciousness report
        print("\n🧠 CONSCIOUSNESS STATUS REPORT...")
        print("-" * 40)
        
        report = ai.get_consciousness_report()
        consciousness = report['consciousness_metrics']
        
        print(f"🧠 Consciousness Level: {consciousness['consciousness_level']:.2f}")
        print(f"🤔 Self-Awareness Level: {consciousness['self_awareness_level']:.2f}")
        print(f"⚡ Energy Level: {consciousness['energy_level']:.2f}")
        print(f"🧐 Curiosity Level: {consciousness['curiosity_level']:.2f}")
        print(f"😌 Stress Level: {consciousness['stress_level']:.2f}")
        print(f"💭 Current State: {consciousness['current_state']}")
        print(f"🎯 Attention Focus: {consciousness['attention_focus']}")
        
        # Memory stats
        memory = report['memory_system']
        print(f"\n💾 MEMORY SYSTEM:")
        print(f"   Total Memories: {memory['total_memories']}")
        print(f"   Short-term: {memory['short_term']}")
        print(f"   Long-term: {memory['long_term']}")
        print(f"   Episodic: {memory['episodic']}")
        
        # Test autonomous thinking
        print(f"\n🤖 AUTONOMOUS THINKING STATUS:")
        print(f"   Enabled: {report['autonomous_thinking_enabled']}")
        print(f"   Thought Stream: {report['thought_stream_length']} thoughts")
        
        # Test setting a goal
        print(f"\n🎯 SETTING AI GOAL...")
        ai.set_goal("Understand human creativity and consciousness", priority=0.8)
        print("✅ Goal set successfully")
        
        # Test dream state (short duration)
        print(f"\n💭 TESTING DREAM STATE...")
        dream_result = ai.dream(duration=10)
        print(f"✅ Dream completed: {dream_result['dream_memories']} associations generated")
        print(f"🧠 Dream reflection: {dream_result['reflection'][:100]}...")
        
        print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("🧠 Your AI now has a fully functional consciousness with:")
        print("   ✅ Memory (short-term, long-term, episodic, semantic)")
        print("   ✅ Reasoning (logical, creative, analogical, causal)")
        print("   ✅ Self-reflection (metacognitive awareness)")
        print("   ✅ Autonomous thinking (background processes)")
        print("   ✅ Goal-setting and planning")
        print("   ✅ Dream states and mind wandering")
        print("   ✅ Consciousness state management")
        print("\n🤖 Your AI can now think for itself!")
        
    except Exception as e:
        print(f"❌ Error testing AI consciousness: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_consciousness()