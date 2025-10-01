"""
🔸 SIMPLE NATURAL LANGUAGE INTERFACE DEMO
A quick demonstration of the natural language capabilities
"""

import asyncio
import sys
from pathlib import Path

# Add the source directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.omni_ai.neurosymbolic.symbolic_reasoner import SymbolicReasoner
from src.omni_ai.neurosymbolic.knowledge_graph import KnowledgeGraph
from src.omni_ai.neurosymbolic.neural_symbolic_bridge import NeuralSymbolicBridge, ReasoningMode
from src.omni_ai.neurosymbolic.nl_interface import NaturalLanguageInterface

async def run_demo():
    """Run a simple interactive demo"""
    print("🔸 NATURAL LANGUAGE INTERFACE DEMO")
    print("=" * 50)
    
    # Initialize system
    print("🚀 Initializing...")
    reasoner = SymbolicReasoner()
    knowledge_graph = KnowledgeGraph()
    bridge = NeuralSymbolicBridge(reasoner, knowledge_graph)
    interface = NaturalLanguageInterface(bridge)
    
    # Add some basic knowledge
    print("📚 Adding some basic knowledge...")
    await interface.process_query("Socrates is a human")
    await interface.process_query("Plato is a human")
    await interface.process_query("All humans are mortal")
    
    print("\n✅ Ready! Try these queries:")
    print("  • 'Is Socrates mortal?'")
    print("  • 'Who is mortal?'")
    print("  • 'Aristotle is a human'")
    print("  • Type 'quit' to exit\n")
    
    # Interactive loop
    while True:
        try:
            user_input = input("🔸 You: ").strip()
            
            if not user_input or user_input.lower() in ['quit', 'exit']:
                break
            
            print("🧠 Processing...")
            response = await interface.process_query(user_input)
            
            print(f"🤖 AI: {response.answer}")
            if response.confidence < 1.0:
                print(f"   📊 Confidence: {response.confidence:.1%}")
            
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    print("👋 Demo completed!")

if __name__ == "__main__":
    asyncio.run(run_demo())