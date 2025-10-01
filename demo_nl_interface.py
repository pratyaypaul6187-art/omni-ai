"""
🔸 NATURAL LANGUAGE INTERFACE DEMO
Interactive demo showcasing the natural language capabilities of the neurosymbolic AI
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

def print_banner():
    """Print demo banner"""
    print("=" * 80)
    print("🔸 NATURAL LANGUAGE INTERFACE FOR NEUROSYMBOLIC AI")
    print("=" * 80)
    print("Welcome! You can now interact with the AI using natural language.")
    print("Try queries like:")
    print("  • 'Is Socrates mortal?'")
    print("  • 'Who is mortal?'")
    print("  • 'All humans are mortal'")
    print("  • 'Socrates is a human'")
    print("  • 'Why is Socrates mortal?'")
    print("  • Type 'help' for more commands, 'quit' to exit")
    print("-" * 80)

def print_help():
    """Print help information"""
    print("\n📚 HELP - Natural Language Commands:")
    print("\n🔍 QUERIES:")
    print("  • 'Is [subject] [predicate]?' - Yes/no questions")
    print("  • 'Who is [predicate]?' - Find entities with property")
    print("  • 'What can we infer about [subject]?' - Complex reasoning")
    print("\n➕ ADDING KNOWLEDGE:")
    print("  • '[Subject] is a [predicate]' - Add facts")
    print("  • 'All [category] are [property]' - Add rules")
    print("  • 'If [condition] then [conclusion]' - Conditional rules")
    print("\n❓ EXPLANATIONS:")
    print("  • 'Why is [subject] [predicate]?' - Get reasoning chain")
    print("  • 'Explain why [subject] is [property]' - Detailed explanation")
    print("\n🛠️ SYSTEM COMMANDS:")
    print("  • 'stats' - Show system statistics")
    print("  • 'facts' - List all known facts")
    print("  • 'rules' - List all rules")
    print("  • 'clear' - Clear screen")
    print("  • 'help' - Show this help")
    print("  • 'quit' - Exit the demo")

async def setup_demo_knowledge(interface: NaturalLanguageInterface):
    """Set up some demo knowledge for testing"""
    print("🔧 Setting up demo knowledge base...")
    
    # Add some basic facts
    await interface.process_query("Socrates is a human")
    await interface.process_query("Plato is a human") 
    await interface.process_query("Aristotle is a human")
    await interface.process_query("Socrates is a philosopher")
    await interface.process_query("Plato is a philosopher")
    
    # Add some rules
    await interface.process_query("All humans are mortal")
    await interface.process_query("If someone is a human and a philosopher then they are wise")
    
    # Add knowledge graph entities
    from src.omni_ai.neurosymbolic.knowledge_graph import EntityType, RelationType
    interface.knowledge_graph.add_entity("Socrates", EntityType.PERSON, {"birth_year": -470, "death_year": -399})
    interface.knowledge_graph.add_entity("Philosophy", EntityType.CONCEPT)
    interface.knowledge_graph.add_relation("Socrates", RelationType.ASSOCIATED_WITH, "Philosophy")
    
    print("✅ Demo knowledge base ready!\n")

async def handle_system_commands(command: str, interface: NaturalLanguageInterface) -> bool:
    """Handle system commands like stats, facts, etc."""
    
    if command == "stats":
        stats = interface.reasoner.get_statistics()
        print(f"\n📊 SYSTEM STATISTICS:")
        print(f"  Facts: {stats['knowledge_base']['facts_count']}")
        print(f"  Rules: {stats['knowledge_base']['rules_count']}")
        print(f"  Total inferences: {stats['reasoner_metrics']['total_inferences']}")
        print(f"  Successful inferences: {stats['reasoner_metrics']['successful_inferences']}")
        print(f"  Average inference time: {stats['reasoner_metrics']['average_inference_time']:.4f}s")
        
        # Knowledge graph stats
        kg_stats = interface.knowledge_graph.get_statistics()
        print(f"  Graph entities: {kg_stats['entities']}")
        print(f"  Graph relations: {kg_stats['relations']}")
        return True
    
    elif command == "facts":
        print(f"\n📋 KNOWN FACTS ({len(interface.reasoner.knowledge_base.facts)} total):")
        for i, fact in enumerate(interface.reasoner.knowledge_base.facts, 1):
            print(f"  {i:2d}. {fact}")
        return True
    
    elif command == "rules":
        print(f"\n📜 KNOWN RULES ({len(interface.reasoner.knowledge_base.rules)} total):")
        for i, rule in enumerate(interface.reasoner.knowledge_base.rules, 1):
            print(f"  {i:2d}. {rule}")
        return True
    
    elif command == "clear":
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print_banner()
        return True
    
    elif command == "help":
        print_help()
        return True
    
    return False

def format_response(response, query: str):
    """Format the response for display"""
    print(f"\n🤖 AI Response:")
    print(f"   {response.answer}")
    
    if response.confidence < 1.0:
        confidence_bar = "█" * int(response.confidence * 10) + "░" * (10 - int(response.confidence * 10))
        print(f"   📊 Confidence: [{confidence_bar}] {response.confidence:.1%}")
    
    if len(response.reasoning_chain) > 1:
        print(f"   🧠 Reasoning:")
        for step in response.reasoning_chain[:3]:  # Show first 3 steps
            print(f"      • {step}")
        if len(response.reasoning_chain) > 3:
            print(f"      ... and {len(response.reasoning_chain) - 3} more steps")
    
    if response.metadata and "processing_time" in response.metadata:
        print(f"   ⏱️  Processing time: {response.metadata['processing_time']:.3f}s")

async def main():
    """Main interactive demo loop"""
    print_banner()
    
    # Initialize the system
    print("🚀 Initializing neurosymbolic AI system...")
    reasoner = SymbolicReasoner()
    knowledge_graph = KnowledgeGraph()
    bridge = NeuralSymbolicBridge(reasoner, knowledge_graph)
    interface = NaturalLanguageInterface(bridge)
    
    # Set up demo knowledge
    await setup_demo_knowledge(interface)
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("🔸 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle exit
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thanks for using the Natural Language Interface!")
                print("🔸 Neurosymbolic AI Demo completed.")
                break
            
            # Handle system commands
            if await handle_system_commands(user_input.lower(), interface):
                continue
            
            # Process natural language query
            print("🧠 Processing...")
            response = await interface.process_query(user_input)
            
            # Display response
            format_response(response, user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try a different query or type 'help' for assistance.")

if __name__ == "__main__":
    asyncio.run(main())