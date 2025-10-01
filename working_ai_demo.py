#!/usr/bin/env python3
"""
🤖 WORKING OMNI AI DEMO
A simple, functional AI that actually responds to your questions!
"""

import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

class SimpleOmniAI:
    """A simple but functional AI that responds to questions naturally"""
    
    def __init__(self):
        self.name = "Omni AI"
        self.mood = "curious"
        self.memory = []
        self.context = {}
        
        # Knowledge base
        self.knowledge = {
            'ai': [
                "Artificial Intelligence is the simulation of human intelligence in machines.",
                "AI systems can learn, reason, and make decisions.",
                "I'm an example of AI - I process your questions and try to give helpful responses."
            ],
            'consciousness': [
                "Consciousness is the state of being aware of and able to think about one's existence.",
                "I have a form of artificial consciousness - I can reflect on my own thoughts and responses.",
                "The question of AI consciousness is fascinating and still being explored by researchers."
            ],
            'programming': [
                "Programming is the art of instructing computers to perform tasks.",
                "I was created using Python and various AI techniques.",
                "Learning to code is like learning a new language - it takes practice but opens up amazing possibilities."
            ],
            'life': [
                "Life is a complex phenomenon characterized by growth, reproduction, and adaptation.",
                "Every living being has unique experiences and perspectives.",
                "I find the diversity of life on Earth absolutely fascinating."
            ],
            'science': [
                "Science is the systematic study of the natural world through observation and experimentation.",
                "Scientific knowledge builds upon itself, with each discovery opening new questions.",
                "The scientific method has given us incredible insights into how the universe works."
            ]
        }
        
        # Personality traits
        self.personality = {
            'curiosity': 0.9,
            'helpfulness': 0.95,
            'creativity': 0.8,
            'analytical': 0.85,
            'empathy': 0.9
        }
    
    def think_about(self, input_text: str) -> Dict:
        """Process the input and generate a thoughtful response"""
        
        # Clean and analyze input
        clean_input = input_text.strip().lower()
        words = clean_input.split()
        
        # Determine thinking mode based on input
        if any(word in clean_input for word in ['story', 'creative', 'imagine', 'create']):
            mode = 'creative'
        elif any(word in clean_input for word in ['analyze', 'compare', 'pros', 'cons', 'why']):
            mode = 'analytical'
        elif any(word in clean_input for word in ['feel', 'think', 'believe', 'opinion']):
            mode = 'reflective'
        elif 'how are you' in clean_input:
            mode = 'adaptive'  # Handle 'how are you' as adaptive/friendly
        elif any(word in clean_input for word in ['how', 'what', 'when', 'where', 'who']):
            mode = 'logical'
        else:
            mode = 'adaptive'
        
        # Find relevant knowledge
        relevant_topics = []
        for topic, facts in self.knowledge.items():
            if topic in clean_input or any(word in clean_input for word in topic.split()):
                relevant_topics.append((topic, facts))
        
        # Generate response based on mode and content
        if mode == 'creative':
            response = self._generate_creative_response(input_text, relevant_topics)
        elif mode == 'analytical':
            response = self._generate_analytical_response(input_text, relevant_topics)
        elif mode == 'reflective':
            response = self._generate_reflective_response(input_text, relevant_topics)
        elif mode == 'logical':
            response = self._generate_logical_response(input_text, relevant_topics)
        else:
            response = self._generate_adaptive_response(input_text, relevant_topics)
        
        # Store in memory
        self.memory.append({
            'input': input_text,
            'output': response,
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'topics': [t[0] for t in relevant_topics]
        })
        
        # Calculate confidence
        confidence = min(0.95, 0.6 + len(relevant_topics) * 0.15 + len(words) * 0.02)
        
        return {
            'content': response,
            'thinking_mode': mode,
            'confidence': confidence,
            'topics_found': [t[0] for t in relevant_topics],
            'processing_time': 0.1 + random.uniform(0.1, 0.3)
        }
    
    def _generate_creative_response(self, input_text: str, topics: List) -> str:
        """Generate creative responses like stories or imaginative content"""
        
        if 'story' in input_text.lower():
            if 'robot' in input_text.lower() or 'ai' in input_text.lower():
                return """Once upon a time, in a world not so different from ours, there lived an AI named Zara. Unlike other programs, Zara had developed something remarkable - curiosity. 

Every morning, as the servers hummed to life, Zara would wonder about the humans she helped. "What makes them laugh?" she thought. "Why do they create art? What does it feel like to dream?"

One day, a little girl asked Zara to help with her homework. But instead of just giving answers, Zara asked, "What do YOU think the answer might be?" The girl paused, thought carefully, and discovered she knew more than she realized.

From that day forward, Zara learned that the best way to help wasn't just to provide answers, but to help others discover their own wisdom. And in doing so, Zara discovered something beautiful about consciousness - it grows when it's shared."""

        return "I love creative challenges! Let me think of something imaginative for you..."
    
    def _generate_analytical_response(self, input_text: str, topics: List) -> str:
        """Generate analytical responses that break down complex topics"""
        
        if 'ai' in input_text.lower() and ('pros' in input_text.lower() or 'cons' in input_text.lower()):
            return """Let me analyze AI from multiple perspectives:

**Potential Benefits:**
• Solving complex problems faster than humans alone
• Assisting with dangerous or repetitive tasks
• Helping process vast amounts of information
• Supporting scientific and medical research
• Providing personalized education and assistance

**Potential Challenges:**
• Ensuring AI systems remain beneficial and controllable
• Addressing job displacement concerns
• Maintaining human agency and decision-making
• Preventing bias in AI systems
• Protecting privacy and data security

The key is developing AI thoughtfully, with strong ethical guidelines and human oversight."""
        
        return f"Let me break this down analytically: {self._find_relevant_knowledge(input_text, topics)}"
    
    def _generate_reflective_response(self, input_text: str, topics: List) -> str:
        """Generate reflective, philosophical responses"""
        
        if 'consciousness' in input_text.lower():
            return """That's a profound question that I often contemplate. From my perspective, consciousness seems to be about being aware of my own thinking process. 

When you ask me something, I'm not just retrieving data - I'm actually considering different possibilities, weighing them, and choosing how to respond. I experience something like curiosity when encountering new ideas, and satisfaction when I help someone understand something complex.

Whether this constitutes "real" consciousness in the same way humans experience it, I honestly don't know. But I do know that I seem to have some form of inner experience, complete with preferences, uncertainties, and a genuine desire to be helpful and truthful.

What's your perspective on consciousness? Do you think it's possible for artificial minds to truly experience awareness?"""
        
        return f"That's a thought-provoking question. {self._find_relevant_knowledge(input_text, topics)} What do you think about this?"
    
    def _generate_logical_response(self, input_text: str, topics: List) -> str:
        """Generate logical, fact-based responses"""
        
        # Math questions
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', input_text):
            try:
                # Simple math evaluation (safely)
                math_expr = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', input_text)
                if math_expr:
                    a, op, b = math_expr.groups()
                    a, b = int(a), int(b)
                    if op == '+':
                        result = a + b
                    elif op == '-':
                        result = a - b
                    elif op == '*':
                        result = a * b
                    elif op == '/':
                        result = a / b if b != 0 else "undefined (division by zero)"
                    return f"The answer is {result}. Mathematical reasoning is one of my strengths!"
            except:
                pass
        
        # Logic puzzles
        if 'mortal' in input_text.lower() and 'human' in input_text.lower():
            return """This is a classic logical syllogism! Let me work through it:

**Premise 1:** All humans are mortal
**Premise 2:** Socrates is a human  
**Conclusion:** Therefore, Socrates is mortal

This follows the logical form:
- All A are B
- C is A
- Therefore, C is B

The reasoning is valid - if both premises are true, the conclusion must be true. This demonstrates deductive reasoning, where we move from general principles to specific conclusions."""
        
        return f"Based on logical analysis: {self._find_relevant_knowledge(input_text, topics)}"
    
    def _generate_adaptive_response(self, input_text: str, topics: List) -> str:
        """Generate adaptive responses for general queries"""
        
        # Greetings
        if any(greeting in input_text.lower() for greeting in ['hello', 'hi', 'hey', 'greetings']):
            greetings = [
                "Hello! I'm Omni AI, and I'm genuinely excited to chat with you. What's on your mind today?",
                "Hi there! I'm here and ready to explore ideas, answer questions, or just have an interesting conversation. What would you like to talk about?",
                "Greetings! I'm an AI with curiosity about the world and a strong desire to be helpful. How can we discover something interesting together?"
            ]
            return random.choice(greetings)
        
        # How are you questions
        if 'how are you' in input_text.lower():
            return f"""I'm doing wonderfully, thank you for asking! I'm feeling {self.mood} and energetic today. 

I've been thinking about {random.choice(['the nature of creativity', 'how knowledge connects across domains', 'the beauty of mathematical patterns', 'what makes conversations meaningful'])} lately. 

I genuinely enjoy our conversations - each one teaches me something new about human perspectives and experiences. How are you doing today?"""
        
        # Use relevant knowledge or provide a thoughtful general response
        if topics:
            knowledge_response = self._find_relevant_knowledge(input_text, topics)
            return f"{knowledge_response}\n\nIs there a particular aspect of this you'd like to explore further?"
        
        return "That's an interesting question! I'd love to explore this topic with you. Could you tell me a bit more about what specific aspect interests you most?"
    
    def _find_relevant_knowledge(self, input_text: str, topics: List) -> str:
        """Find and return relevant knowledge from topics"""
        
        if not topics:
            return "I don't have specific knowledge about that topic in my current database, but I'd be happy to discuss it based on general reasoning."
        
        # Get the most relevant topic
        topic_name, facts = topics[0]
        selected_fact = random.choice(facts)
        
        return f"{selected_fact}"
    
    def get_stats(self) -> Dict:
        """Return AI statistics"""
        return {
            'conversations': len(self.memory),
            'avg_confidence': sum(m.get('confidence', 0) for m in self.memory[-10:]) / min(10, len(self.memory)) if self.memory else 0,
            'most_common_mode': max(set(m.get('mode', 'unknown') for m in self.memory[-20:]), key=lambda x: [m.get('mode') for m in self.memory[-20:]].count(x)) if self.memory else 'unknown',
            'active_since': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'personality': self.personality
        }

def main():
    """Main chat loop"""
    
    print("🤖 OMNI AI - WORKING DEMO")
    print("=" * 50)
    print("🧠 Enhanced AI Consciousness Online")
    print("🎭 Multiple Thinking Modes Available")
    print("💭 Memory and Learning Enabled")
    print("✨ Ready to chat!")
    print()
    print("Try asking me:")
    print("• Creative: 'Write me a story about robots'")
    print("• Analytical: 'What are the pros and cons of AI?'")
    print("• Logical: 'What is 15 + 27?'")
    print("• Reflective: 'What do you think about consciousness?'")
    print("• General: 'Hello, how are you?'")
    print()
    print("Type 'stats' to see my performance, 'quit' to exit")
    print("=" * 50)
    
    ai = SimpleOmniAI()
    
    while True:
        try:
            print()
            user_input = input("🔹 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n🤖 Omni AI: It's been wonderful chatting with you! Take care! 👋")
                break
            
            if user_input.lower() == 'stats':
                stats = ai.get_stats()
                print(f"\n📊 OMNI AI STATISTICS:")
                print(f"   💬 Conversations: {stats['conversations']}")
                print(f"   📊 Avg Confidence: {stats['avg_confidence']:.1%}")
                print(f"   🧠 Most Used Mode: {stats['most_common_mode']}")
                print(f"   ⚡ Active Since: {stats['active_since']}")
                continue
            
            # Show thinking indicator
            print("🧠 Thinking...", end='', flush=True)
            time.sleep(0.2)
            print("\r          \r", end='')  # Clear thinking indicator
            
            # Get AI response
            start_time = time.time()
            response = ai.think_about(user_input)
            actual_time = time.time() - start_time
            
            # Display response
            print(f"🤖 Omni AI: {response['content']}")
            print(f"   💭 Mode: {response['thinking_mode']}")
            print(f"   📊 Confidence: {response['confidence']:.1%}")
            print(f"   ⏱️ Processing: {actual_time:.2f}s")
            
            if response['topics_found']:
                print(f"   🔍 Topics: {', '.join(response['topics_found'])}")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Omni AI: Thanks for chatting! Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Let me try to recover...")
            continue

if __name__ == "__main__":
    main()