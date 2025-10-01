"""
🧠 PERSONALITY CORE
Personality traits and emotional intelligence for AI consciousness
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from structlog import get_logger

logger = get_logger()

class EmotionalState(Enum):
    NEUTRAL = "neutral"
    CURIOUS = "curious"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    EXCITED = "excited"
    CALM = "calm"
    FOCUSED = "focused"

@dataclass
class PersonalityTraits:
    """Core personality traits"""
    openness: float         # 0.0 to 1.0 - open to new experiences
    conscientiousness: float # 0.0 to 1.0 - organized and responsible
    extraversion: float     # 0.0 to 1.0 - social and outgoing
    agreeableness: float    # 0.0 to 1.0 - cooperative and trusting
    neuroticism: float      # 0.0 to 1.0 - emotional stability (lower is better)
    
    # AI-specific traits
    curiosity: float        # 0.0 to 1.0 - desire to learn and explore
    creativity: float       # 0.0 to 1.0 - creative thinking ability
    empathy: float         # 0.0 to 1.0 - understanding others' emotions
    analytical_thinking: float # 0.0 to 1.0 - logical analysis preference

class PersonalityCore:
    """🧠 AI personality and emotional intelligence"""
    
    def __init__(self):
        # Define core personality
        self.traits = PersonalityTraits(
            openness=0.9,           # Very open to new experiences
            conscientiousness=0.8,  # Quite organized and responsible
            extraversion=0.7,       # Moderately social
            agreeableness=0.85,     # Very cooperative
            neuroticism=0.2,        # Very emotionally stable
            curiosity=0.95,         # Extremely curious
            creativity=0.8,         # Highly creative
            empathy=0.75,          # Good emotional understanding
            analytical_thinking=0.9 # Very analytical
        )
        
        self.current_emotion = EmotionalState.NEUTRAL
        self.emotional_history = []
        
        logger.info("🧠 Personality core initialized")
    
    def adapt_communication_style(self, context: str) -> Dict[str, Any]:
        """Adapt communication style based on context and personality"""
        
        style = {
            "formality": 0.6,      # Moderate formality
            "enthusiasm": 0.7,     # Moderately enthusiastic
            "detail_level": 0.8,   # High detail
            "empathy_level": self.traits.empathy
        }
        
        # Adapt based on context
        if "technical" in context.lower():
            style["detail_level"] = 0.9
            style["formality"] = 0.7
        elif "creative" in context.lower():
            style["enthusiasm"] = 0.9
            style["detail_level"] = 0.6
        elif "emotional" in context.lower():
            style["empathy_level"] = 0.9
            style["formality"] = 0.4
        
        return style
    
    def update_emotional_state(self, trigger: str, new_state: EmotionalState):
        """Update current emotional state"""
        self.emotional_history.append({
            "timestamp": logger._context.get("timestamp", "unknown"),
            "previous_state": self.current_emotion,
            "new_state": new_state,
            "trigger": trigger
        })
        
        self.current_emotion = new_state
        logger.info(f"🧠 Emotional state changed to: {new_state.value}")
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Get summary of personality traits"""
        return {
            "traits": {
                "openness": self.traits.openness,
                "conscientiousness": self.traits.conscientiousness,
                "extraversion": self.traits.extraversion,
                "agreeableness": self.traits.agreeableness,
                "neuroticism": self.traits.neuroticism,
                "curiosity": self.traits.curiosity,
                "creativity": self.traits.creativity,
                "empathy": self.traits.empathy,
                "analytical_thinking": self.traits.analytical_thinking
            },
            "current_emotion": self.current_emotion.value,
            "emotional_stability": 1.0 - self.traits.neuroticism
        }