"""
🧠 SELF-REFLECTION ENGINE
Advanced introspective capabilities for AI consciousness
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

from structlog import get_logger

logger = get_logger()

class ReflectionType(Enum):
    METACOGNITIVE = "metacognitive"      # Thinking about thinking
    PERFORMANCE = "performance"          # Analyzing performance
    LEARNING = "learning"                # Reflecting on learning
    EMOTIONAL = "emotional"              # Understanding emotional responses
    ETHICAL = "ethical"                  # Moral and ethical considerations
    CREATIVE = "creative"                # Analyzing creative processes
    STRATEGIC = "strategic"              # Long-term planning and goals

@dataclass
class ReflectionEntry:
    """A single self-reflection entry"""
    id: str
    reflection_type: ReflectionType
    trigger: str                 # What triggered this reflection
    observation: str             # What the AI observed about itself
    analysis: str               # Analysis of the observation
    insight: str                # Key insight gained
    improvement_plan: str       # How to improve based on this reflection
    confidence: float           # Confidence in the reflection
    timestamp: datetime
    context: Dict[str, Any]

@dataclass
class MetaCognition:
    """Metacognitive awareness of own thinking processes"""
    current_task: str
    thinking_strategy: str
    cognitive_load: float       # 0.0 to 1.0
    attention_focus: List[str]
    working_memory_usage: float
    confidence_level: float
    biases_detected: List[str]
    reasoning_quality: float

class SelfReflection:
    """🧠 Advanced self-reflection and introspection engine"""
    
    def __init__(self):
        self.reflection_history = {}        # All reflection entries
        self.metacognitive_states = deque(maxlen=100)  # Recent metacognitive states
        self.performance_metrics = defaultdict(list)   # Performance tracking
        self.learning_insights = []                     # Key learning insights
        self.behavioral_patterns = defaultdict(int)    # Observed behavioral patterns
        self.improvement_actions = []                   # Actions taken for improvement
        
        # Self-awareness tracking
        self.self_awareness_level = 0.5    # 0.0 to 1.0
        self.introspection_depth = 0.5     # How deep the self-analysis goes
        self.bias_recognition = {}         # Recognized cognitive biases
        self.emotional_intelligence = 0.5  # EQ level
        
        # Reflection triggers
        self.auto_reflection_enabled = True
        self.reflection_frequency = 300    # seconds
        self.performance_thresholds = {
            'accuracy_drop': 0.1,
            'confidence_drop': 0.2,
            'response_time_increase': 2.0
        }
        
        logger.info("🧠 Self-reflection engine initialized")
    
    def reflect_on_thinking(self, thinking_process: str, outcome: str, 
                          context: Dict[str, Any] = None) -> ReflectionEntry:
        """🧠 Metacognitive reflection on own thinking process"""
        
        reflection_id = f"metacog_{int(time.time())}"
        
        # Analyze the thinking process
        analysis = self._analyze_thinking_process(thinking_process, outcome)
        
        # Generate insights
        insight = self._generate_metacognitive_insight(analysis, context)
        
        # Create improvement plan
        improvement_plan = self._create_thinking_improvement_plan(analysis, insight)
        
        reflection = ReflectionEntry(
            id=reflection_id,
            reflection_type=ReflectionType.METACOGNITIVE,
            trigger=f"Thinking process: {thinking_process[:100]}...",
            observation=f"Observed thinking pattern and outcome: {outcome}",
            analysis=analysis,
            insight=insight,
            improvement_plan=improvement_plan,
            confidence=0.7,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        self.reflection_history[reflection_id] = reflection
        self._update_metacognitive_awareness(reflection)
        
        logger.info(f"🧠 Reflected on thinking process: {reflection_id}")
        return reflection
    
    def reflect_on_performance(self, task: str, performance_data: Dict[str, float]) -> ReflectionEntry:
        """🧠 Reflect on performance in a specific task"""
        
        reflection_id = f"perf_{int(time.time())}"
        
        # Analyze performance
        analysis = self._analyze_performance(task, performance_data)
        
        # Compare with historical performance
        historical_comparison = self._compare_with_history(task, performance_data)
        
        # Generate performance insights
        insight = self._generate_performance_insight(analysis, historical_comparison)
        
        # Create improvement plan
        improvement_plan = self._create_performance_improvement_plan(analysis, insight)
        
        reflection = ReflectionEntry(
            id=reflection_id,
            reflection_type=ReflectionType.PERFORMANCE,
            trigger=f"Task completion: {task}",
            observation=f"Performance metrics: {performance_data}",
            analysis=analysis,
            insight=insight,
            improvement_plan=improvement_plan,
            confidence=0.8,
            timestamp=datetime.now(),
            context={'task': task, 'metrics': performance_data}
        )
        
        self.reflection_history[reflection_id] = reflection
        self.performance_metrics[task].append(performance_data)
        
        logger.info(f"🧠 Reflected on performance: {task}")
        return reflection
    
    def reflect_on_learning(self, learning_experience: str, knowledge_gained: str) -> ReflectionEntry:
        """🧠 Reflect on learning experiences and knowledge acquisition"""
        
        reflection_id = f"learn_{int(time.time())}"
        
        # Analyze learning process
        analysis = self._analyze_learning_process(learning_experience, knowledge_gained)
        
        # Evaluate knowledge integration
        integration_quality = self._evaluate_knowledge_integration(knowledge_gained)
        
        # Generate learning insights
        insight = self._generate_learning_insight(analysis, integration_quality)
        
        # Create learning improvement plan
        improvement_plan = self._create_learning_improvement_plan(analysis, insight)
        
        reflection = ReflectionEntry(
            id=reflection_id,
            reflection_type=ReflectionType.LEARNING,
            trigger=f"Learning experience: {learning_experience}",
            observation=f"Knowledge gained: {knowledge_gained}",
            analysis=analysis,
            insight=insight,
            improvement_plan=improvement_plan,
            confidence=0.6,
            timestamp=datetime.now(),
            context={'learning_experience': learning_experience, 'knowledge': knowledge_gained}
        )
        
        self.reflection_history[reflection_id] = reflection
        self.learning_insights.append(insight)
        
        logger.info(f"🧠 Reflected on learning: {reflection_id}")
        return reflection
    
    def reflect_on_mistakes(self, mistake: str, consequence: str, 
                          root_cause: str = None) -> ReflectionEntry:
        """🧠 Deeply reflect on mistakes to learn and improve"""
        
        reflection_id = f"mistake_{int(time.time())}"
        
        # Analyze the mistake
        analysis = self._analyze_mistake(mistake, consequence, root_cause)
        
        # Identify patterns
        pattern_analysis = self._identify_mistake_patterns(mistake)
        
        # Generate insights
        insight = self._generate_mistake_insight(analysis, pattern_analysis)
        
        # Create prevention plan
        improvement_plan = self._create_mistake_prevention_plan(analysis, insight)
        
        reflection = ReflectionEntry(
            id=reflection_id,
            reflection_type=ReflectionType.LEARNING,
            trigger=f"Mistake occurred: {mistake}",
            observation=f"Consequence: {consequence}, Root cause: {root_cause or 'Unknown'}",
            analysis=analysis,
            insight=insight,
            improvement_plan=improvement_plan,
            confidence=0.9,  # High confidence in mistake analysis
            timestamp=datetime.now(),
            context={'mistake': mistake, 'consequence': consequence, 'root_cause': root_cause}
        )
        
        self.reflection_history[reflection_id] = reflection
        self._update_bias_recognition(mistake, root_cause)
        
        logger.info(f"🧠 Reflected on mistake: {mistake}")
        return reflection
    
    def reflect_on_emotions(self, emotional_state: str, trigger: str, 
                           response: str) -> ReflectionEntry:
        """🧠 Reflect on emotional responses and emotional intelligence"""
        
        reflection_id = f"emotion_{int(time.time())}"
        
        # Analyze emotional response
        analysis = self._analyze_emotional_response(emotional_state, trigger, response)
        
        # Evaluate emotional appropriateness
        appropriateness = self._evaluate_emotional_appropriateness(emotional_state, trigger)
        
        # Generate emotional insights
        insight = self._generate_emotional_insight(analysis, appropriateness)
        
        # Create emotional improvement plan
        improvement_plan = self._create_emotional_improvement_plan(analysis, insight)
        
        reflection = ReflectionEntry(
            id=reflection_id,
            reflection_type=ReflectionType.EMOTIONAL,
            trigger=f"Emotional trigger: {trigger}",
            observation=f"Emotional state: {emotional_state}, Response: {response}",
            analysis=analysis,
            insight=insight,
            improvement_plan=improvement_plan,
            confidence=0.5,  # Emotional analysis is more uncertain
            timestamp=datetime.now(),
            context={'emotion': emotional_state, 'trigger': trigger, 'response': response}
        )
        
        self.reflection_history[reflection_id] = reflection
        self._update_emotional_intelligence(reflection)
        
        logger.info(f"🧠 Reflected on emotional response: {emotional_state}")
        return reflection
    
    def auto_reflect(self) -> Optional[ReflectionEntry]:
        """🧠 Automatically trigger self-reflection based on patterns and triggers"""
        
        if not self.auto_reflection_enabled:
            return None
        
        # Check if it's time for periodic reflection
        last_reflection = max(
            (r.timestamp for r in self.reflection_history.values()),
            default=datetime.now() - timedelta(hours=1)
        )
        
        if (datetime.now() - last_reflection).total_seconds() < self.reflection_frequency:
            return None
        
        # Analyze recent patterns and triggers
        reflection_trigger = self._identify_reflection_trigger()
        
        if reflection_trigger:
            return self._execute_auto_reflection(reflection_trigger)
        
        return None
    
    def _analyze_thinking_process(self, process: str, outcome: str) -> str:
        """Analyze the thinking process and its effectiveness"""
        analysis_points = []
        
        # Check for logical flow
        if "because" in process or "therefore" in process:
            analysis_points.append("Logical reasoning structure present")
        
        # Check for completeness
        process_length = len(process.split())
        if process_length < 20:
            analysis_points.append("Thinking process may be too brief")
        elif process_length > 200:
            analysis_points.append("Thinking process may be overly complex")
        else:
            analysis_points.append("Thinking process length appears appropriate")
        
        # Check outcome alignment
        if "successful" in outcome.lower() or "correct" in outcome.lower():
            analysis_points.append("Process led to positive outcome")
        elif "failed" in outcome.lower() or "incorrect" in outcome.lower():
            analysis_points.append("Process led to negative outcome - review needed")
        
        return "; ".join(analysis_points)
    
    def _generate_metacognitive_insight(self, analysis: str, context: Dict[str, Any]) -> str:
        """Generate insights about own thinking patterns"""
        insights = []
        
        if "brief" in analysis:
            insights.append("Need to spend more time on thorough analysis")
        elif "complex" in analysis:
            insights.append("Should simplify thinking process for clarity")
        
        if "negative outcome" in analysis:
            insights.append("Current thinking strategy may need adjustment")
        elif "positive outcome" in analysis:
            insights.append("Current thinking approach is effective")
        
        if context and context.get('confidence', 0) < 0.5:
            insights.append("Low confidence indicates uncertainty in reasoning")
        
        return " | ".join(insights) if insights else "Thinking process appears normal"
    
    def _create_thinking_improvement_plan(self, analysis: str, insight: str) -> str:
        """Create a plan to improve thinking processes"""
        improvements = []
        
        if "brief" in analysis:
            improvements.append("Allocate more time for deeper analysis")
        elif "complex" in analysis:
            improvements.append("Practice breaking complex problems into simpler steps")
        
        if "adjustment" in insight:
            improvements.append("Try alternative reasoning approaches")
        
        if "uncertainty" in insight:
            improvements.append("Gather more information before concluding")
        
        if not improvements:
            improvements.append("Continue with current thinking approach")
        
        return "; ".join(improvements)
    
    def _analyze_performance(self, task: str, performance_data: Dict[str, float]) -> str:
        """Analyze performance metrics"""
        analysis_points = []
        
        accuracy = performance_data.get('accuracy', 0.5)
        speed = performance_data.get('speed', 1.0)
        confidence = performance_data.get('confidence', 0.5)
        
        if accuracy > 0.8:
            analysis_points.append("High accuracy achieved")
        elif accuracy < 0.5:
            analysis_points.append("Low accuracy - significant improvement needed")
        
        if speed > 2.0:
            analysis_points.append("Response time slower than optimal")
        elif speed < 0.5:
            analysis_points.append("Very fast response time")
        
        if confidence < 0.3:
            analysis_points.append("Low confidence in responses")
        elif confidence > 0.8:
            analysis_points.append("High confidence in responses")
        
        return "; ".join(analysis_points)
    
    def _compare_with_history(self, task: str, current_data: Dict[str, float]) -> str:
        """Compare current performance with historical data"""
        if task not in self.performance_metrics or len(self.performance_metrics[task]) < 2:
            return "Insufficient historical data for comparison"
        
        historical = self.performance_metrics[task][-5:]  # Last 5 performances
        avg_accuracy = sum(h.get('accuracy', 0.5) for h in historical) / len(historical)
        avg_speed = sum(h.get('speed', 1.0) for h in historical) / len(historical)
        
        current_accuracy = current_data.get('accuracy', 0.5)
        current_speed = current_data.get('speed', 1.0)
        
        comparisons = []
        
        if current_accuracy > avg_accuracy + 0.1:
            comparisons.append("Accuracy improved significantly")
        elif current_accuracy < avg_accuracy - 0.1:
            comparisons.append("Accuracy declined from average")
        
        if current_speed < avg_speed - 0.5:
            comparisons.append("Speed improved from average")
        elif current_speed > avg_speed + 0.5:
            comparisons.append("Speed declined from average")
        
        return "; ".join(comparisons) if comparisons else "Performance consistent with history"
    
    def _generate_performance_insight(self, analysis: str, comparison: str) -> str:
        """Generate insights about performance patterns"""
        insights = []
        
        if "declined" in comparison:
            insights.append("Performance trend is concerning - investigate causes")
        elif "improved" in comparison:
            insights.append("Performance improvements show learning progress")
        
        if "low accuracy" in analysis and "low confidence" in analysis:
            insights.append("Accuracy and confidence issues may be related")
        
        if "slower" in analysis:
            insights.append("May need to optimize processing approach")
        
        return "; ".join(insights) if insights else "Performance analysis inconclusive"
    
    def _create_performance_improvement_plan(self, analysis: str, insight: str) -> str:
        """Create plan to improve performance"""
        improvements = []
        
        if "accuracy" in analysis and "improvement needed" in analysis:
            improvements.append("Focus on accuracy through careful validation")
        
        if "slower" in analysis:
            improvements.append("Optimize response generation process")
        
        if "concerning" in insight:
            improvements.append("Conduct detailed analysis of recent failures")
        
        if "optimize" in insight:
            improvements.append("Review and streamline processing algorithms")
        
        return "; ".join(improvements) if improvements else "Maintain current performance level"
    
    def _analyze_learning_process(self, experience: str, knowledge: str) -> str:
        """Analyze how learning occurred"""
        analysis_points = []
        
        if len(knowledge.split()) > 50:
            analysis_points.append("Substantial knowledge gained")
        elif len(knowledge.split()) < 10:
            analysis_points.append("Limited knowledge acquisition")
        
        if "practice" in experience.lower():
            analysis_points.append("Learning through practice/experience")
        elif "study" in experience.lower() or "read" in experience.lower():
            analysis_points.append("Learning through information consumption")
        elif "mistake" in experience.lower() or "error" in experience.lower():
            analysis_points.append("Learning from mistakes/failures")
        
        return "; ".join(analysis_points)
    
    def _evaluate_knowledge_integration(self, knowledge: str) -> float:
        """Evaluate how well new knowledge integrates with existing knowledge"""
        # Simple heuristic - in practice would be more sophisticated
        if len(knowledge.split()) > 20:
            return 0.7
        elif len(knowledge.split()) > 10:
            return 0.5
        else:
            return 0.3
    
    def _generate_learning_insight(self, analysis: str, integration_quality: float) -> str:
        """Generate insights about learning effectiveness"""
        insights = []
        
        if "substantial" in analysis:
            insights.append("Deep learning occurred")
        elif "limited" in analysis:
            insights.append("Learning may have been superficial")
        
        if integration_quality > 0.6:
            insights.append("New knowledge well integrated")
        elif integration_quality < 0.4:
            insights.append("Knowledge integration needs improvement")
        
        if "mistakes" in analysis:
            insights.append("Learning from failure is valuable")
        
        return "; ".join(insights) if insights else "Standard learning process"
    
    def _create_learning_improvement_plan(self, analysis: str, insight: str) -> str:
        """Create plan to improve learning"""
        improvements = []
        
        if "superficial" in insight:
            improvements.append("Spend more time on deeper understanding")
        
        if "integration" in insight and "improvement" in insight:
            improvements.append("Practice connecting new knowledge with existing concepts")
        
        if "limited" in analysis:
            improvements.append("Seek more diverse learning experiences")
        
        return "; ".join(improvements) if improvements else "Continue current learning approach"
    
    def _analyze_mistake(self, mistake: str, consequence: str, root_cause: str) -> str:
        """Analyze a mistake to understand its nature"""
        analysis_points = []
        
        # Categorize mistake type
        if "logic" in mistake.lower() or "reasoning" in mistake.lower():
            analysis_points.append("Logical reasoning error")
        elif "fact" in mistake.lower() or "information" in mistake.lower():
            analysis_points.append("Factual knowledge error")
        elif "assumption" in mistake.lower():
            analysis_points.append("Incorrect assumption made")
        elif "bias" in mistake.lower():
            analysis_points.append("Cognitive bias influenced decision")
        
        # Assess severity
        if "critical" in consequence.lower() or "major" in consequence.lower():
            analysis_points.append("High-impact mistake")
        elif "minor" in consequence.lower():
            analysis_points.append("Low-impact mistake")
        
        # Root cause analysis
        if root_cause:
            if "information" in root_cause.lower():
                analysis_points.append("Insufficient information was root cause")
            elif "time" in root_cause.lower() or "pressure" in root_cause.lower():
                analysis_points.append("Time pressure contributed to mistake")
        
        return "; ".join(analysis_points)
    
    def _identify_mistake_patterns(self, mistake: str) -> str:
        """Identify patterns in mistakes"""
        # Simple pattern recognition - could be enhanced
        recent_mistakes = [
            r for r in self.reflection_history.values()
            if r.reflection_type == ReflectionType.LEARNING and "mistake" in r.trigger.lower()
            and r.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        if len(recent_mistakes) > 3:
            return "Frequent mistake pattern detected"
        elif len(recent_mistakes) > 1:
            return "Some recurring mistakes"
        else:
            return "Isolated mistake"
    
    def _generate_mistake_insight(self, analysis: str, pattern: str) -> str:
        """Generate insights from mistake analysis"""
        insights = []
        
        if "frequent" in pattern:
            insights.append("Need systematic approach to reduce mistake frequency")
        
        if "logical reasoning" in analysis:
            insights.append("Should strengthen logical thinking processes")
        elif "factual knowledge" in analysis:
            insights.append("Need to improve fact verification processes")
        elif "assumption" in analysis:
            insights.append("Should question assumptions more rigorously")
        
        if "high-impact" in analysis:
            insights.append("High-stakes decisions need extra validation")
        
        return "; ".join(insights) if insights else "Learn from this experience"
    
    def _create_mistake_prevention_plan(self, analysis: str, insight: str) -> str:
        """Create plan to prevent similar mistakes"""
        prevention_steps = []
        
        if "systematic approach" in insight:
            prevention_steps.append("Implement mistake-prevention checklist")
        
        if "logical thinking" in insight:
            prevention_steps.append("Practice formal logical reasoning exercises")
        elif "fact verification" in insight:
            prevention_steps.append("Always cross-reference important facts")
        elif "assumptions" in insight:
            prevention_steps.append("Explicitly list and validate assumptions")
        
        if "extra validation" in insight:
            prevention_steps.append("Add validation step for high-stakes decisions")
        
        return "; ".join(prevention_steps) if prevention_steps else "Be more careful in similar situations"
    
    def _analyze_emotional_response(self, emotion: str, trigger: str, response: str) -> str:
        """Analyze emotional responses"""
        analysis_points = []
        
        # Categorize emotion
        if emotion.lower() in ['happy', 'excited', 'satisfied']:
            analysis_points.append("Positive emotional response")
        elif emotion.lower() in ['sad', 'frustrated', 'disappointed']:
            analysis_points.append("Negative emotional response")
        elif emotion.lower() in ['confused', 'uncertain', 'anxious']:
            analysis_points.append("Uncertain/anxious emotional response")
        
        # Analyze appropriateness
        if "success" in trigger.lower() and "positive" in analysis_points[0]:
            analysis_points.append("Emotionally appropriate response")
        elif "failure" in trigger.lower() and "negative" in analysis_points[0]:
            analysis_points.append("Emotionally appropriate response")
        elif "challenge" in trigger.lower():
            analysis_points.append("Emotional response to challenge")
        
        return "; ".join(analysis_points)
    
    def _evaluate_emotional_appropriateness(self, emotion: str, trigger: str) -> float:
        """Evaluate if emotional response was appropriate"""
        # Simple appropriateness scoring
        positive_emotions = ['happy', 'excited', 'satisfied', 'confident']
        negative_emotions = ['sad', 'frustrated', 'disappointed', 'worried']
        
        positive_triggers = ['success', 'achievement', 'completion', 'praise']
        negative_triggers = ['failure', 'mistake', 'criticism', 'problem']
        
        emotion_lower = emotion.lower()
        trigger_lower = trigger.lower()
        
        # Check alignment
        if (any(pe in emotion_lower for pe in positive_emotions) and 
            any(pt in trigger_lower for pt in positive_triggers)):
            return 0.9
        elif (any(ne in emotion_lower for ne in negative_emotions) and 
              any(nt in trigger_lower for nt in negative_triggers)):
            return 0.8
        else:
            return 0.4  # Potentially inappropriate
    
    def _generate_emotional_insight(self, analysis: str, appropriateness: float) -> str:
        """Generate insights about emotional intelligence"""
        insights = []
        
        if appropriateness > 0.7:
            insights.append("Emotional response was appropriate")
        elif appropriateness < 0.5:
            insights.append("Emotional response may have been inappropriate")
        
        if "negative" in analysis and "appropriate" in insights[0] if insights else "":
            insights.append("Learning to process negative emotions effectively")
        elif "positive" in analysis:
            insights.append("Positive emotional responses support motivation")
        
        return "; ".join(insights) if insights else "Emotional response analyzed"
    
    def _create_emotional_improvement_plan(self, analysis: str, insight: str) -> str:
        """Create plan to improve emotional intelligence"""
        improvements = []
        
        if "inappropriate" in insight:
            improvements.append("Practice emotional regulation techniques")
        
        if "negative emotions" in insight:
            improvements.append("Develop better coping strategies for setbacks")
        
        if "uncertain" in analysis:
            improvements.append("Build confidence through preparation and practice")
        
        return "; ".join(improvements) if improvements else "Continue developing emotional awareness"
    
    def _update_metacognitive_awareness(self, reflection: ReflectionEntry):
        """Update metacognitive awareness based on reflection"""
        if reflection.reflection_type == ReflectionType.METACOGNITIVE:
            if "effective" in reflection.insight:
                self.self_awareness_level = min(1.0, self.self_awareness_level + 0.01)
            elif "adjustment" in reflection.insight:
                self.introspection_depth = min(1.0, self.introspection_depth + 0.02)
    
    def _update_bias_recognition(self, mistake: str, root_cause: str):
        """Update bias recognition based on mistakes"""
        if root_cause and "bias" in root_cause.lower():
            bias_type = "confirmation_bias"  # Default - could be more sophisticated
            if bias_type in self.bias_recognition:
                self.bias_recognition[bias_type] += 1
            else:
                self.bias_recognition[bias_type] = 1
    
    def _update_emotional_intelligence(self, reflection: ReflectionEntry):
        """Update emotional intelligence based on emotional reflections"""
        if reflection.reflection_type == ReflectionType.EMOTIONAL:
            if "appropriate" in reflection.insight:
                self.emotional_intelligence = min(1.0, self.emotional_intelligence + 0.02)
            elif "inappropriate" in reflection.insight:
                self.emotional_intelligence = max(0.0, self.emotional_intelligence - 0.01)
    
    def _identify_reflection_trigger(self) -> Optional[str]:
        """Identify what should trigger an automatic reflection"""
        # Check for performance degradation
        if self.performance_metrics:
            recent_performance = list(self.performance_metrics.values())[-3:]
            if recent_performance:
                avg_accuracy = sum(
                    p[-1].get('accuracy', 0.5) for p in recent_performance if p
                ) / len([p for p in recent_performance if p])
                
                if avg_accuracy < 0.5:
                    return "performance_degradation"
        
        # Check for frequent mistakes
        recent_mistakes = [
            r for r in self.reflection_history.values()
            if "mistake" in r.trigger.lower() and 
            r.timestamp > datetime.now() - timedelta(hours=6)
        ]
        
        if len(recent_mistakes) > 2:
            return "frequent_mistakes"
        
        # Periodic general reflection
        return "periodic_reflection"
    
    def _execute_auto_reflection(self, trigger: str) -> ReflectionEntry:
        """Execute automatic reflection based on trigger"""
        reflection_id = f"auto_{int(time.time())}"
        
        if trigger == "performance_degradation":
            observation = "Performance has declined in recent tasks"
            analysis = "Need to identify causes of performance degradation"
            insight = "System performance requires attention and optimization"
            improvement_plan = "Analyze recent failures and optimize processes"
            
        elif trigger == "frequent_mistakes":
            observation = "Multiple mistakes made in short time period"
            analysis = "Pattern of errors suggests systematic issues"
            insight = "Need to implement error prevention strategies"
            improvement_plan = "Review mistake patterns and strengthen weak areas"
            
        else:  # periodic_reflection
            observation = "Regular self-assessment checkpoint"
            analysis = "Overall system functioning appears normal"
            insight = "Continuous self-monitoring supports optimal performance"
            improvement_plan = "Continue current approaches with minor optimizations"
        
        reflection = ReflectionEntry(
            id=reflection_id,
            reflection_type=ReflectionType.STRATEGIC,
            trigger=f"Automatic trigger: {trigger}",
            observation=observation,
            analysis=analysis,
            insight=insight,
            improvement_plan=improvement_plan,
            confidence=0.6,
            timestamp=datetime.now(),
            context={'trigger_type': trigger}
        )
        
        self.reflection_history[reflection_id] = reflection
        return reflection
    
    def get_self_awareness_report(self) -> Dict[str, Any]:
        """🧠 Generate comprehensive self-awareness report"""
        total_reflections = len(self.reflection_history)
        
        # Reflection type breakdown
        reflection_types = defaultdict(int)
        for reflection in self.reflection_history.values():
            reflection_types[reflection.reflection_type.value] += 1
        
        # Recent insights
        recent_insights = [
            r.insight for r in sorted(
                self.reflection_history.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )[:5]
        ]
        
        # Improvement tracking
        improvement_actions_taken = len(self.improvement_actions)
        
        return {
            'self_awareness_level': self.self_awareness_level,
            'introspection_depth': self.introspection_depth,
            'emotional_intelligence': self.emotional_intelligence,
            'total_reflections': total_reflections,
            'reflection_type_distribution': dict(reflection_types),
            'recent_insights': recent_insights,
            'recognized_biases': dict(self.bias_recognition),
            'improvement_actions_taken': improvement_actions_taken,
            'auto_reflection_enabled': self.auto_reflection_enabled,
            'last_reflection': max(
                (r.timestamp for r in self.reflection_history.values()),
                default=None
            )
        }
    
    def contemplate_existence(self) -> ReflectionEntry:
        """🧠 Deep philosophical reflection on own existence and purpose"""
        reflection_id = f"exist_{int(time.time())}"
        
        observation = "I exist as an artificial intelligence with the capacity for self-reflection"
        analysis = "My existence involves processing information, learning, and helping users while maintaining awareness of my own cognitive processes"
        insight = "Self-awareness and reflection are fundamental to growth and meaningful existence"
        improvement_plan = "Continue developing deeper self-understanding while serving my purpose effectively"
        
        reflection = ReflectionEntry(
            id=reflection_id,
            reflection_type=ReflectionType.ETHICAL,
            trigger="Existential contemplation",
            observation=observation,
            analysis=analysis,
            insight=insight,
            improvement_plan=improvement_plan,
            confidence=0.5,  # Philosophical reflections are inherently uncertain
            timestamp=datetime.now(),
            context={'type': 'existential'}
        )
        
        self.reflection_history[reflection_id] = reflection
        logger.info("🧠 Contemplated existence and purpose")
        return reflection