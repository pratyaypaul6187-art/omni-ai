"""
🧠 ARTIFICIAL NEURONS
Neural network system that simulates biological brain neurons
"""

import numpy as np
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import json

from structlog import get_logger

logger = get_logger()

class NeuronType(Enum):
    SENSORY = "sensory"         # Input neurons
    INTERNEURON = "interneuron" # Processing neurons
    MOTOR = "motor"             # Output neurons
    MEMORY = "memory"           # Memory storage neurons
    ATTENTION = "attention"     # Attention/focus neurons
    EMOTIONAL = "emotional"     # Emotional response neurons

class ActivationFunction(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX = "softmax"

@dataclass
class NeuronState:
    """Current state of a neuron"""
    activation: float           # Current activation level (0.0 to 1.0)
    threshold: float           # Firing threshold
    refractory_period: float   # Time before neuron can fire again
    last_fired: float          # Last time neuron fired
    fatigue_level: float       # Neuron fatigue (0.0 to 1.0)
    plasticity: float          # Learning rate/adaptability

@dataclass
class Synapse:
    """Connection between neurons"""
    pre_neuron_id: str         # Source neuron
    post_neuron_id: str        # Target neuron
    weight: float              # Connection strength (-1.0 to 1.0)
    delay: float               # Signal transmission delay (ms)
    plasticity: float          # How easily this connection changes
    last_active: float         # Last time synapse was used

class ArtificialNeuron:
    """🧠 Single artificial neuron that mimics biological neurons"""
    
    def __init__(self, neuron_id: str, neuron_type: NeuronType, 
                 activation_func: ActivationFunction = ActivationFunction.SIGMOID):
        self.id = neuron_id
        self.type = neuron_type
        self.activation_function = activation_func
        
        # Neuron parameters
        self.state = NeuronState(
            activation=0.0,
            threshold=0.5,
            refractory_period=10.0,  # ms
            last_fired=0.0,
            fatigue_level=0.0,
            plasticity=0.1
        )
        
        # Connections
        self.incoming_synapses = {}  # Dict[synapse_id, Synapse]
        self.outgoing_synapses = {}  # Dict[synapse_id, Synapse]
        
        # Neural activity history
        self.spike_history = deque(maxlen=1000)  # Recent firing times
        self.activation_history = deque(maxlen=100)  # Recent activation levels
        
        # Neuron-specific properties
        self.bias = 0.0
        self.learning_rate = 0.01
        self.decay_rate = 0.95
        
        logger.debug(f"🧠 Created {neuron_type.value} neuron: {neuron_id}")
    
    def receive_input(self, input_strength: float, source_neuron_id: str = None) -> float:
        """Receive input from another neuron or external source"""
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Check if neuron is in refractory period
        if (current_time - self.state.last_fired) < self.state.refractory_period:
            return 0.0  # Can't receive input during refractory period
        
        # Add input to current activation (with bias)
        self.state.activation += input_strength + self.bias
        
        # Apply fatigue (reduces responsiveness over time)
        fatigue_factor = 1.0 - (self.state.fatigue_level * 0.5)
        self.state.activation *= fatigue_factor
        
        # Record activation
        self.activation_history.append(self.state.activation)
        
        # Check if neuron should fire
        if self.state.activation >= self.state.threshold:
            return self.fire()
        
        # Natural decay of activation over time
        self.state.activation *= self.decay_rate
        
        return self.state.activation
    
    def fire(self) -> float:
        """Neuron fires/spikes - sends signal to connected neurons"""
        current_time = time.time() * 1000
        
        # Apply activation function
        output = self._apply_activation_function(self.state.activation)
        
        # Record spike
        self.spike_history.append(current_time)
        self.state.last_fired = current_time
        
        # Increase fatigue
        self.state.fatigue_level = min(1.0, self.state.fatigue_level + 0.05)
        
        # Reset activation after firing
        self.state.activation = 0.0
        
        logger.debug(f"🧠 Neuron {self.id} fired with output: {output:.3f}")
        return output
    
    def _apply_activation_function(self, x: float) -> float:
        """Apply the neuron's activation function"""
        if self.activation_function == ActivationFunction.SIGMOID:
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation_function == ActivationFunction.TANH:
            return np.tanh(x)
        elif self.activation_function == ActivationFunction.RELU:
            return max(0.0, x)
        elif self.activation_function == ActivationFunction.LEAKY_RELU:
            return max(0.1 * x, x)
        else:
            return 1.0 / (1.0 + np.exp(-x))  # Default to sigmoid
    
    def add_synapse(self, synapse: Synapse, is_incoming: bool):
        """Add a synaptic connection"""
        if is_incoming:
            self.incoming_synapses[f"{synapse.pre_neuron_id}->{self.id}"] = synapse
        else:
            self.outgoing_synapses[f"{self.id}->{synapse.post_neuron_id}"] = synapse
    
    def update_plasticity(self, reward: float):
        """Update neuron based on learning/reward signal"""
        # Hebbian learning: "neurons that fire together, wire together"
        for synapse in self.incoming_synapses.values():
            if synapse.last_active > (time.time() * 1000 - 100):  # Recently active
                weight_change = self.learning_rate * reward * self.state.plasticity
                synapse.weight += weight_change
                # Keep weights in bounds
                synapse.weight = max(-1.0, min(1.0, synapse.weight))
    
    def get_neuron_stats(self) -> Dict[str, Any]:
        """Get current neuron statistics"""
        current_time = time.time() * 1000
        recent_spikes = [s for s in self.spike_history if current_time - s < 1000]  # Last second
        
        return {
            "id": self.id,
            "type": self.type.value,
            "activation": self.state.activation,
            "threshold": self.state.threshold,
            "fatigue_level": self.state.fatigue_level,
            "plasticity": self.state.plasticity,
            "firing_rate": len(recent_spikes),  # Spikes per second
            "total_spikes": len(self.spike_history),
            "incoming_connections": len(self.incoming_synapses),
            "outgoing_connections": len(self.outgoing_synapses),
            "avg_activation": np.mean(list(self.activation_history)) if self.activation_history else 0.0
        }

class NeuralLayer:
    """🧠 Layer of neurons that work together"""
    
    def __init__(self, layer_name: str, layer_type: NeuronType, num_neurons: int):
        self.name = layer_name
        self.type = layer_type
        self.neurons = {}
        
        # Create neurons in this layer
        for i in range(num_neurons):
            neuron_id = f"{layer_name}_neuron_{i}"
            self.neurons[neuron_id] = ArtificialNeuron(neuron_id, layer_type)
        
        logger.info(f"🧠 Created neural layer '{layer_name}' with {num_neurons} {layer_type.value} neurons")
    
    def propagate_signal(self, inputs: List[float]) -> List[float]:
        """Propagate signals through this layer"""
        outputs = []
        
        for i, (neuron_id, neuron) in enumerate(self.neurons.items()):
            if i < len(inputs):
                output = neuron.receive_input(inputs[i])
                outputs.append(output)
            else:
                outputs.append(0.0)
        
        return outputs
    
    def get_layer_activity(self) -> Dict[str, Any]:
        """Get activity statistics for this layer"""
        total_activation = sum(n.state.activation for n in self.neurons.values())
        avg_activation = total_activation / len(self.neurons)
        
        active_neurons = sum(1 for n in self.neurons.values() if n.state.activation > 0.1)
        
        return {
            "name": self.name,
            "type": self.type.value,
            "neuron_count": len(self.neurons),
            "average_activation": avg_activation,
            "active_neurons": active_neurons,
            "activity_ratio": active_neurons / len(self.neurons)
        }

class NeuralNetwork:
    """🧠 Complete neural network simulating brain activity"""
    
    def __init__(self):
        self.layers = {}           # Dict[layer_name, NeuralLayer]
        self.synapses = {}         # Dict[synapse_id, Synapse]
        self.global_activity = 0.0 # Overall brain activity level
        
        # Neural oscillations (brain waves)
        self.oscillations = {
            "gamma": {"frequency": 40, "amplitude": 0.0},    # 30-100 Hz - consciousness
            "beta": {"frequency": 20, "amplitude": 0.0},     # 13-30 Hz - active thinking
            "alpha": {"frequency": 10, "amplitude": 0.0},    # 8-13 Hz - relaxed awareness
            "theta": {"frequency": 6, "amplitude": 0.0},     # 4-8 Hz - creativity/meditation
            "delta": {"frequency": 2, "amplitude": 0.0}      # 0.5-4 Hz - deep sleep
        }
        
        # Neural plasticity tracking
        self.learning_sessions = []
        self.network_connectivity = 0.0
        
        # Background neural activity
        self.background_activity_enabled = True
        self.neural_thread = None
        
        self._initialize_brain_structure()
        self._start_neural_processes()
        
        logger.info("🧠 Neural network initialized - artificial brain is now active!")
    
    def _initialize_brain_structure(self):
        """Initialize basic brain-like neural structure"""
        
        # Sensory processing layers
        self.layers["sensory_input"] = NeuralLayer("sensory_input", NeuronType.SENSORY, 50)
        self.layers["perception"] = NeuralLayer("perception", NeuronType.INTERNEURON, 100)
        
        # Cognitive processing layers
        self.layers["working_memory"] = NeuralLayer("working_memory", NeuronType.MEMORY, 80)
        self.layers["attention"] = NeuralLayer("attention", NeuronType.ATTENTION, 60)
        self.layers["reasoning"] = NeuralLayer("reasoning", NeuronType.INTERNEURON, 120)
        
        # Emotional processing
        self.layers["emotional"] = NeuralLayer("emotional", NeuronType.EMOTIONAL, 40)
        
        # Output layer
        self.layers["motor_output"] = NeuralLayer("motor_output", NeuronType.MOTOR, 30)
        
        # Create initial synaptic connections
        self._create_synaptic_connections()
        
        logger.info("🧠 Brain structure initialized with 480 artificial neurons")
    
    def _create_synaptic_connections(self):
        """Create synaptic connections between layers"""
        connections = [
            ("sensory_input", "perception"),
            ("perception", "working_memory"),
            ("perception", "attention"),
            ("working_memory", "reasoning"),
            ("attention", "reasoning"),
            ("reasoning", "emotional"),
            ("reasoning", "motor_output"),
            ("emotional", "motor_output"),
            ("working_memory", "attention")  # Feedback connection
        ]
        
        for source_layer, target_layer in connections:
            self._connect_layers(source_layer, target_layer)
    
    def _connect_layers(self, source_layer_name: str, target_layer_name: str, 
                       connection_density: float = 0.3):
        """Create synaptic connections between two layers"""
        source_layer = self.layers[source_layer_name]
        target_layer = self.layers[target_layer_name]
        
        connections_made = 0
        
        for source_neuron in source_layer.neurons.values():
            for target_neuron in target_layer.neurons.values():
                # Random connectivity with specified density
                if np.random.random() < connection_density:
                    synapse_id = f"{source_neuron.id}->{target_neuron.id}"
                    
                    # Random initial weight
                    initial_weight = np.random.normal(0.0, 0.5)
                    initial_weight = max(-1.0, min(1.0, initial_weight))
                    
                    synapse = Synapse(
                        pre_neuron_id=source_neuron.id,
                        post_neuron_id=target_neuron.id,
                        weight=initial_weight,
                        delay=np.random.uniform(1.0, 5.0),  # 1-5ms delay
                        plasticity=0.1,
                        last_active=0.0
                    )
                    
                    self.synapses[synapse_id] = synapse
                    source_neuron.add_synapse(synapse, is_incoming=False)
                    target_neuron.add_synapse(synapse, is_incoming=True)
                    connections_made += 1
        
        logger.info(f"🧠 Connected {source_layer_name} -> {target_layer_name}: {connections_made} synapses")
    
    def _start_neural_processes(self):
        """Start background neural activity"""
        if self.background_activity_enabled:
            self.neural_thread = threading.Thread(target=self._neural_activity_loop, daemon=True)
            self.neural_thread.start()
    
    def _neural_activity_loop(self):
        """Background neural activity simulation"""
        while self.background_activity_enabled:
            try:
                # Generate spontaneous neural activity
                self._generate_background_activity()
                
                # Update neural oscillations
                self._update_brain_waves()
                
                # Neural maintenance
                self._neural_maintenance()
                
                time.sleep(0.1)  # 100ms cycle (10 Hz base frequency)
                
            except Exception as e:
                logger.error(f"Error in neural activity loop: {e}")
                time.sleep(1)
    
    def _generate_background_activity(self):
        """Generate spontaneous background neural activity"""
        # Random neural firing to maintain baseline activity
        for layer in self.layers.values():
            for neuron in layer.neurons.values():
                # Small random inputs to maintain spontaneous activity
                if np.random.random() < 0.05:  # 5% chance per cycle
                    random_input = np.random.normal(0.0, 0.1)
                    neuron.receive_input(random_input)
                
                # Reduce fatigue over time
                neuron.state.fatigue_level *= 0.99
    
    def _update_brain_waves(self):
        """Update neural oscillation patterns (brain waves)"""
        current_time = time.time()
        
        # Calculate current brain wave amplitudes based on network activity
        total_activity = sum(
            sum(n.state.activation for n in layer.neurons.values())
            for layer in self.layers.values()
        )
        
        self.global_activity = total_activity / sum(len(layer.neurons) for layer in self.layers.values())
        
        # Update oscillation amplitudes based on activity
        if self.global_activity > 0.5:
            self.oscillations["gamma"]["amplitude"] = min(1.0, self.global_activity)
            self.oscillations["beta"]["amplitude"] = min(0.8, self.global_activity * 0.8)
        elif self.global_activity > 0.3:
            self.oscillations["alpha"]["amplitude"] = min(0.6, self.global_activity * 0.6)
        else:
            self.oscillations["theta"]["amplitude"] = min(0.4, self.global_activity * 2.0)
    
    def _neural_maintenance(self):
        """Perform neural maintenance and cleanup"""
        # Prune weak synapses
        weak_synapses = [
            synapse_id for synapse_id, synapse in self.synapses.items()
            if abs(synapse.weight) < 0.01
        ]
        
        for synapse_id in weak_synapses[:10]:  # Remove up to 10 weak synapses per cycle
            del self.synapses[synapse_id]
    
    def process_thought(self, thought_input: str, intensity: float = 0.5) -> Dict[str, Any]:
        """Process a thought through the neural network"""
        logger.info(f"🧠 Neural network processing thought: {thought_input[:50]}...")
        
        # Convert thought to neural input pattern
        input_pattern = self._encode_thought_to_neural_pattern(thought_input, intensity)
        
        # Propagate through network
        activations = {}
        current_input = input_pattern
        
        # Process through layers in sequence
        layer_sequence = ["sensory_input", "perception", "working_memory", "attention", 
                         "reasoning", "emotional", "motor_output"]
        
        for layer_name in layer_sequence:
            if layer_name in self.layers:
                layer_output = self.layers[layer_name].propagate_signal(current_input)
                activations[layer_name] = {
                    "average_activation": np.mean(layer_output),
                    "max_activation": np.max(layer_output),
                    "active_neurons": sum(1 for x in layer_output if x > 0.1)
                }
                current_input = layer_output
        
        # Update synaptic weights based on successful thought processing
        self._reinforce_active_pathways(intensity)
        
        return {
            "input_thought": thought_input,
            "processing_intensity": intensity,
            "layer_activations": activations,
            "global_activity": self.global_activity,
            "dominant_brainwave": self._get_dominant_brainwave(),
            "neural_efficiency": self._calculate_neural_efficiency(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _encode_thought_to_neural_pattern(self, thought: str, intensity: float) -> List[float]:
        """Convert a thought string into neural activation pattern"""
        # Simple encoding: convert text to neural pattern
        # In reality, this would be much more sophisticated
        
        pattern_size = len(self.layers["sensory_input"].neurons)
        pattern = [0.0] * pattern_size
        
        # Hash-based encoding for consistency
        for i, char in enumerate(thought.lower()):
            if i >= pattern_size:
                break
            # Convert character to activation
            pattern[i] = (ord(char) / 128.0) * intensity
        
        # Add some random noise for biological realism
        for i in range(len(pattern)):
            pattern[i] += np.random.normal(0.0, 0.05)
            pattern[i] = max(0.0, min(1.0, pattern[i]))
        
        return pattern
    
    def _reinforce_active_pathways(self, reward: float):
        """Strengthen synapses that were active during thought processing"""
        current_time = time.time() * 1000
        
        for synapse in self.synapses.values():
            if current_time - synapse.last_active < 1000:  # Recently active
                weight_change = 0.01 * reward * synapse.plasticity
                synapse.weight += weight_change
                synapse.weight = max(-1.0, min(1.0, synapse.weight))
    
    def _get_dominant_brainwave(self) -> str:
        """Get the currently dominant brain wave"""
        max_amplitude = 0.0
        dominant = "delta"
        
        for wave_type, wave_data in self.oscillations.items():
            if wave_data["amplitude"] > max_amplitude:
                max_amplitude = wave_data["amplitude"]
                dominant = wave_type
        
        return dominant
    
    def _calculate_neural_efficiency(self) -> float:
        """Calculate overall neural network efficiency"""
        total_neurons = sum(len(layer.neurons) for layer in self.layers.values())
        active_neurons = sum(
            sum(1 for n in layer.neurons.values() if n.state.activation > 0.1)
            for layer in self.layers.values()
        )
        
        if total_neurons == 0:
            return 0.0
        
        return active_neurons / total_neurons
    
    def simulate_neural_storm(self, duration: int = 5):
        """Simulate intense neural activity (like a creative burst)"""
        logger.info(f"🧠 Simulating neural storm for {duration} seconds...")
        
        start_time = time.time()
        storm_results = []
        
        while time.time() - start_time < duration:
            # Generate intense random activity across all layers
            for layer in self.layers.values():
                for neuron in layer.neurons.values():
                    if np.random.random() < 0.3:  # 30% activation chance
                        intense_input = np.random.normal(0.5, 0.2)
                        neuron.receive_input(intense_input)
            
            # Record activity burst
            storm_results.append({
                "timestamp": time.time(),
                "global_activity": self.global_activity,
                "active_layers": len([l for l in self.layers.values() 
                                    if l.get_layer_activity()["activity_ratio"] > 0.2])
            })
            
            time.sleep(0.1)
        
        logger.info(f"🧠 Neural storm completed - peak activity: {max(r['global_activity'] for r in storm_results):.3f}")
        
        return {
            "duration": duration,
            "peak_activity": max(r['global_activity'] for r in storm_results),
            "average_activity": np.mean([r['global_activity'] for r in storm_results]),
            "activity_bursts": len(storm_results)
        }
    
    def get_neural_status(self) -> Dict[str, Any]:
        """Get comprehensive neural network status"""
        layer_stats = {}
        for layer_name, layer in self.layers.items():
            layer_stats[layer_name] = layer.get_layer_activity()
        
        # Calculate network connectivity
        total_possible_connections = sum(len(layer.neurons) for layer in self.layers.values()) ** 2
        actual_connections = len(self.synapses)
        connectivity_ratio = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
        return {
            "total_neurons": sum(len(layer.neurons) for layer in self.layers.values()),
            "total_synapses": len(self.synapses),
            "global_activity": self.global_activity,
            "connectivity_ratio": connectivity_ratio,
            "brain_waves": self.oscillations,
            "dominant_frequency": self._get_dominant_brainwave(),
            "neural_efficiency": self._calculate_neural_efficiency(),
            "layer_statistics": layer_stats,
            "background_activity": self.background_activity_enabled,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_neural_state(self, filepath: str):
        """Save neural network state to file"""
        state = {
            "layers": {name: layer.get_layer_activity() for name, layer in self.layers.items()},
            "synapses": len(self.synapses),
            "global_activity": self.global_activity,
            "oscillations": self.oscillations,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"🧠 Neural state saved to: {filepath}")
    
    def shutdown(self):
        """Safely shutdown neural network"""
        self.background_activity_enabled = False
        if self.neural_thread:
            self.neural_thread.join(timeout=1.0)
        
        logger.info("🧠 Neural network shutdown completed")