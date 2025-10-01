"""
🔹 MULTIMODAL PROCESSING SYSTEM
Support for text, images, audio, video, and sensor data
with unified embedding space and cross-modal reasoning
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime
import base64
import io
import json
import uuid

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_VISION_AVAILABLE = True
except ImportError:
    TORCH_VISION_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from structlog import get_logger

logger = get_logger()

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"
    STRUCTURED = "structured"  # Tables, graphs, etc.
    TEMPORAL = "temporal"  # Time series data

class ProcessingPriority(Enum):
    REALTIME = "realtime"  # < 100ms
    INTERACTIVE = "interactive"  # < 1s
    BATCH = "batch"  # < 10s
    BACKGROUND = "background"  # No time constraint

@dataclass
class MultimodalInput:
    """Represents input data from any modality"""
    id: str
    modality: ModalityType
    data: Any  # Raw data (text, image bytes, audio samples, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: ProcessingPriority = ProcessingPriority.INTERACTIVE
    embedding: Optional[np.ndarray] = None
    processed: bool = False

@dataclass
class CrossModalContext:
    """Context for cross-modal reasoning"""
    inputs: List[MultimodalInput]
    relationships: Dict[str, List[str]]  # input_id -> related_input_ids
    temporal_sequence: List[str]  # Ordered sequence of input_ids
    attention_weights: Dict[Tuple[str, str], float] = field(default_factory=dict)

class MultimodalProcessor:
    """🔹 Unified multimodal processing system"""
    
    def __init__(self, 
                 embedding_dim: int = 1024,
                 max_sequence_length: int = 8192,
                 enable_cross_modal_attention: bool = True):
        
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.enable_cross_modal_attention = enable_cross_modal_attention
        
        # Processing pipelines for each modality
        self.processors = {
            ModalityType.TEXT: TextProcessor(embedding_dim),
            ModalityType.IMAGE: ImageProcessor(embedding_dim),
            ModalityType.AUDIO: AudioProcessor(embedding_dim),
            ModalityType.VIDEO: VideoProcessor(embedding_dim),
            ModalityType.SENSOR: SensorProcessor(embedding_dim),
            ModalityType.STRUCTURED: StructuredProcessor(embedding_dim),
            ModalityType.TEMPORAL: TemporalProcessor(embedding_dim)
        }
        
        # Cross-modal fusion layers
        if TORCH_VISION_AVAILABLE:
            self.fusion_network = CrossModalFusionNetwork(embedding_dim)
        
        # Unified embedding space
        self.embedding_cache = {}
        self.embedding_index = {}  # For similarity search
        
        # Processing queues by priority
        self.processing_queues = {
            priority: asyncio.Queue() for priority in ProcessingPriority
        }
        
        # Performance metrics
        self.metrics = {
            "inputs_processed": {modality.value: 0 for modality in ModalityType},
            "average_processing_time": {modality.value: 0.0 for modality in ModalityType},
            "cross_modal_queries": 0,
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0
        }
        
        # Start processing workers
        self._start_processing_workers()
        
        logger.info(f"🔹 Multimodal Processor initialized")
        logger.info(f"🔹 Embedding dim: {embedding_dim}, Max sequence: {max_sequence_length}")
    
    def _start_processing_workers(self):
        """Start background workers for different priority queues"""
        for priority in ProcessingPriority:
            worker = asyncio.create_task(self._processing_worker(priority))
            # Store worker reference to prevent garbage collection
            if not hasattr(self, '_workers'):
                self._workers = []
            self._workers.append(worker)
    
    async def _processing_worker(self, priority: ProcessingPriority):
        """Worker to process inputs by priority"""
        queue = self.processing_queues[priority]
        
        while True:
            try:
                # Get timeout based on priority
                timeout = {
                    ProcessingPriority.REALTIME: 0.1,
                    ProcessingPriority.INTERACTIVE: 1.0,
                    ProcessingPriority.BATCH: 10.0,
                    ProcessingPriority.BACKGROUND: None
                }[priority]
                
                # Wait for input with timeout
                try:
                    input_data = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
                
                # Process the input
                await self._process_input(input_data)
                
            except Exception as e:
                logger.error(f"Error in {priority.value} processing worker: {e}")
                await asyncio.sleep(1)
    
    async def process_input(self, input_data: MultimodalInput) -> MultimodalInput:
        """Process a single multimodal input"""
        start_time = time.time()
        
        # Add to appropriate processing queue
        await self.processing_queues[input_data.priority].put(input_data)
        
        # Wait for processing to complete
        while not input_data.processed:
            await asyncio.sleep(0.01)
        
        # Update metrics
        processing_time = time.time() - start_time
        modality = input_data.modality.value
        
        # Update average processing time
        current_avg = self.metrics["average_processing_time"][modality]
        current_count = self.metrics["inputs_processed"][modality]
        new_avg = (current_avg * current_count + processing_time) / (current_count + 1)
        
        self.metrics["average_processing_time"][modality] = new_avg
        self.metrics["inputs_processed"][modality] += 1
        
        return input_data
    
    async def _process_input(self, input_data: MultimodalInput):
        """Internal method to process input data"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(input_data)
            if cache_key in self.embedding_cache:
                input_data.embedding = self.embedding_cache[cache_key]
                self.metrics["embedding_cache_hits"] += 1
                input_data.processed = True
                return
            
            self.metrics["embedding_cache_misses"] += 1
            
            # Get appropriate processor
            processor = self.processors[input_data.modality]
            
            # Process the input
            embedding = await processor.process(input_data.data, input_data.metadata)
            input_data.embedding = embedding
            
            # Cache the embedding
            self.embedding_cache[cache_key] = embedding
            
            # Add to searchable index
            self.embedding_index[input_data.id] = {
                'embedding': embedding,
                'modality': input_data.modality,
                'metadata': input_data.metadata,
                'timestamp': input_data.timestamp
            }
            
            input_data.processed = True
            
        except Exception as e:
            logger.error(f"Error processing {input_data.modality.value} input: {e}")
            # Create zero embedding as fallback
            input_data.embedding = np.zeros(self.embedding_dim)
            input_data.processed = True
    
    def _get_cache_key(self, input_data: MultimodalInput) -> str:
        """Generate cache key for input data"""
        # Simple hash-based cache key
        data_hash = hash(str(input_data.data)[:1000])  # First 1000 chars for efficiency
        return f"{input_data.modality.value}_{data_hash}"
    
    async def cross_modal_query(self, 
                               query_input: MultimodalInput,
                               target_modalities: List[ModalityType],
                               top_k: int = 10) -> List[Tuple[str, float, MultimodalInput]]:
        """Perform cross-modal similarity search"""
        self.metrics["cross_modal_queries"] += 1
        
        # Process query input if not already processed
        if not query_input.processed:
            await self.process_input(query_input)
        
        query_embedding = query_input.embedding
        results = []
        
        # Search through embedding index
        for input_id, indexed_data in self.embedding_index.items():
            if indexed_data['modality'] in target_modalities:
                # Calculate similarity
                similarity = self._calculate_similarity(query_embedding, indexed_data['embedding'])
                
                # Create result input object
                result_input = MultimodalInput(
                    id=input_id,
                    modality=indexed_data['modality'],
                    data=None,  # Data not stored in index
                    metadata=indexed_data['metadata'],
                    timestamp=indexed_data['timestamp'],
                    embedding=indexed_data['embedding'],
                    processed=True
                )
                
                results.append((input_id, similarity, result_input))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    async def cross_modal_reasoning(self, context: CrossModalContext) -> Dict[str, Any]:
        """Perform cross-modal reasoning across multiple inputs"""
        if not self.enable_cross_modal_attention or not TORCH_VISION_AVAILABLE:
            # Fallback to simple aggregation
            return await self._simple_cross_modal_aggregation(context)
        
        # Process all inputs if needed
        for input_data in context.inputs:
            if not input_data.processed:
                await self.process_input(input_data)
        
        # Extract embeddings
        embeddings = []
        input_ids = []
        modalities = []
        
        for input_data in context.inputs:
            if input_data.embedding is not None:
                embeddings.append(torch.tensor(input_data.embedding))
                input_ids.append(input_data.id)
                modalities.append(input_data.modality)
        
        if not embeddings:
            return {"error": "No valid embeddings found"}
        
        # Stack embeddings
        embeddings_tensor = torch.stack(embeddings)
        
        # Apply cross-modal fusion
        fused_representation = self.fusion_network(embeddings_tensor, modalities)
        
        # Calculate attention weights
        attention_weights = self.fusion_network.get_attention_weights()
        
        # Update context with attention weights
        for i, source_id in enumerate(input_ids):
            for j, target_id in enumerate(input_ids):
                if i != j:
                    context.attention_weights[(source_id, target_id)] = float(attention_weights[i, j])
        
        return {
            "fused_representation": fused_representation.detach().numpy(),
            "attention_weights": context.attention_weights,
            "input_contributions": {
                input_ids[i]: float(attention_weights[i].sum())
                for i in range(len(input_ids))
            }
        }
    
    async def _simple_cross_modal_aggregation(self, context: CrossModalContext) -> Dict[str, Any]:
        """Simple aggregation when advanced fusion is not available"""
        # Process all inputs
        valid_embeddings = []
        input_ids = []
        
        for input_data in context.inputs:
            if not input_data.processed:
                await self.process_input(input_data)
            
            if input_data.embedding is not None:
                valid_embeddings.append(input_data.embedding)
                input_ids.append(input_data.id)
        
        if not valid_embeddings:
            return {"error": "No valid embeddings found"}
        
        # Simple average aggregation
        fused_representation = np.mean(valid_embeddings, axis=0)
        
        # Equal weights for all inputs
        equal_weight = 1.0 / len(valid_embeddings)
        input_contributions = {input_id: equal_weight for input_id in input_ids}
        
        return {
            "fused_representation": fused_representation,
            "attention_weights": {},
            "input_contributions": input_contributions
        }
    
    def get_modality_statistics(self) -> Dict[str, Any]:
        """Get processing statistics by modality"""
        return {
            "supported_modalities": [modality.value for modality in ModalityType],
            "inputs_processed": self.metrics["inputs_processed"],
            "average_processing_time": self.metrics["average_processing_time"],
            "cross_modal_queries": self.metrics["cross_modal_queries"],
            "cache_hit_rate": (
                self.metrics["embedding_cache_hits"] / 
                (self.metrics["embedding_cache_hits"] + self.metrics["embedding_cache_misses"])
                if (self.metrics["embedding_cache_hits"] + self.metrics["embedding_cache_misses"]) > 0
                else 0.0
            ),
            "embedding_cache_size": len(self.embedding_cache),
            "indexed_embeddings": len(self.embedding_index)
        }


class BaseModalityProcessor:
    """Base class for modality-specific processors"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
    
    async def process(self, data: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Process input data and return embedding"""
        raise NotImplementedError


class TextProcessor(BaseModalityProcessor):
    """Process text inputs"""
    
    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)
        # In a real implementation, this would use a language model
        self.vocab_size = 50000  # Placeholder
    
    async def process(self, data: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Process text data"""
        if isinstance(data, str):
            text = data
        else:
            text = str(data)
        
        # Simple text embedding (in reality, would use transformer models)
        # Hash-based embedding for demonstration
        text_hash = hash(text.lower())
        
        # Create pseudo-embedding
        np.random.seed(abs(text_hash) % (2**31))  # Deterministic randomness
        embedding = np.random.normal(0, 1, self.embedding_dim)
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)


class ImageProcessor(BaseModalityProcessor):
    """Process image inputs"""
    
    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)
        
        if TORCH_VISION_AVAILABLE:
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    async def process(self, data: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Process image data"""
        try:
            # Handle different input formats
            if isinstance(data, str):
                # Assume base64 encoded image
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(data, bytes):
                image = Image.open(io.BytesIO(data))
            elif hasattr(data, 'size'):  # PIL Image
                image = data
            else:
                # Create placeholder embedding
                return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if TORCH_VISION_AVAILABLE:
                # Apply transforms
                tensor = self.transform(image)
                
                # Simple CNN-like processing (placeholder)
                # In reality, would use ResNet, CLIP, etc.
                flattened = tensor.flatten()
                
                # Reduce to embedding dimension
                if len(flattened) > self.embedding_dim:
                    # Simple pooling
                    pooled = torch.nn.functional.adaptive_avg_pool1d(
                        flattened.unsqueeze(0).unsqueeze(0), 
                        self.embedding_dim
                    )
                    embedding = pooled.squeeze().detach().numpy()
                else:
                    # Pad to embedding dimension
                    embedding = np.pad(flattened.numpy(), 
                                     (0, max(0, self.embedding_dim - len(flattened))),
                                     mode='constant')[:self.embedding_dim]
            else:
                # Fallback: simple color histogram
                image_array = np.array(image)
                color_hist = np.histogram(image_array.flatten(), bins=min(256, self.embedding_dim))[0]
                embedding = color_hist.astype(np.float32)
                
                # Pad or truncate to embedding dimension
                if len(embedding) < self.embedding_dim:
                    embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
                else:
                    embedding = embedding[:self.embedding_dim]
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)


class AudioProcessor(BaseModalityProcessor):
    """Process audio inputs"""
    
    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)
        self.sample_rate = 22050  # Standard sample rate
    
    async def process(self, data: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Process audio data"""
        try:
            if AUDIO_AVAILABLE:
                # Handle different audio input formats
                if isinstance(data, str) and data.endswith(('.wav', '.mp3', '.flac')):
                    # Audio file path
                    audio, sr = librosa.load(data, sr=self.sample_rate)
                elif isinstance(data, bytes):
                    # Audio bytes
                    audio_file = io.BytesIO(data)
                    audio, sr = sf.read(audio_file)
                elif isinstance(data, (list, np.ndarray)):
                    # Raw audio samples
                    audio = np.array(data)
                    sr = metadata.get('sample_rate', self.sample_rate)
                else:
                    # Fallback
                    return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
                
                # Extract audio features
                # MFCC features
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(audio)
                
                # Combine features
                features = np.concatenate([
                    mfcc.mean(axis=1),  # Average MFCC
                    [spectral_centroids.mean()],
                    [spectral_rolloff.mean()],
                    [zcr.mean()]
                ])
                
                # Adjust to embedding dimension
                if len(features) < self.embedding_dim:
                    features = np.pad(features, (0, self.embedding_dim - len(features)))
                else:
                    features = features[:self.embedding_dim]
                
                # Normalize
                embedding = features / (np.linalg.norm(features) + 1e-8)
                return embedding.astype(np.float32)
            
            else:
                # Fallback without audio libraries
                # Simple spectral analysis if possible
                if isinstance(data, (list, np.ndarray)):
                    audio = np.array(data)
                    # Simple FFT-based features
                    fft = np.fft.fft(audio)
                    magnitude = np.abs(fft)[:self.embedding_dim//2]
                    
                    # Pad to embedding dimension
                    embedding = np.pad(magnitude, (0, self.embedding_dim - len(magnitude)))
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    return embedding.astype(np.float32)
                
                return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)


class VideoProcessor(BaseModalityProcessor):
    """Process video inputs"""
    
    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)
        self.image_processor = ImageProcessor(embedding_dim // 2)
        self.audio_processor = AudioProcessor(embedding_dim // 2)
    
    async def process(self, data: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Process video data by combining visual and audio features"""
        try:
            if OPENCV_AVAILABLE:
                # Handle video file
                if isinstance(data, str):
                    cap = cv2.VideoCapture(data)
                    
                    # Extract a few key frames
                    frames = []
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    for i in range(0, frame_count, max(1, frame_count // 10)):  # Sample 10 frames
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if ret:
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame_rgb)
                    
                    cap.release()
                    
                    if frames:
                        # Process frames and average embeddings
                        frame_embeddings = []
                        for frame in frames[:5]:  # Use first 5 frames
                            frame_pil = Image.fromarray(frame)
                            frame_emb = await self.image_processor.process(frame_pil, {})
                            frame_embeddings.append(frame_emb)
                        
                        visual_embedding = np.mean(frame_embeddings, axis=0)
                    else:
                        visual_embedding = np.random.normal(0, 1, self.embedding_dim // 2)
                    
                    # For audio, would extract audio track
                    # For now, use placeholder
                    audio_embedding = np.random.normal(0, 1, self.embedding_dim // 2)
                    
                    # Combine visual and audio
                    embedding = np.concatenate([visual_embedding, audio_embedding])
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    
                    return embedding.astype(np.float32)
            
            # Fallback
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)


class SensorProcessor(BaseModalityProcessor):
    """Process sensor data (IoT, telemetry, etc.)"""
    
    async def process(self, data: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Process sensor data"""
        try:
            # Handle different sensor data formats
            if isinstance(data, dict):
                # Extract numerical values
                values = []
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        values.append(float(value))
                    elif isinstance(value, (list, tuple)):
                        values.extend([float(v) for v in value if isinstance(v, (int, float))])
                
                sensor_array = np.array(values) if values else np.array([0])
            
            elif isinstance(data, (list, tuple, np.ndarray)):
                sensor_array = np.array(data, dtype=float).flatten()
            
            else:
                # Try to convert to float
                sensor_array = np.array([float(data)])
            
            # Statistical features
            if len(sensor_array) > 0:
                features = [
                    np.mean(sensor_array),
                    np.std(sensor_array),
                    np.min(sensor_array),
                    np.max(sensor_array),
                    np.median(sensor_array)
                ]
                
                # Add frequency domain features if enough data
                if len(sensor_array) > 10:
                    fft = np.fft.fft(sensor_array)
                    magnitude = np.abs(fft)[:min(10, len(fft)//2)]
                    features.extend(magnitude.tolist())
                
                # Pad or truncate to embedding dimension
                features = np.array(features)
                if len(features) < self.embedding_dim:
                    features = np.pad(features, (0, self.embedding_dim - len(features)))
                else:
                    features = features[:self.embedding_dim]
                
                # Normalize
                embedding = features / (np.linalg.norm(features) + 1e-8)
                return embedding.astype(np.float32)
            
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)


class StructuredProcessor(BaseModalityProcessor):
    """Process structured data (tables, graphs, etc.)"""
    
    async def process(self, data: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Process structured data"""
        try:
            if isinstance(data, dict):
                # JSON/dict structure
                embedding = self._process_dict(data)
            elif isinstance(data, list):
                # List/array structure
                embedding = self._process_list(data)
            else:
                # Convert to string and use hash-based embedding
                data_str = str(data)
                data_hash = hash(data_str)
                np.random.seed(abs(data_hash) % (2**31))
                embedding = np.random.normal(0, 1, self.embedding_dim)
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error processing structured data: {e}")
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
    
    def _process_dict(self, data: dict) -> np.ndarray:
        """Process dictionary data"""
        features = []
        
        # Count different data types
        type_counts = {}
        for value in data.values():
            type_name = type(value).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Add type distribution features
        common_types = ['str', 'int', 'float', 'list', 'dict', 'bool']
        for type_name in common_types:
            features.append(type_counts.get(type_name, 0))
        
        # Add structural features
        features.extend([
            len(data),  # Number of keys
            sum(len(str(k)) for k in data.keys()) / len(data) if data else 0,  # Avg key length
            sum(len(str(v)) for v in data.values()) / len(data) if data else 0,  # Avg value length
        ])
        
        # Hash-based features for key names
        key_hash = hash(tuple(sorted(data.keys())))
        np.random.seed(abs(key_hash) % (2**31))
        hash_features = np.random.normal(0, 1, max(0, self.embedding_dim - len(features)))
        
        features.extend(hash_features)
        return np.array(features[:self.embedding_dim])
    
    def _process_list(self, data: list) -> np.ndarray:
        """Process list data"""
        features = []
        
        if data:
            # Basic statistics
            numeric_values = [x for x in data if isinstance(x, (int, float))]
            if numeric_values:
                features.extend([
                    np.mean(numeric_values),
                    np.std(numeric_values),
                    len(numeric_values) / len(data)  # Numeric ratio
                ])
            else:
                features.extend([0, 0, 0])
            
            # Type distribution
            type_counts = {}
            for item in data:
                type_name = type(item).__name__
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            common_types = ['str', 'int', 'float', 'list', 'dict']
            for type_name in common_types:
                features.append(type_counts.get(type_name, 0) / len(data))
        
        # Pad to embedding dimension
        while len(features) < self.embedding_dim:
            features.append(0.0)
        
        return np.array(features[:self.embedding_dim])


class TemporalProcessor(BaseModalityProcessor):
    """Process temporal/time series data"""
    
    async def process(self, data: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Process time series data"""
        try:
            # Convert to numpy array
            if isinstance(data, (list, tuple)):
                time_series = np.array(data, dtype=float)
            elif isinstance(data, np.ndarray):
                time_series = data.flatten().astype(float)
            else:
                return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
            
            if len(time_series) == 0:
                return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
            
            # Time series features
            features = []
            
            # Statistical features
            features.extend([
                np.mean(time_series),
                np.std(time_series),
                np.min(time_series),
                np.max(time_series),
                np.median(time_series),
                np.percentile(time_series, 25),
                np.percentile(time_series, 75)
            ])
            
            # Trend analysis
            if len(time_series) > 1:
                # Linear trend
                x = np.arange(len(time_series))
                slope = np.polyfit(x, time_series, 1)[0]
                features.append(slope)
                
                # Autocorrelation (lag 1)
                if len(time_series) > 2:
                    autocorr = np.corrcoef(time_series[:-1], time_series[1:])[0, 1]
                    features.append(autocorr if not np.isnan(autocorr) else 0)
                else:
                    features.append(0)
            else:
                features.extend([0, 0])
            
            # Frequency domain features
            if len(time_series) > 4:
                fft = np.fft.fft(time_series)
                magnitude = np.abs(fft)[:len(fft)//2]
                
                # Dominant frequency
                dominant_freq_idx = np.argmax(magnitude)
                features.append(dominant_freq_idx / len(magnitude))
                
                # Spectral energy
                spectral_energy = np.sum(magnitude**2)
                features.append(np.log(spectral_energy + 1e-8))
                
                # Add some magnitude features
                top_freqs = sorted(magnitude, reverse=True)[:5]
                features.extend(top_freqs)
            else:
                features.extend([0] * 7)  # Placeholder for frequency features
            
            # Pad or truncate to embedding dimension
            features = np.array(features)
            if len(features) < self.embedding_dim:
                features = np.pad(features, (0, self.embedding_dim - len(features)))
            else:
                features = features[:self.embedding_dim]
            
            # Normalize
            embedding = features / (np.linalg.norm(features) + 1e-8)
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error processing temporal data: {e}")
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)


class CrossModalFusionNetwork(nn.Module):
    """Neural network for cross-modal attention and fusion"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Store attention weights for analysis
        self.last_attention_weights = None
    
    def forward(self, embeddings: torch.Tensor, modalities: List[ModalityType]) -> torch.Tensor:
        """Apply cross-modal fusion"""
        # embeddings shape: (num_inputs, embedding_dim)
        
        # Add batch dimension and transpose for attention
        embeddings = embeddings.unsqueeze(1)  # (num_inputs, 1, embedding_dim)
        embeddings = embeddings.transpose(0, 1)  # (1, num_inputs, embedding_dim)
        
        # Apply cross-attention
        attended, attention_weights = self.cross_attention(
            embeddings, embeddings, embeddings
        )
        
        # Store attention weights
        self.last_attention_weights = attention_weights.squeeze(0)  # Remove batch dim
        
        # Pool attended representations
        pooled = attended.mean(dim=0)  # Average over sequence length
        
        # Apply fusion layers
        fused = self.fusion_layers(pooled)
        
        return fused.squeeze(0)  # Remove batch dimension
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get the last computed attention weights"""
        return self.last_attention_weights if self.last_attention_weights is not None else torch.zeros(1, 1)