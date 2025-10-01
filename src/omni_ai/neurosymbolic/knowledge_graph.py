"""
🔹 KNOWLEDGE GRAPH SYSTEM
Advanced knowledge graph with entity relationships, semantic networks,
and integration with symbolic reasoning for neurosymbolic AI
"""

import asyncio
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

from structlog import get_logger

logger = get_logger()

class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    OBJECT = "object"
    ABSTRACT = "abstract"
    TEMPORAL = "temporal"

class RelationType(Enum):
    IS_A = "is_a"                    # Taxonomic relationship
    PART_OF = "part_of"              # Meronymic relationship
    HAS_PROPERTY = "has_property"    # Property relationship
    LOCATED_IN = "located_in"        # Spatial relationship
    TEMPORAL_BEFORE = "temporal_before"  # Temporal relationship
    TEMPORAL_AFTER = "temporal_after"
    CAUSED_BY = "caused_by"          # Causal relationship
    CAUSES = "causes"
    SIMILAR_TO = "similar_to"        # Similarity relationship
    OPPOSITE_TO = "opposite_to"      # Opposition relationship
    ASSOCIATED_WITH = "associated_with"  # General association
    CREATED_BY = "created_by"        # Creation relationship
    OWNS = "owns"                    # Ownership relationship
    WORKS_FOR = "works_for"          # Employment relationship
    FRIEND_OF = "friend_of"          # Social relationship
    PARENT_OF = "parent_of"          # Family relationship
    CHILD_OF = "child_of"
    PRACTICES = "practices"          # Practice/engagement relationship
    CUSTOM = "custom"                # User-defined relationship

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    aliases: Set[str] = field(default_factory=set)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"{self.name} ({self.entity_type.value})"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def add_alias(self, alias: str):
        """Add an alias for this entity"""
        self.aliases.add(alias)
        self.updated_at = datetime.now()
    
    def add_property(self, key: str, value: Any):
        """Add a property to this entity"""
        self.properties[key] = value
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type.value,
            "properties": self.properties,
            "aliases": list(self.aliases),
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

@dataclass
class Relation:
    """Represents a relationship between entities in the knowledge graph"""
    id: str
    source_entity: str  # Entity ID
    target_entity: str  # Entity ID
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"{self.source_entity} --{self.relation_type.value}-> {self.target_entity}"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def reverse_relation_type(self) -> Optional[RelationType]:
        """Get the reverse relation type if it exists"""
        reverse_mapping = {
            RelationType.IS_A: None,  # No direct reverse
            RelationType.PART_OF: None,  # Would need "has_part"
            RelationType.HAS_PROPERTY: None,
            RelationType.LOCATED_IN: None,
            RelationType.TEMPORAL_BEFORE: RelationType.TEMPORAL_AFTER,
            RelationType.TEMPORAL_AFTER: RelationType.TEMPORAL_BEFORE,
            RelationType.CAUSED_BY: RelationType.CAUSES,
            RelationType.CAUSES: RelationType.CAUSED_BY,
            RelationType.SIMILAR_TO: RelationType.SIMILAR_TO,
            RelationType.OPPOSITE_TO: RelationType.OPPOSITE_TO,
            RelationType.PARENT_OF: RelationType.CHILD_OF,
            RelationType.CHILD_OF: RelationType.PARENT_OF,
            RelationType.FRIEND_OF: RelationType.FRIEND_OF
        }
        return reverse_mapping.get(self.relation_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary representation"""
        return {
            "id": self.id,
            "source": self.source_entity,
            "target": self.target_entity,
            "type": self.relation_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "properties": self.properties,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class KnowledgeGraph:
    """🔹 Advanced Knowledge Graph with entity relationships and semantic networks"""
    
    def __init__(self, enable_embeddings: bool = True):
        # Core graph structure
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        
        # Indexes for efficient querying
        self.entity_name_index: Dict[str, str] = {}  # name -> entity_id
        self.entity_type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self.relation_source_index: Dict[str, Set[str]] = defaultdict(set)  # source -> relation_ids
        self.relation_target_index: Dict[str, Set[str]] = defaultdict(set)  # target -> relation_ids
        self.relation_type_index: Dict[RelationType, Set[str]] = defaultdict(set)
        
        # Configuration
        self.enable_embeddings = enable_embeddings
        self.embedding_dimension = 768 if enable_embeddings else 0
        
        # Performance metrics
        self.metrics = {
            "total_entities": 0,
            "total_relations": 0,
            "entities_by_type": {et.value: 0 for et in EntityType},
            "relations_by_type": {rt.value: 0 for rt in RelationType},
            "average_entity_degree": 0.0,
            "graph_density": 0.0,
            "last_updated": datetime.now()
        }
        
        logger.info("🔹 Knowledge Graph initialized")
        logger.info(f"🔹 Embedding support: {'enabled' if enable_embeddings else 'disabled'}")
    
    def add_entity(self, name: str, entity_type: EntityType, 
                  properties: Optional[Dict[str, Any]] = None,
                  aliases: Optional[Set[str]] = None,
                  confidence: float = 1.0,
                  embedding: Optional[np.ndarray] = None) -> Entity:
        """Add an entity to the knowledge graph"""
        
        # Check if entity already exists by name
        if name.lower() in self.entity_name_index:
            existing_id = self.entity_name_index[name.lower()]
            logger.debug(f"Entity '{name}' already exists with ID: {existing_id}")
            return self.entities[existing_id]
        
        # Create new entity
        entity_id = str(uuid.uuid4())
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            aliases=aliases or set(),
            confidence=confidence,
            embedding=embedding
        )
        
        # Add to graph
        self.entities[entity_id] = entity
        
        # Update indexes
        self.entity_name_index[name.lower()] = entity_id
        self.entity_type_index[entity_type].add(entity_id)
        
        # Add aliases to name index
        for alias in entity.aliases:
            self.entity_name_index[alias.lower()] = entity_id
        
        # Update metrics
        self.metrics["total_entities"] += 1
        self.metrics["entities_by_type"][entity_type.value] += 1
        self.metrics["last_updated"] = datetime.now()
        self._update_graph_metrics()
        
        logger.debug(f"Added entity: {entity}")
        return entity
    
    def add_relation(self, source_name: str, target_name: str, 
                    relation_type: RelationType,
                    weight: float = 1.0,
                    confidence: float = 1.0,
                    properties: Optional[Dict[str, Any]] = None,
                    bidirectional: bool = False) -> Relation:
        """Add a relation between entities"""
        
        # Find or create entities
        source_entity = self.get_entity_by_name(source_name)
        target_entity = self.get_entity_by_name(target_name)
        
        if not source_entity:
            logger.warning(f"Source entity '{source_name}' not found, creating with CONCEPT type")
            source_entity = self.add_entity(source_name, EntityType.CONCEPT)
        
        if not target_entity:
            logger.warning(f"Target entity '{target_name}' not found, creating with CONCEPT type")
            target_entity = self.add_entity(target_name, EntityType.CONCEPT)
        
        # Create relation
        relation_id = str(uuid.uuid4())
        relation = Relation(
            id=relation_id,
            source_entity=source_entity.id,
            target_entity=target_entity.id,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
            properties=properties or {}
        )
        
        # Add to graph
        self.relations[relation_id] = relation
        
        # Update indexes
        self.relation_source_index[source_entity.id].add(relation_id)
        self.relation_target_index[target_entity.id].add(relation_id)
        self.relation_type_index[relation_type].add(relation_id)
        
        # Add reverse relation if bidirectional
        if bidirectional:
            reverse_type = relation.reverse_relation_type()
            if reverse_type:
                self.add_relation(target_name, source_name, reverse_type, 
                                weight, confidence, properties, False)
            else:
                # Create symmetric relation
                self.add_relation(target_name, source_name, relation_type,
                                weight, confidence, properties, False)
        
        # Update metrics
        self.metrics["total_relations"] += 1
        self.metrics["relations_by_type"][relation_type.value] += 1
        self.metrics["last_updated"] = datetime.now()
        self._update_graph_metrics()
        
        logger.debug(f"Added relation: {relation}")
        return relation
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get an entity by name or alias"""
        entity_id = self.entity_name_index.get(name.lower())
        return self.entities.get(entity_id) if entity_id else None
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type"""
        entity_ids = self.entity_type_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def get_relations(self, source_name: Optional[str] = None,
                     target_name: Optional[str] = None,
                     relation_type: Optional[RelationType] = None) -> List[Relation]:
        """Get relations based on filtering criteria"""
        
        candidate_relation_ids = set(self.relations.keys())
        
        # Filter by source entity
        if source_name:
            source_entity = self.get_entity_by_name(source_name)
            if source_entity:
                source_relations = self.relation_source_index.get(source_entity.id, set())
                candidate_relation_ids &= source_relations
            else:
                return []
        
        # Filter by target entity
        if target_name:
            target_entity = self.get_entity_by_name(target_name)
            if target_entity:
                target_relations = self.relation_target_index.get(target_entity.id, set())
                candidate_relation_ids &= target_relations
            else:
                return []
        
        # Filter by relation type
        if relation_type:
            type_relations = self.relation_type_index.get(relation_type, set())
            candidate_relation_ids &= type_relations
        
        return [self.relations[rid] for rid in candidate_relation_ids if rid in self.relations]
    
    def get_neighbors(self, entity_name: str, 
                     relation_type: Optional[RelationType] = None,
                     direction: str = "both") -> List[Tuple[Entity, Relation]]:
        """Get neighboring entities and their connecting relations"""
        
        entity = self.get_entity_by_name(entity_name)
        if not entity:
            return []
        
        neighbors = []
        
        # Outgoing relations (entity as source)
        if direction in ["out", "both"]:
            outgoing_relation_ids = self.relation_source_index.get(entity.id, set())
            for rel_id in outgoing_relation_ids:
                if rel_id in self.relations:
                    relation = self.relations[rel_id]
                    if not relation_type or relation.relation_type == relation_type:
                        target_entity = self.entities.get(relation.target_entity)
                        if target_entity:
                            neighbors.append((target_entity, relation))
        
        # Incoming relations (entity as target)
        if direction in ["in", "both"]:
            incoming_relation_ids = self.relation_target_index.get(entity.id, set())
            for rel_id in incoming_relation_ids:
                if rel_id in self.relations:
                    relation = self.relations[rel_id]
                    if not relation_type or relation.relation_type == relation_type:
                        source_entity = self.entities.get(relation.source_entity)
                        if source_entity:
                            neighbors.append((source_entity, relation))
        
        return neighbors
    
    def shortest_path(self, source_name: str, target_name: str,
                     max_depth: int = 6) -> Optional[List[Tuple[Entity, Relation]]]:
        """Find shortest path between two entities using BFS"""
        
        source_entity = self.get_entity_by_name(source_name)
        target_entity = self.get_entity_by_name(target_name)
        
        if not source_entity or not target_entity:
            return None
        
        if source_entity.id == target_entity.id:
            return [(source_entity, None)]
        
        # BFS to find shortest path
        queue = deque([(source_entity.id, [])])
        visited = {source_entity.id}
        
        for depth in range(max_depth):
            if not queue:
                break
                
            level_size = len(queue)
            for _ in range(level_size):
                current_id, path = queue.popleft()
                
                # Get neighbors
                neighbors = self.get_neighbors(self.entities[current_id].name)
                
                for neighbor_entity, relation in neighbors:
                    if neighbor_entity.id == target_entity.id:
                        # Found target
                        final_path = path + [(neighbor_entity, relation)]
                        return [(source_entity, None)] + final_path
                    
                    if neighbor_entity.id not in visited:
                        visited.add(neighbor_entity.id)
                        new_path = path + [(neighbor_entity, relation)]
                        queue.append((neighbor_entity.id, new_path))
        
        return None  # No path found within max_depth
    
    def find_similar_entities(self, entity_name: str, 
                            similarity_threshold: float = 0.7,
                            max_results: int = 10) -> List[Tuple[Entity, float]]:
        """Find entities similar to the given entity based on embeddings"""
        
        if not self.enable_embeddings:
            logger.warning("Embeddings not enabled for similarity search")
            return []
        
        target_entity = self.get_entity_by_name(entity_name)
        if not target_entity or target_entity.embedding is None:
            return []
        
        similar_entities = []
        target_embedding = target_entity.embedding
        
        for entity in self.entities.values():
            if entity.id == target_entity.id or entity.embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(target_embedding, entity.embedding)
            
            if similarity >= similarity_threshold:
                similar_entities.append((entity, similarity))
        
        # Sort by similarity and limit results
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        return similar_entities[:max_results]
    
    def get_subgraph(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """Extract a subgraph around an entity up to a certain depth"""
        
        center_entity = self.get_entity_by_name(entity_name)
        if not center_entity:
            return {"entities": [], "relations": []}
        
        visited_entities = {center_entity.id}
        visited_relations = set()
        current_level = [center_entity.id]
        
        for d in range(depth):
            next_level = []
            
            for entity_id in current_level:
                neighbors = self.get_neighbors(self.entities[entity_id].name)
                
                for neighbor_entity, relation in neighbors:
                    visited_relations.add(relation.id)
                    
                    if neighbor_entity.id not in visited_entities:
                        visited_entities.add(neighbor_entity.id)
                        next_level.append(neighbor_entity.id)
            
            current_level = next_level
        
        # Build subgraph
        subgraph_entities = [self.entities[eid].to_dict() for eid in visited_entities]
        subgraph_relations = [self.relations[rid].to_dict() for rid in visited_relations]
        
        return {
            "center_entity": center_entity.to_dict(),
            "depth": depth,
            "entities": subgraph_entities,
            "relations": subgraph_relations,
            "entity_count": len(subgraph_entities),
            "relation_count": len(subgraph_relations)
        }
    
    def query_by_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        """Query entities and relations using a simple pattern language"""
        # Simplified pattern matching - could be extended with more sophisticated parsing
        
        results = []
        
        # Pattern: "entity_name --relation_type-> ?target"
        if "--" in pattern and "->" in pattern:
            parts = pattern.split("--")
            if len(parts) == 2:
                source_part = parts[0].strip()
                relation_and_target = parts[1]
                
                if "->" in relation_and_target:
                    relation_part, target_part = relation_and_target.split("->", 1)
                    relation_type_str = relation_part.strip()
                    target_name = target_part.strip()
                    
                    # Find matching relation type
                    relation_type = None
                    for rt in RelationType:
                        if rt.value == relation_type_str:
                            relation_type = rt
                            break
                    
                    if relation_type:
                        if target_name == "?" or target_name.startswith("?"):
                            # Query for all targets
                            relations = self.get_relations(
                                source_name=source_part,
                                relation_type=relation_type
                            )
                            for relation in relations:
                                target_entity = self.entities.get(relation.target_entity)
                                if target_entity:
                                    results.append({
                                        "source": source_part,
                                        "relation": relation_type_str,
                                        "target": target_entity.name,
                                        "confidence": relation.confidence
                                    })
                        else:
                            # Specific source and target
                            relations = self.get_relations(
                                source_name=source_part,
                                target_name=target_name,
                                relation_type=relation_type
                            )
                            for relation in relations:
                                results.append({
                                    "source": source_part,
                                    "relation": relation_type_str,
                                    "target": target_name,
                                    "confidence": relation.confidence,
                                    "verified": True
                                })
        
        logger.info(f"Pattern query '{pattern}' returned {len(results)} results")
        return results
    
    def get_context(self, entity_name: str, context_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get contextual information about an entity for reasoning"""
        
        entity = self.get_entity_by_name(entity_name)
        if not entity:
            return {}
        
        context = {
            "entity": entity.to_dict(),
            "direct_relations": [],
            "properties": entity.properties,
            "type_hierarchy": [],
            "associations": []
        }
        
        # Get direct relations
        neighbors = self.get_neighbors(entity_name)
        for neighbor_entity, relation in neighbors:
            context["direct_relations"].append({
                "entity": neighbor_entity.to_dict(),
                "relation": relation.to_dict(),
                "direction": "outgoing" if relation.source_entity == entity.id else "incoming"
            })
        
        # Get type hierarchy (IS_A relations)
        type_relations = self.get_relations(
            source_name=entity_name,
            relation_type=RelationType.IS_A
        )
        for relation in type_relations:
            parent_entity = self.entities.get(relation.target_entity)
            if parent_entity:
                context["type_hierarchy"].append(parent_entity.name)
        
        # Get general associations
        associated_relations = self.get_relations(
            source_name=entity_name,
            relation_type=RelationType.ASSOCIATED_WITH
        )
        for relation in associated_relations:
            assoc_entity = self.entities.get(relation.target_entity)
            if assoc_entity:
                context["associations"].append(assoc_entity.name)
        
        return context
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _update_graph_metrics(self):
        """Update graph-level metrics"""
        total_entities = len(self.entities)
        total_relations = len(self.relations)
        
        # Calculate average degree
        if total_entities > 0:
            total_degree = 0
            for entity_id in self.entities.keys():
                outgoing = len(self.relation_source_index.get(entity_id, set()))
                incoming = len(self.relation_target_index.get(entity_id, set()))
                total_degree += outgoing + incoming
            
            self.metrics["average_entity_degree"] = total_degree / total_entities
        
        # Calculate graph density
        if total_entities > 1:
            max_possible_edges = total_entities * (total_entities - 1)
            self.metrics["graph_density"] = (total_relations * 2) / max_possible_edges
        
        self.metrics["total_entities"] = total_entities
        self.metrics["total_relations"] = total_relations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge graph statistics"""
        
        # Entity type distribution
        entity_type_stats = {}
        for entity_type, entity_ids in self.entity_type_index.items():
            entity_type_stats[entity_type.value] = len(entity_ids)
        
        # Relation type distribution
        relation_type_stats = {}
        for relation_type, relation_ids in self.relation_type_index.items():
            relation_type_stats[relation_type.value] = len(relation_ids)
        
        # Top connected entities
        entity_degrees = []
        for entity_id, entity in self.entities.items():
            outgoing = len(self.relation_source_index.get(entity_id, set()))
            incoming = len(self.relation_target_index.get(entity_id, set()))
            degree = outgoing + incoming
            entity_degrees.append((entity.name, degree))
        
        entity_degrees.sort(key=lambda x: x[1], reverse=True)
        top_connected = entity_degrees[:10]
        
        return {
            **self.metrics,
            "entity_types": entity_type_stats,
            "relation_types": relation_type_stats,
            "top_connected_entities": [
                {"name": name, "degree": degree} for name, degree in top_connected
            ],
            "graph_components": self._count_connected_components(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _count_connected_components(self) -> int:
        """Count the number of connected components in the graph"""
        visited = set()
        components = 0
        
        def dfs(entity_id: str):
            if entity_id in visited:
                return
            visited.add(entity_id)
            
            # Visit neighbors
            neighbors = self.get_neighbors(self.entities[entity_id].name)
            for neighbor_entity, _ in neighbors:
                if neighbor_entity.id not in visited:
                    dfs(neighbor_entity.id)
        
        for entity_id in self.entities.keys():
            if entity_id not in visited:
                dfs(entity_id)
                components += 1
        
        return components
    
    def export_graph(self, format: str = "json") -> Union[Dict[str, Any], str]:
        """Export the knowledge graph in various formats"""
        
        if format == "json":
            return {
                "entities": [entity.to_dict() for entity in self.entities.values()],
                "relations": [relation.to_dict() for relation in self.relations.values()],
                "statistics": self.get_statistics(),
                "exported_at": datetime.now().isoformat()
            }
        
        elif format == "networkx":
            # For NetworkX-compatible format
            return {
                "nodes": [
                    {
                        "id": entity.id,
                        "label": entity.name,
                        "type": entity.entity_type.value,
                        **entity.properties
                    }
                    for entity in self.entities.values()
                ],
                "edges": [
                    {
                        "source": relation.source_entity,
                        "target": relation.target_entity,
                        "type": relation.relation_type.value,
                        "weight": relation.weight,
                        **relation.properties
                    }
                    for relation in self.relations.values()
                ]
            }
        
        elif format == "cypher":
            # For Neo4j Cypher format
            cypher_statements = []
            
            # Create entities
            for entity in self.entities.values():
                props = ", ".join([f"{k}: '{v}'" for k, v in entity.properties.items()])
                cypher_statements.append(
                    f"CREATE (e{entity.id[:8]}:{entity.entity_type.value.capitalize()} "
                    f"{{name: '{entity.name}', id: '{entity.id}'{', ' + props if props else ''}}})"
                )
            
            # Create relations
            for relation in self.relations.values():
                src_short = relation.source_entity[:8]
                tgt_short = relation.target_entity[:8]
                rel_type = relation.relation_type.value.upper()
                cypher_statements.append(
                    f"CREATE (e{src_short})-[:{rel_type} "
                    f"{{weight: {relation.weight}, confidence: {relation.confidence}}}]->(e{tgt_short})"
                )
            
            return "\n".join(cypher_statements)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear(self):
        """Clear all entities and relations from the graph"""
        self.entities.clear()
        self.relations.clear()
        self.entity_name_index.clear()
        self.entity_type_index.clear()
        self.relation_source_index.clear()
        self.relation_target_index.clear()
        self.relation_type_index.clear()
        
        # Reset metrics
        self.metrics = {
            "total_entities": 0,
            "total_relations": 0,
            "entities_by_type": {et.value: 0 for et in EntityType},
            "relations_by_type": {rt.value: 0 for rt in RelationType},
            "average_entity_degree": 0.0,
            "graph_density": 0.0,
            "last_updated": datetime.now()
        }
        
        logger.info("🧹 Knowledge graph cleared")