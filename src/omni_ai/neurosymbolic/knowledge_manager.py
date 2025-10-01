"""
📚 KNOWLEDGE BASE MANAGEMENT SYSTEM
Advanced knowledge base management with domain templates, import/export, and validation
"""

import json
import yaml
import os
import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import hashlib
import shutil

from structlog import get_logger

from .symbolic_reasoner import SymbolicReasoner, Predicate, Rule
from .knowledge_graph import KnowledgeGraph, Entity, Relation, EntityType, RelationType

logger = get_logger()

class KnowledgeFormat(Enum):
    """Supported knowledge base formats"""
    JSON = "json"
    YAML = "yaml"
    SQLITE = "sqlite"
    TEXT = "text"
    RDF = "rdf"

class DomainType(Enum):
    """Pre-defined domain types with specialized templates"""
    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"
    SCIENTIFIC = "scientific"
    EDUCATIONAL = "educational"
    BUSINESS = "business"
    TECHNICAL = "technical"
    PHILOSOPHICAL = "philosophical"

@dataclass
class KnowledgeBaseMetadata:
    """Metadata for a knowledge base"""
    id: str
    name: str
    description: str
    domain: DomainType
    version: str
    created_at: datetime
    updated_at: datetime
    author: str
    tags: List[str] = field(default_factory=list)
    format: KnowledgeFormat = KnowledgeFormat.JSON
    facts_count: int = 0
    rules_count: int = 0
    entities_count: int = 0
    relations_count: int = 0
    checksum: str = ""

@dataclass
class KnowledgeBaseContent:
    """Complete knowledge base content"""
    metadata: KnowledgeBaseMetadata
    facts: List[Dict[str, Any]] = field(default_factory=list)
    rules: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)

class KnowledgeBaseTemplate:
    """Template for domain-specific knowledge bases"""
    
    def __init__(self, domain: DomainType):
        self.domain = domain
        self.template_facts = []
        self.template_rules = []
        self.template_entities = []
        self.template_relations = []
        self._initialize_template()
    
    def _initialize_template(self):
        """Initialize domain-specific templates"""
        if self.domain == DomainType.MEDICAL:
            self._init_medical_template()
        elif self.domain == DomainType.LEGAL:
            self._init_legal_template()
        elif self.domain == DomainType.SCIENTIFIC:
            self._init_scientific_template()
        elif self.domain == DomainType.EDUCATIONAL:
            self._init_educational_template()
        elif self.domain == DomainType.BUSINESS:
            self._init_business_template()
        elif self.domain == DomainType.TECHNICAL:
            self._init_technical_template()
        elif self.domain == DomainType.PHILOSOPHICAL:
            self._init_philosophical_template()
        else:
            self._init_general_template()
    
    def _init_medical_template(self):
        """Medical domain template"""
        self.template_facts = [
            "is_a(human, organism)",
            "has_property(human, cardiovascular_system)",
            "has_property(human, nervous_system)",
            "has_property(aspirin, pain_reliever)",
            "causes(bacteria, infection)",
            "treats(antibiotic, bacterial_infection)"
        ]
        
        self.template_rules = [
            "IF has_symptom(X, fever) AND has_symptom(X, cough) THEN likely_condition(X, respiratory_infection)",
            "IF is_a(X, antibiotic) AND has_condition(Y, bacterial_infection) THEN can_treat(X, Y)",
            "IF age(X, Y) AND Y > 65 THEN at_risk(X, age_related_conditions)"
        ]
        
        self.template_entities = [
            {"name": "Human", "type": EntityType.PERSON, "properties": {"organism_type": "mammal"}},
            {"name": "Heart", "type": EntityType.CONCEPT, "properties": {"organ_system": "cardiovascular"}},
            {"name": "Aspirin", "type": EntityType.CONCEPT, "properties": {"drug_class": "nsaid"}}
        ]
        
        self.template_relations = [
            ("Human", RelationType.HAS_PROPERTY, "Heart"),
            ("Aspirin", RelationType.ASSOCIATED_WITH, "Pain")
        ]
    
    def _init_legal_template(self):
        """Legal domain template"""
        self.template_facts = [
            "is_a(contract, legal_document)",
            "requires(contract, mutual_agreement)",
            "requires(contract, consideration)",
            "is_a(tort, civil_wrong)",
            "requires(crime, mens_rea)",
            "requires(crime, actus_reus)"
        ]
        
        self.template_rules = [
            "IF is_a(X, contract) AND has_property(X, breach) THEN liable_for(party, damages)",
            "IF committed(X, crime) AND proven(X, beyond_reasonable_doubt) THEN guilty(X)",
            "IF age(X, Y) AND Y < 18 THEN minor(X)"
        ]
        
        self.template_entities = [
            {"name": "Contract", "type": EntityType.CONCEPT, "properties": {"legal_type": "agreement"}},
            {"name": "Court", "type": EntityType.ORGANIZATION, "properties": {"jurisdiction": "legal"}},
            {"name": "Judge", "type": EntityType.PERSON, "properties": {"role": "judicial"}}
        ]
    
    def _init_scientific_template(self):
        """Scientific domain template"""
        self.template_facts = [
            "is_a(water, chemical_compound)",
            "has_formula(water, h2o)",
            "boils_at(water, 100_celsius)",
            "freezes_at(water, 0_celsius)",
            "is_a(photosynthesis, biological_process)",
            "requires(photosynthesis, sunlight)"
        ]
        
        self.template_rules = [
            "IF is_a(X, gas) AND temperature(X, T) AND T < boiling_point(X) THEN state(X, liquid)",
            "IF has_property(X, mass) AND has_property(X, acceleration) THEN has_property(X, force)",
            "IF is_a(X, hypothesis) AND supported_by(X, evidence) THEN theory(X)"
        ]
        
        self.template_entities = [
            {"name": "Water", "type": EntityType.CONCEPT, "properties": {"chemical_formula": "H2O"}},
            {"name": "Photosynthesis", "type": EntityType.CONCEPT, "properties": {"type": "biological"}},
            {"name": "Newton", "type": EntityType.PERSON, "properties": {"field": "physics"}}
        ]
    
    def _init_educational_template(self):
        """Educational domain template"""
        self.template_facts = [
            "is_a(student, learner)",
            "is_a(teacher, educator)",
            "teaches(teacher, subject)",
            "learns(student, subject)",
            "has_prerequisite(calculus, algebra)",
            "has_grade_level(algebra, high_school)"
        ]
        
        self.template_rules = [
            "IF learns(X, Y) AND practices(X, Y) THEN improves_skill(X, Y)",
            "IF has_prerequisite(X, Y) AND not_completed(student, Y) THEN cannot_take(student, X)",
            "IF grade(student, X) >= passing_grade THEN completed(student, X)"
        ]
        
        self.template_entities = [
            {"name": "Student", "type": EntityType.PERSON, "properties": {"role": "learner"}},
            {"name": "Mathematics", "type": EntityType.CONCEPT, "properties": {"subject_area": "stem"}},
            {"name": "University", "type": EntityType.ORGANIZATION, "properties": {"type": "educational"}}
        ]
    
    def _init_business_template(self):
        """Business domain template"""
        self.template_facts = [
            "is_a(company, organization)",
            "has_role(ceo, leadership)",
            "generates(company, revenue)",
            "has_expense(company, operating_costs)",
            "serves(company, customers)",
            "competes_with(company_a, company_b)"
        ]
        
        self.template_rules = [
            "IF revenue(X, R) AND expenses(X, E) AND R > E THEN profitable(X)",
            "IF satisfies(company, customer_needs) THEN likely_success(company)",
            "IF market_share(X, Y) AND Y > 50 THEN market_leader(X)"
        ]
        
        self.template_entities = [
            {"name": "Corporation", "type": EntityType.ORGANIZATION, "properties": {"business_type": "for_profit"}},
            {"name": "CEO", "type": EntityType.PERSON, "properties": {"role": "executive"}},
            {"name": "Market", "type": EntityType.CONCEPT, "properties": {"type": "economic"}}
        ]
    
    def _init_technical_template(self):
        """Technical domain template"""
        self.template_facts = [
            "is_a(python, programming_language)",
            "supports(python, object_oriented_programming)",
            "is_a(database, data_storage)",
            "uses(web_application, database)",
            "requires(software, hardware)",
            "connects_to(client, server)"
        ]
        
        self.template_rules = [
            "IF is_a(X, programming_language) AND has_feature(X, garbage_collection) THEN memory_managed(X)",
            "IF load(system, X) AND X > capacity THEN performance_degraded(system)",
            "IF version(software, X) AND X < latest_version THEN needs_update(software)"
        ]
        
        self.template_entities = [
            {"name": "Python", "type": EntityType.CONCEPT, "properties": {"type": "programming_language"}},
            {"name": "Database", "type": EntityType.CONCEPT, "properties": {"category": "data_storage"}},
            {"name": "Server", "type": EntityType.CONCEPT, "properties": {"role": "computing_resource"}}
        ]
    
    def _init_philosophical_template(self):
        """Philosophical domain template"""
        self.template_facts = [
            "is_a(socrates, philosopher)",
            "believes(socrates, examined_life)",
            "is_a(consciousness, mental_state)",
            "has_property(human, free_will)",
            "questions(philosophy, existence)",
            "seeks(philosophy, truth)"
        ]
        
        self.template_rules = [
            "IF is_a(X, human) AND has_property(X, consciousness) THEN capable_of(X, thought)",
            "IF questions(X, fundamental_nature) THEN philosophical_inquiry(X)",
            "IF seeks(X, wisdom) AND practices(X, philosophy) THEN philosopher(X)"
        ]
        
        self.template_entities = [
            {"name": "Socrates", "type": EntityType.PERSON, "properties": {"era": "ancient_greek"}},
            {"name": "Consciousness", "type": EntityType.CONCEPT, "properties": {"domain": "philosophy_of_mind"}},
            {"name": "Ethics", "type": EntityType.CONCEPT, "properties": {"branch": "moral_philosophy"}}
        ]
    
    def _init_general_template(self):
        """General domain template"""
        self.template_facts = [
            "is_a(cat, animal)",
            "is_a(dog, animal)",
            "has_property(animal, living)",
            "has_property(human, intelligent)",
            "lives_in(fish, water)",
            "needs(plant, sunlight)"
        ]
        
        self.template_rules = [
            "IF is_a(X, animal) THEN living(X)",
            "IF is_a(X, mammal) THEN warm_blooded(X)",
            "IF needs(X, Y) AND lacks(X, Y) THEN vulnerable(X)"
        ]
        
        self.template_entities = [
            {"name": "Animal", "type": EntityType.CONCEPT, "properties": {"category": "living_being"}},
            {"name": "Human", "type": EntityType.PERSON, "properties": {"species": "homo_sapiens"}},
            {"name": "Earth", "type": EntityType.LOCATION, "properties": {"type": "planet"}}
        ]

class KnowledgeBaseManager:
    """📚 Comprehensive Knowledge Base Management System"""
    
    def __init__(self, base_path: str = "knowledge_bases"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize storage paths
        self.data_path = self.base_path / "data"
        self.templates_path = self.base_path / "templates"
        self.exports_path = self.base_path / "exports"
        self.backups_path = self.base_path / "backups"
        
        for path in [self.data_path, self.templates_path, self.exports_path, self.backups_path]:
            path.mkdir(exist_ok=True)
        
        # Initialize index database
        self.index_db_path = self.base_path / "index.db"
        self._init_index_database()
        
        # Cache for loaded knowledge bases
        self.loaded_kb_cache = {}
        
        logger.info(f"📚 Knowledge Base Manager initialized at: {self.base_path}")
    
    def _init_index_database(self):
        """Initialize SQLite database for knowledge base indexing"""
        with sqlite3.connect(self.index_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    domain TEXT,
                    version TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    author TEXT,
                    format TEXT,
                    file_path TEXT,
                    facts_count INTEGER,
                    rules_count INTEGER,
                    entities_count INTEGER,
                    relations_count INTEGER,
                    checksum TEXT,
                    tags TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_dependencies (
                    kb_id TEXT,
                    depends_on TEXT,
                    dependency_type TEXT,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases (id),
                    FOREIGN KEY (depends_on) REFERENCES knowledge_bases (id)
                )
            """)
            conn.commit()
    
    def create_knowledge_base(self, 
                            name: str, 
                            description: str,
                            domain: DomainType = DomainType.GENERAL,
                            author: str = "system",
                            use_template: bool = True) -> KnowledgeBaseContent:
        """Create a new knowledge base, optionally from a domain template"""
        
        kb_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        metadata = KnowledgeBaseMetadata(
            id=kb_id,
            name=name,
            description=description,
            domain=domain,
            version="1.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            author=author,
            tags=[domain.value]
        )
        
        kb_content = KnowledgeBaseContent(metadata=metadata)
        
        # Apply domain template if requested
        if use_template:
            template = KnowledgeBaseTemplate(domain)
            
            # Add template facts
            for fact_str in template.template_facts:
                kb_content.facts.append({
                    "content": fact_str,
                    "confidence": 1.0,
                    "source": "template",
                    "added_at": datetime.now().isoformat()
                })
            
            # Add template rules
            for rule_str in template.template_rules:
                kb_content.rules.append({
                    "content": rule_str,
                    "priority": 1,
                    "confidence": 1.0,
                    "source": "template",
                    "added_at": datetime.now().isoformat()
                })
            
            # Add template entities
            for entity_data in template.template_entities:
                kb_content.entities.append({
                    **entity_data,
                    "source": "template",
                    "added_at": datetime.now().isoformat()
                })
            
            # Add template relations
            for rel_data in template.template_relations:
                kb_content.relations.append({
                    "source": rel_data[0],
                    "relation_type": rel_data[1].value if hasattr(rel_data[1], 'value') else str(rel_data[1]),
                    "target": rel_data[2],
                    "source_origin": "template",
                    "added_at": datetime.now().isoformat()
                })
        
        # Update metadata counts
        metadata.facts_count = len(kb_content.facts)
        metadata.rules_count = len(kb_content.rules)
        metadata.entities_count = len(kb_content.entities)
        metadata.relations_count = len(kb_content.relations)
        
        logger.info(f"📚 Created knowledge base: {name} ({domain.value})")
        return kb_content
    
    def save_knowledge_base(self, 
                          kb_content: KnowledgeBaseContent, 
                          format: KnowledgeFormat = KnowledgeFormat.JSON) -> str:
        """Save knowledge base to disk"""
        
        # Update metadata
        kb_content.metadata.updated_at = datetime.now()
        kb_content.metadata.format = format
        kb_content.metadata.facts_count = len(kb_content.facts)
        kb_content.metadata.rules_count = len(kb_content.rules)
        kb_content.metadata.entities_count = len(kb_content.entities)
        kb_content.metadata.relations_count = len(kb_content.relations)
        
        # Generate filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in kb_content.metadata.name)
        if format == KnowledgeFormat.JSON:
            filename = f"{safe_name}_{kb_content.metadata.id[:8]}.json"
            file_path = self.data_path / filename
            
            # Convert to serializable format
            data = asdict(kb_content)
            data['metadata']['created_at'] = data['metadata']['created_at'].isoformat()
            data['metadata']['updated_at'] = data['metadata']['updated_at'].isoformat()
            data['metadata']['domain'] = data['metadata']['domain'].value
            data['metadata']['format'] = data['metadata']['format'].value
            
            # Convert EntityType enums to strings in entities
            for entity in data.get('entities', []):
                if hasattr(entity.get('type'), 'value'):
                    entity['type'] = entity['type'].value
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == KnowledgeFormat.YAML:
            filename = f"{safe_name}_{kb_content.metadata.id[:8]}.yaml"
            file_path = self.data_path / filename
            
            data = asdict(kb_content)
            data['metadata']['created_at'] = data['metadata']['created_at'].isoformat()
            data['metadata']['updated_at'] = data['metadata']['updated_at'].isoformat()
            data['metadata']['domain'] = data['metadata']['domain'].value
            data['metadata']['format'] = data['metadata']['format'].value
            
            # Convert EntityType enums to strings in entities
            for entity in data.get('entities', []):
                if hasattr(entity.get('type'), 'value'):
                    entity['type'] = entity['type'].value
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        elif format == KnowledgeFormat.SQLITE:
            filename = f"{safe_name}_{kb_content.metadata.id[:8]}.db"
            file_path = self.data_path / filename
            self._save_to_sqlite(kb_content, file_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Calculate checksum
        kb_content.metadata.checksum = self._calculate_checksum(file_path)
        
        # Update index
        self._update_index(kb_content, str(file_path))
        
        # Update cache
        self.loaded_kb_cache[kb_content.metadata.id] = kb_content
        
        logger.info(f"📚 Saved knowledge base: {kb_content.metadata.name} -> {file_path}")
        return str(file_path)
    
    def load_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBaseContent]:
        """Load knowledge base by ID"""
        
        # Check cache first
        if kb_id in self.loaded_kb_cache:
            return self.loaded_kb_cache[kb_id]
        
        # Query index for file path
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path, format FROM knowledge_bases WHERE id = ?",
                (kb_id,)
            )
            result = cursor.fetchone()
        
        if not result:
            logger.warning(f"📚 Knowledge base not found: {kb_id}")
            return None
        
        file_path, format_str = result
        format = KnowledgeFormat(format_str)
        
        try:
            if format == KnowledgeFormat.JSON:
                kb_content = self._load_from_json(Path(file_path))
            elif format == KnowledgeFormat.YAML:
                kb_content = self._load_from_yaml(Path(file_path))
            elif format == KnowledgeFormat.SQLITE:
                kb_content = self._load_from_sqlite(Path(file_path))
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Update cache
            self.loaded_kb_cache[kb_id] = kb_content
            
            logger.info(f"📚 Loaded knowledge base: {kb_content.metadata.name}")
            return kb_content
            
        except Exception as e:
            logger.error(f"📚 Failed to load knowledge base {kb_id}: {e}")
            return None
    
    def _load_from_json(self, file_path: Path) -> KnowledgeBaseContent:
        """Load knowledge base from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert back to proper types
        metadata_data = data['metadata']
        metadata_data['created_at'] = datetime.fromisoformat(metadata_data['created_at'])
        metadata_data['updated_at'] = datetime.fromisoformat(metadata_data['updated_at'])
        metadata_data['domain'] = DomainType(metadata_data['domain'])
        metadata_data['format'] = KnowledgeFormat(metadata_data['format'])
        
        metadata = KnowledgeBaseMetadata(**metadata_data)
        
        return KnowledgeBaseContent(
            metadata=metadata,
            facts=data.get('facts', []),
            rules=data.get('rules', []),
            entities=data.get('entities', []),
            relations=data.get('relations', []),
            custom_data=data.get('custom_data', {})
        )
    
    def _load_from_yaml(self, file_path: Path) -> KnowledgeBaseContent:
        """Load knowledge base from YAML file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Convert back to proper types (similar to JSON)
        metadata_data = data['metadata']
        metadata_data['created_at'] = datetime.fromisoformat(metadata_data['created_at'])
        metadata_data['updated_at'] = datetime.fromisoformat(metadata_data['updated_at'])
        metadata_data['domain'] = DomainType(metadata_data['domain'])
        metadata_data['format'] = KnowledgeFormat(metadata_data['format'])
        
        metadata = KnowledgeBaseMetadata(**metadata_data)
        
        return KnowledgeBaseContent(
            metadata=metadata,
            facts=data.get('facts', []),
            rules=data.get('rules', []),
            entities=data.get('entities', []),
            relations=data.get('relations', []),
            custom_data=data.get('custom_data', {})
        )
    
    def _save_to_sqlite(self, kb_content: KnowledgeBaseContent, file_path: Path):
        """Save knowledge base to SQLite database"""
        with sqlite3.connect(file_path) as conn:
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    confidence REAL,
                    source TEXT,
                    added_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    priority INTEGER,
                    confidence REAL,
                    source TEXT,
                    added_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    type TEXT,
                    properties TEXT,
                    source TEXT,
                    added_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    relation_type TEXT,
                    target TEXT,
                    source_origin TEXT,
                    added_at TEXT
                )
            """)
            
            # Insert metadata
            metadata_dict = asdict(kb_content.metadata)
            for key, value in metadata_dict.items():
                if isinstance(value, (datetime, DomainType, KnowledgeFormat)):
                    value = str(value)
                elif isinstance(value, list):
                    value = json.dumps(value)
                
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (key, str(value))
                )
            
            # Insert facts
            for fact in kb_content.facts:
                conn.execute(
                    "INSERT INTO facts (content, confidence, source, added_at) VALUES (?, ?, ?, ?)",
                    (fact['content'], fact['confidence'], fact['source'], fact['added_at'])
                )
            
            # Insert rules
            for rule in kb_content.rules:
                conn.execute(
                    "INSERT INTO rules (content, priority, confidence, source, added_at) VALUES (?, ?, ?, ?, ?)",
                    (rule['content'], rule['priority'], rule['confidence'], rule['source'], rule['added_at'])
                )
            
            # Insert entities
            for entity in kb_content.entities:
                conn.execute(
                    "INSERT INTO entities (name, type, properties, source, added_at) VALUES (?, ?, ?, ?, ?)",
                    (entity['name'], str(entity['type']), json.dumps(entity.get('properties', {})), 
                     entity['source'], entity['added_at'])
                )
            
            # Insert relations
            for relation in kb_content.relations:
                conn.execute(
                    "INSERT INTO relations (source, relation_type, target, source_origin, added_at) VALUES (?, ?, ?, ?, ?)",
                    (relation['source'], relation['relation_type'], relation['target'], 
                     relation['source_origin'], relation['added_at'])
                )
            
            conn.commit()
    
    def _load_from_sqlite(self, file_path: Path) -> KnowledgeBaseContent:
        """Load knowledge base from SQLite database"""
        with sqlite3.connect(file_path) as conn:
            # Load metadata
            cursor = conn.execute("SELECT key, value FROM metadata")
            metadata_dict = dict(cursor.fetchall())
            
            # Convert metadata back to proper types
            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
            metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
            metadata_dict['domain'] = DomainType(metadata_dict['domain'])
            metadata_dict['format'] = KnowledgeFormat(metadata_dict['format'])
            metadata_dict['tags'] = json.loads(metadata_dict.get('tags', '[]'))
            
            # Convert numeric fields
            for field in ['facts_count', 'rules_count', 'entities_count', 'relations_count']:
                if field in metadata_dict:
                    metadata_dict[field] = int(metadata_dict[field])
            
            metadata = KnowledgeBaseMetadata(**metadata_dict)
            
            # Load facts
            cursor = conn.execute("SELECT content, confidence, source, added_at FROM facts")
            facts = [
                {
                    'content': row[0],
                    'confidence': row[1],
                    'source': row[2],
                    'added_at': row[3]
                }
                for row in cursor.fetchall()
            ]
            
            # Load rules
            cursor = conn.execute("SELECT content, priority, confidence, source, added_at FROM rules")
            rules = [
                {
                    'content': row[0],
                    'priority': row[1],
                    'confidence': row[2],
                    'source': row[3],
                    'added_at': row[4]
                }
                for row in cursor.fetchall()
            ]
            
            # Load entities
            cursor = conn.execute("SELECT name, type, properties, source, added_at FROM entities")
            entities = [
                {
                    'name': row[0],
                    'type': row[1],
                    'properties': json.loads(row[2]),
                    'source': row[3],
                    'added_at': row[4]
                }
                for row in cursor.fetchall()
            ]
            
            # Load relations
            cursor = conn.execute("SELECT source, relation_type, target, source_origin, added_at FROM relations")
            relations = [
                {
                    'source': row[0],
                    'relation_type': row[1],
                    'target': row[2],
                    'source_origin': row[3],
                    'added_at': row[4]
                }
                for row in cursor.fetchall()
            ]
        
        return KnowledgeBaseContent(
            metadata=metadata,
            facts=facts,
            rules=rules,
            entities=entities,
            relations=relations
        )
    
    def _update_index(self, kb_content: KnowledgeBaseContent, file_path: str):
        """Update the index database with knowledge base information"""
        with sqlite3.connect(self.index_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_bases (
                    id, name, description, domain, version, created_at, updated_at, 
                    author, format, file_path, facts_count, rules_count, 
                    entities_count, relations_count, checksum, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kb_content.metadata.id,
                kb_content.metadata.name,
                kb_content.metadata.description,
                kb_content.metadata.domain.value,
                kb_content.metadata.version,
                kb_content.metadata.created_at.isoformat(),
                kb_content.metadata.updated_at.isoformat(),
                kb_content.metadata.author,
                kb_content.metadata.format.value,
                file_path,
                kb_content.metadata.facts_count,
                kb_content.metadata.rules_count,
                kb_content.metadata.entities_count,
                kb_content.metadata.relations_count,
                kb_content.metadata.checksum,
                json.dumps(kb_content.metadata.tags)
            ))
            conn.commit()
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def list_knowledge_bases(self, domain: Optional[DomainType] = None) -> List[Dict[str, Any]]:
        """List all available knowledge bases"""
        with sqlite3.connect(self.index_db_path) as conn:
            if domain:
                cursor = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE domain = ? ORDER BY updated_at DESC",
                    (domain.value,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM knowledge_bases ORDER BY updated_at DESC"
                )
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                kb_info = dict(zip(columns, row))
                kb_info['tags'] = json.loads(kb_info.get('tags', '[]'))
                results.append(kb_info)
            
            return results
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete a knowledge base"""
        # Get file path from index
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM knowledge_bases WHERE id = ?",
                (kb_id,)
            )
            result = cursor.fetchone()
        
        if not result:
            logger.warning(f"📚 Knowledge base not found for deletion: {kb_id}")
            return False
        
        file_path = Path(result[0])
        
        try:
            # Delete file
            if file_path.exists():
                file_path.unlink()
            
            # Remove from index
            with sqlite3.connect(self.index_db_path) as conn:
                conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
                conn.execute("DELETE FROM kb_dependencies WHERE kb_id = ? OR depends_on = ?", (kb_id, kb_id))
                conn.commit()
            
            # Remove from cache
            if kb_id in self.loaded_kb_cache:
                del self.loaded_kb_cache[kb_id]
            
            logger.info(f"📚 Deleted knowledge base: {kb_id}")
            return True
            
        except Exception as e:
            logger.error(f"📚 Failed to delete knowledge base {kb_id}: {e}")
            return False
    
    def backup_knowledge_base(self, kb_id: str) -> Optional[str]:
        """Create a backup of a knowledge base"""
        kb_content = self.load_knowledge_base(kb_id)
        if not kb_content:
            return None
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{kb_content.metadata.name}_backup_{timestamp}"
        backup_filename = f"{backup_name}_{kb_id[:8]}.json"
        backup_path = self.backups_path / backup_filename
        
        # Save as JSON
        data = asdict(kb_content)
        data['metadata']['created_at'] = data['metadata']['created_at'].isoformat()
        data['metadata']['updated_at'] = data['metadata']['updated_at'].isoformat()
        data['metadata']['domain'] = data['metadata']['domain'].value
        data['metadata']['format'] = data['metadata']['format'].value
        
        # Convert EntityType enums to strings in entities
        for entity in data.get('entities', []):
            if hasattr(entity.get('type'), 'value'):
                entity['type'] = entity['type'].value
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📚 Created backup: {backup_path}")
        return str(backup_path)
    
    def export_knowledge_base(self, 
                            kb_id: str, 
                            export_format: KnowledgeFormat,
                            include_metadata: bool = True) -> Optional[str]:
        """Export knowledge base to different formats"""
        kb_content = self.load_knowledge_base(kb_id)
        if not kb_content:
            return None
        
        # Create export filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in kb_content.metadata.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == KnowledgeFormat.TEXT:
            export_filename = f"{safe_name}_export_{timestamp}.txt"
            export_path = self.exports_path / export_filename
            self._export_to_text(kb_content, export_path, include_metadata)
        
        elif export_format == KnowledgeFormat.RDF:
            export_filename = f"{safe_name}_export_{timestamp}.rdf"
            export_path = self.exports_path / export_filename
            self._export_to_rdf(kb_content, export_path, include_metadata)
        
        else:
            # Use regular save for JSON/YAML/SQLite
            temp_content = kb_content
            temp_content.metadata.name = f"{kb_content.metadata.name}_export_{timestamp}"
            export_path = self.save_knowledge_base(temp_content, export_format)
            # Move to exports directory
            final_path = self.exports_path / Path(export_path).name
            try:
                shutil.move(export_path, final_path)
                export_path = str(final_path)
            except (PermissionError, OSError) as e:
                logger.warning(f"Could not move export file: {e}. Keeping at: {export_path}")
                # Keep the original path if move fails
        
        logger.info(f"📚 Exported knowledge base: {export_path}")
        return str(export_path)
    
    def _export_to_text(self, kb_content: KnowledgeBaseContent, file_path: Path, include_metadata: bool):
        """Export knowledge base to human-readable text format"""
        with open(file_path, 'w', encoding='utf-8') as f:
            if include_metadata:
                f.write(f"Knowledge Base: {kb_content.metadata.name}\n")
                f.write(f"Description: {kb_content.metadata.description}\n")
                f.write(f"Domain: {kb_content.metadata.domain.value}\n")
                f.write(f"Version: {kb_content.metadata.version}\n")
                f.write(f"Author: {kb_content.metadata.author}\n")
                f.write(f"Created: {kb_content.metadata.created_at}\n")
                f.write(f"Updated: {kb_content.metadata.updated_at}\n")
                f.write(f"Tags: {', '.join(kb_content.metadata.tags)}\n")
                f.write("\n" + "="*60 + "\n\n")
            
            if kb_content.facts:
                f.write("FACTS:\n")
                f.write("-" * 40 + "\n")
                for i, fact in enumerate(kb_content.facts, 1):
                    f.write(f"{i:3d}. {fact['content']}\n")
                    if fact.get('confidence', 1.0) < 1.0:
                        f.write(f"     Confidence: {fact['confidence']:.3f}\n")
                f.write("\n")
            
            if kb_content.rules:
                f.write("RULES:\n")
                f.write("-" * 40 + "\n")
                for i, rule in enumerate(kb_content.rules, 1):
                    f.write(f"{i:3d}. {rule['content']}\n")
                    if rule.get('priority', 1) != 1:
                        f.write(f"     Priority: {rule['priority']}\n")
                    if rule.get('confidence', 1.0) < 1.0:
                        f.write(f"     Confidence: {rule['confidence']:.3f}\n")
                f.write("\n")
            
            if kb_content.entities:
                f.write("ENTITIES:\n")
                f.write("-" * 40 + "\n")
                for i, entity in enumerate(kb_content.entities, 1):
                    f.write(f"{i:3d}. {entity['name']} ({entity['type']})\n")
                    if entity.get('properties'):
                        for key, value in entity['properties'].items():
                            f.write(f"     {key}: {value}\n")
                f.write("\n")
            
            if kb_content.relations:
                f.write("RELATIONS:\n")
                f.write("-" * 40 + "\n")
                for i, relation in enumerate(kb_content.relations, 1):
                    f.write(f"{i:3d}. {relation['source']} --{relation['relation_type']}--> {relation['target']}\n")
    
    def _export_to_rdf(self, kb_content: KnowledgeBaseContent, file_path: Path, include_metadata: bool):
        """Export knowledge base to RDF format (simplified)"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n')
            f.write('         xmlns:kb="http://omni-ai.org/knowledge-base#"\n')
            f.write('         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">\n\n')
            
            if include_metadata:
                f.write(f'  <kb:KnowledgeBase rdf:about="#{kb_content.metadata.id}">\n')
                f.write(f'    <rdfs:label>{kb_content.metadata.name}</rdfs:label>\n')
                f.write(f'    <rdfs:comment>{kb_content.metadata.description}</rdfs:comment>\n')
                f.write(f'    <kb:domain>{kb_content.metadata.domain.value}</kb:domain>\n')
                f.write(f'    <kb:version>{kb_content.metadata.version}</kb:version>\n')
                f.write(f'    <kb:author>{kb_content.metadata.author}</kb:author>\n')
                f.write(f'  </kb:KnowledgeBase>\n\n')
            
            # Export facts as RDF statements
            for i, fact in enumerate(kb_content.facts):
                f.write(f'  <kb:Fact rdf:ID="fact_{i}">\n')
                f.write(f'    <kb:content>{fact["content"]}</kb:content>\n')
                f.write(f'    <kb:confidence>{fact.get("confidence", 1.0)}</kb:confidence>\n')
                f.write(f'  </kb:Fact>\n')
            
            # Export entities
            for i, entity in enumerate(kb_content.entities):
                entity_id = entity['name'].replace(' ', '_')
                f.write(f'  <kb:Entity rdf:ID="{entity_id}">\n')
                f.write(f'    <rdfs:label>{entity["name"]}</rdfs:label>\n')
                f.write(f'    <kb:type>{entity["type"]}</kb:type>\n')
                for key, value in entity.get('properties', {}).items():
                    f.write(f'    <kb:{key}>{value}</kb:{key}>\n')
                f.write(f'  </kb:Entity>\n')
            
            f.write('</rdf:RDF>\n')
    
    async def import_from_reasoner(self, 
                                 reasoner: SymbolicReasoner,
                                 name: str,
                                 description: str = "Imported from symbolic reasoner",
                                 domain: DomainType = DomainType.GENERAL) -> KnowledgeBaseContent:
        """Import knowledge from a symbolic reasoner"""
        
        kb_content = self.create_knowledge_base(
            name=name,
            description=description,
            domain=domain,
            use_template=False
        )
        
        # Import facts
        for fact in reasoner.knowledge_base.facts:
            kb_content.facts.append({
                "content": str(fact),
                "confidence": fact.confidence,
                "source": "reasoner_import",
                "added_at": datetime.now().isoformat()
            })
        
        # Import rules
        for rule in reasoner.knowledge_base.rules:
            kb_content.rules.append({
                "content": str(rule),
                "priority": rule.priority,
                "confidence": rule.confidence,
                "source": "reasoner_import",
                "added_at": datetime.now().isoformat()
            })
        
        # Update counts
        kb_content.metadata.facts_count = len(kb_content.facts)
        kb_content.metadata.rules_count = len(kb_content.rules)
        
        logger.info(f"📚 Imported from reasoner: {name}")
        return kb_content
    
    async def import_from_knowledge_graph(self,
                                        kg: KnowledgeGraph,
                                        name: str,
                                        description: str = "Imported from knowledge graph",
                                        domain: DomainType = DomainType.GENERAL) -> KnowledgeBaseContent:
        """Import knowledge from a knowledge graph"""
        
        kb_content = self.create_knowledge_base(
            name=name,
            description=description,
            domain=domain,
            use_template=False
        )
        
        # Import entities
        for entity_id, entity in kg.entities.items():
            kb_content.entities.append({
                "name": entity.name,
                "type": entity.entity_type.value,
                "properties": entity.properties,
                "source": "kg_import",
                "added_at": datetime.now().isoformat()
            })
        
        # Import relations
        for relation_id, relation in kg.relations.items():
            kb_content.relations.append({
                "source": relation.source_entity,
                "relation_type": relation.relation_type.value,
                "target": relation.target_entity,
                "source_origin": "kg_import",
                "added_at": datetime.now().isoformat()
            })
        
        # Update counts
        kb_content.metadata.entities_count = len(kb_content.entities)
        kb_content.metadata.relations_count = len(kb_content.relations)
        
        logger.info(f"📚 Imported from knowledge graph: {name}")
        return kb_content
    
    async def load_into_reasoner(self, kb_id: str, reasoner: SymbolicReasoner) -> bool:
        """Load knowledge base content into a symbolic reasoner"""
        kb_content = self.load_knowledge_base(kb_id)
        if not kb_content:
            return False
        
        try:
            # Load facts
            for fact_data in kb_content.facts:
                reasoner.add_fact(fact_data['content'], fact_data.get('confidence', 1.0))
            
            # Load rules
            for rule_data in kb_content.rules:
                reasoner.add_rule(
                    rule_data['content'],
                    rule_data.get('priority', 1),
                    rule_data.get('confidence', 1.0)
                )
            
            logger.info(f"📚 Loaded KB {kb_content.metadata.name} into reasoner")
            return True
            
        except Exception as e:
            logger.error(f"📚 Failed to load KB into reasoner: {e}")
            return False
    
    async def load_into_knowledge_graph(self, kb_id: str, kg: KnowledgeGraph) -> bool:
        """Load knowledge base content into a knowledge graph"""
        kb_content = self.load_knowledge_base(kb_id)
        if not kb_content:
            return False
        
        try:
            # Load entities
            entity_id_map = {}
            for entity_data in kb_content.entities:
                entity_type = EntityType(entity_data['type']) if isinstance(entity_data['type'], str) else entity_data['type']
                entity_id = kg.add_entity(
                    entity_data['name'],
                    entity_type,
                    entity_data.get('properties', {})
                )
                entity_id_map[entity_data['name']] = entity_id
            
            # Load relations
            for relation_data in kb_content.relations:
                source_id = entity_id_map.get(relation_data['source'])
                target_id = entity_id_map.get(relation_data['target'])
                
                if source_id and target_id:
                    relation_type = RelationType(relation_data['relation_type']) if isinstance(relation_data['relation_type'], str) else relation_data['relation_type']
                    kg.add_relation(source_id, relation_type, target_id)
            
            logger.info(f"📚 Loaded KB {kb_content.metadata.name} into knowledge graph")
            return True
            
        except Exception as e:
            logger.error(f"📚 Failed to load KB into knowledge graph: {e}")
            return False
    
    def search_knowledge_bases(self, 
                             query: str,
                             domain: Optional[DomainType] = None,
                             tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search knowledge bases by name, description, or tags"""
        with sqlite3.connect(self.index_db_path) as conn:
            sql = """
                SELECT * FROM knowledge_bases 
                WHERE (name LIKE ? OR description LIKE ?)
            """
            params = [f"%{query}%", f"%{query}%"]
            
            if domain:
                sql += " AND domain = ?"
                params.append(domain.value)
            
            if tags:
                # Simple tag search (could be improved with proper JSON querying)
                for tag in tags:
                    sql += " AND tags LIKE ?"
                    params.append(f"%{tag}%")
            
            sql += " ORDER BY updated_at DESC"
            
            cursor = conn.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                kb_info = dict(zip(columns, row))
                kb_info['tags'] = json.loads(kb_info.get('tags', '[]'))
                results.append(kb_info)
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base system"""
        with sqlite3.connect(self.index_db_path) as conn:
            # Basic counts
            cursor = conn.execute("SELECT COUNT(*) FROM knowledge_bases")
            total_kbs = cursor.fetchone()[0]
            
            # Domain distribution
            cursor = conn.execute("SELECT domain, COUNT(*) FROM knowledge_bases GROUP BY domain")
            domain_dist = dict(cursor.fetchall())
            
            # Format distribution
            cursor = conn.execute("SELECT format, COUNT(*) FROM knowledge_bases GROUP BY format")
            format_dist = dict(cursor.fetchall())
            
            # Size statistics
            cursor = conn.execute("""
                SELECT 
                    SUM(facts_count) as total_facts,
                    SUM(rules_count) as total_rules,
                    SUM(entities_count) as total_entities,
                    SUM(relations_count) as total_relations,
                    AVG(facts_count) as avg_facts,
                    AVG(rules_count) as avg_rules
                FROM knowledge_bases
            """)
            stats = cursor.fetchone()
            
            # Storage information
            storage_size = sum(f.stat().st_size for f in self.data_path.rglob('*') if f.is_file())
            backup_size = sum(f.stat().st_size for f in self.backups_path.rglob('*') if f.is_file())
            
            return {
                "total_knowledge_bases": total_kbs,
                "domain_distribution": domain_dist,
                "format_distribution": format_dist,
                "content_statistics": {
                    "total_facts": stats[0] or 0,
                    "total_rules": stats[1] or 0,
                    "total_entities": stats[2] or 0,
                    "total_relations": stats[3] or 0,
                    "average_facts_per_kb": round(stats[4] or 0, 2),
                    "average_rules_per_kb": round(stats[5] or 0, 2)
                },
                "storage": {
                    "data_size_bytes": storage_size,
                    "backup_size_bytes": backup_size,
                    "total_size_bytes": storage_size + backup_size
                },
                "cache_status": {
                    "loaded_kbs": len(self.loaded_kb_cache),
                    "cache_hit_potential": len(self.loaded_kb_cache) / max(total_kbs, 1)
                }
            }


# Factory functions for easy usage
def create_knowledge_manager(base_path: str = "knowledge_bases") -> KnowledgeBaseManager:
    """Create a knowledge base manager"""
    return KnowledgeBaseManager(base_path)

def get_domain_template(domain: DomainType) -> KnowledgeBaseTemplate:
    """Get a domain-specific template"""
    return KnowledgeBaseTemplate(domain)