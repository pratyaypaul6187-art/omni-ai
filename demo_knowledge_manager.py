#!/usr/bin/env python3
"""
🧠 KNOWLEDGE BASE MANAGEMENT SYSTEM DEMO
Comprehensive demonstration of the knowledge base management capabilities
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omni_ai.neurosymbolic.knowledge_manager import (
    KnowledgeBaseManager, DomainType, KnowledgeFormat,
    create_knowledge_manager, get_domain_template
)
from omni_ai.neurosymbolic.symbolic_reasoner import SymbolicReasoner
from omni_ai.neurosymbolic.knowledge_graph import KnowledgeGraph

def print_section(title: str, emoji: str = "🔷"):
    """Print a formatted section header"""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n📌 {title}")
    print("-" * (len(title) + 4))

async def demo_knowledge_base_creation():
    """Demonstrate creating knowledge bases with different domain templates"""
    print_section("KNOWLEDGE BASE CREATION", "🏗️")
    
    # Initialize knowledge manager
    kb_manager = create_knowledge_manager("demo_knowledge_bases")
    
    # Test different domain templates
    domains_to_test = [
        (DomainType.MEDICAL, "Medical Knowledge", "Healthcare and medical information"),
        (DomainType.SCIENTIFIC, "Science Facts", "Scientific principles and discoveries"),
        (DomainType.TECHNICAL, "Tech Stack", "Programming and technical knowledge"),
        (DomainType.PHILOSOPHICAL, "Philosophy", "Philosophical concepts and thinkers"),
        (DomainType.BUSINESS, "Business Logic", "Business processes and concepts")
    ]
    
    created_kbs = []
    
    for domain, name, description in domains_to_test:
        print_subsection(f"Creating {domain.value.title()} Knowledge Base")
        
        # Create knowledge base with template
        kb_content = kb_manager.create_knowledge_base(
            name=name,
            description=description,
            domain=domain,
            author="Demo System",
            use_template=True
        )
        
        print(f"✅ Created KB: {kb_content.metadata.name}")
        print(f"   ID: {kb_content.metadata.id}")
        print(f"   Facts: {len(kb_content.facts)}")
        print(f"   Rules: {len(kb_content.rules)}")
        print(f"   Entities: {len(kb_content.entities)}")
        print(f"   Relations: {len(kb_content.relations)}")
        
        # Save in different formats
        json_path = kb_manager.save_knowledge_base(kb_content, KnowledgeFormat.JSON)
        print(f"   💾 Saved as JSON: {Path(json_path).name}")
        
        # Try YAML format for one of them
        if domain == DomainType.SCIENTIFIC:
            yaml_path = kb_manager.save_knowledge_base(kb_content, KnowledgeFormat.YAML)
            print(f"   💾 Also saved as YAML: {Path(yaml_path).name}")
        
        created_kbs.append(kb_content.metadata.id)
    
    return kb_manager, created_kbs

async def demo_knowledge_base_operations(kb_manager, kb_ids):
    """Demonstrate various knowledge base operations"""
    print_section("KNOWLEDGE BASE OPERATIONS", "⚙️")
    
    # List all knowledge bases
    print_subsection("Listing Knowledge Bases")
    all_kbs = kb_manager.list_knowledge_bases()
    print(f"📊 Total knowledge bases: {len(all_kbs)}")
    
    for kb in all_kbs:
        print(f"   • {kb['name']} ({kb['domain']}) - {kb['facts_count']} facts, {kb['rules_count']} rules")
    
    # Search functionality
    print_subsection("Searching Knowledge Bases")
    
    search_results = kb_manager.search_knowledge_bases("medical")
    print(f"🔍 Search results for 'medical': {len(search_results)} found")
    for result in search_results:
        print(f"   • {result['name']}: {result['description']}")
    
    # Domain filtering
    tech_kbs = kb_manager.list_knowledge_bases(domain=DomainType.TECHNICAL)
    print(f"🔍 Technical knowledge bases: {len(tech_kbs)} found")
    for kb in tech_kbs:
        print(f"   • {kb['name']}")
    
    # Load and inspect a knowledge base
    if kb_ids:
        print_subsection("Loading and Inspecting Knowledge Base")
        kb_id = kb_ids[0]
        kb_content = kb_manager.load_knowledge_base(kb_id)
        
        if kb_content:
            print(f"📖 Loaded: {kb_content.metadata.name}")
            print(f"   Domain: {kb_content.metadata.domain.value}")
            print(f"   Created: {kb_content.metadata.created_at}")
            print(f"   Checksum: {kb_content.metadata.checksum[:12]}...")
            
            # Show some facts and rules
            if kb_content.facts:
                print(f"   📝 Sample Facts (showing first 3):")
                for i, fact in enumerate(kb_content.facts[:3]):
                    print(f"      {i+1}. {fact['content']}")
            
            if kb_content.rules:
                print(f"   📋 Sample Rules (showing first 2):")
                for i, rule in enumerate(kb_content.rules[:2]):
                    print(f"      {i+1}. {rule['content']}")

async def demo_import_export(kb_manager, kb_ids):
    """Demonstrate import/export functionality"""
    print_section("IMPORT/EXPORT OPERATIONS", "📤")
    
    if not kb_ids:
        print("⚠️  No knowledge bases available for export demo")
        return
    
    kb_id = kb_ids[0]
    
    # Create backup
    print_subsection("Creating Backup")
    backup_path = kb_manager.backup_knowledge_base(kb_id)
    if backup_path:
        print(f"💾 Backup created: {Path(backup_path).name}")
    
    # Export to different formats
    print_subsection("Exporting to Different Formats")
    
    # Export to text
    text_export = kb_manager.export_knowledge_base(kb_id, KnowledgeFormat.TEXT)
    if text_export:
        print(f"📄 Text export: {Path(text_export).name}")
        
        # Show a preview of the text export
        with open(text_export, 'r', encoding='utf-8') as f:
            preview = f.read(500)
            print(f"   Preview: {preview[:200]}...")
    
    # Export to RDF
    rdf_export = kb_manager.export_knowledge_base(kb_id, KnowledgeFormat.RDF)
    if rdf_export:
        print(f"🌐 RDF export: {Path(rdf_export).name}")
    
    # Export to SQLite
    sqlite_export = kb_manager.export_knowledge_base(kb_id, KnowledgeFormat.SQLITE)
    if sqlite_export:
        print(f"🗄️  SQLite export: {Path(sqlite_export).name}")

async def demo_integration_with_reasoner(kb_manager):
    """Demonstrate integration with symbolic reasoner"""
    print_section("REASONER INTEGRATION", "🧠")
    
    # Create a symbolic reasoner
    reasoner = SymbolicReasoner()
    
    # Add some facts and rules manually
    print_subsection("Adding Knowledge to Reasoner")
    reasoner.add_fact("is_a(socrates, human)", 1.0)
    reasoner.add_fact("is_a(plato, human)", 1.0)
    reasoner.add_fact("teacher_of(socrates, plato)", 0.9)
    reasoner.add_rule("IF is_a(X, human) THEN mortal(X)", 1, 1.0)
    reasoner.add_rule("IF teacher_of(X, Y) THEN influences(X, Y)", 1, 0.8)
    
    print(f"📝 Added {len(reasoner.knowledge_base.facts)} facts")
    print(f"📋 Added {len(reasoner.knowledge_base.rules)} rules")
    
    # Import from reasoner to knowledge base
    print_subsection("Importing from Reasoner to Knowledge Base")
    imported_kb = await kb_manager.import_from_reasoner(
        reasoner,
        "Philosophy from Reasoner",
        "Knowledge imported from symbolic reasoner",
        DomainType.PHILOSOPHICAL
    )
    
    # Save the imported knowledge base
    imported_path = kb_manager.save_knowledge_base(imported_kb)
    print(f"💾 Imported KB saved: {Path(imported_path).name}")
    print(f"   Facts imported: {len(imported_kb.facts)}")
    print(f"   Rules imported: {len(imported_kb.rules)}")
    
    # Load knowledge base back into a new reasoner
    print_subsection("Loading Knowledge Base into New Reasoner")
    new_reasoner = SymbolicReasoner()
    success = await kb_manager.load_into_reasoner(imported_kb.metadata.id, new_reasoner)
    
    if success:
        print(f"✅ Successfully loaded KB into new reasoner")
        print(f"   New reasoner now has {len(new_reasoner.knowledge_base.facts)} facts")
        print(f"   New reasoner now has {len(new_reasoner.knowledge_base.rules)} rules")
        
        # Test reasoning
        result = await new_reasoner.query("mortal(socrates)")
        print(f"   Query 'mortal(socrates)': {result}")

async def demo_integration_with_knowledge_graph(kb_manager):
    """Demonstrate integration with knowledge graph"""
    print_section("KNOWLEDGE GRAPH INTEGRATION", "🕸️")
    
    # Create a knowledge graph
    kg = KnowledgeGraph()
    
    # Add some entities and relations
    print_subsection("Building Knowledge Graph")
    from omni_ai.neurosymbolic.knowledge_graph import EntityType, RelationType
    
    # Add entities
    human_entity = kg.add_entity("Human", EntityType.CONCEPT, {"category": "species"})
    socrates_entity = kg.add_entity("Socrates", EntityType.PERSON, {"era": "ancient", "nationality": "greek"})
    philosophy_entity = kg.add_entity("Philosophy", EntityType.CONCEPT, {"field": "wisdom"})
    
    # Add relations
    kg.add_relation("Socrates", "Human", RelationType.IS_A)
    kg.add_relation("Socrates", "Philosophy", RelationType.PRACTICES)
    
    print(f"🕸️  Created knowledge graph with {len(kg.entities)} entities and {len(kg.relations)} relations")
    
    # Import from knowledge graph
    print_subsection("Importing from Knowledge Graph")
    kg_imported_kb = await kb_manager.import_from_knowledge_graph(
        kg,
        "Philosophy Graph",
        "Knowledge imported from knowledge graph",
        DomainType.PHILOSOPHICAL
    )
    
    # Save the imported knowledge base
    kg_path = kb_manager.save_knowledge_base(kg_imported_kb)
    print(f"💾 KG imported KB saved: {Path(kg_path).name}")
    print(f"   Entities imported: {len(kg_imported_kb.entities)}")
    print(f"   Relations imported: {len(kg_imported_kb.relations)}")
    
    # Load back into a new knowledge graph
    print_subsection("Loading Knowledge Base into New Knowledge Graph")
    new_kg = KnowledgeGraph()
    success = await kb_manager.load_into_knowledge_graph(kg_imported_kb.metadata.id, new_kg)
    
    if success:
        print(f"✅ Successfully loaded KB into new knowledge graph")
        print(f"   New KG now has {len(new_kg.entities)} entities")
        print(f"   New KG now has {len(new_kg.relations)} relations")

def demo_domain_templates():
    """Demonstrate domain-specific templates"""
    print_section("DOMAIN TEMPLATES", "📋")
    
    for domain in DomainType:
        print_subsection(f"{domain.value.title()} Template")
        template = get_domain_template(domain)
        
        print(f"📝 Template Facts: {len(template.template_facts)}")
        if template.template_facts:
            for fact in template.template_facts[:2]:  # Show first 2
                print(f"   • {fact}")
            if len(template.template_facts) > 2:
                print(f"   ... and {len(template.template_facts) - 2} more")
        
        print(f"📋 Template Rules: {len(template.template_rules)}")
        if template.template_rules:
            for rule in template.template_rules[:1]:  # Show first 1
                print(f"   • {rule}")
            if len(template.template_rules) > 1:
                print(f"   ... and {len(template.template_rules) - 1} more")
        
        print(f"🏷️  Template Entities: {len(template.template_entities)}")
        print(f"🔗 Template Relations: {len(template.template_relations)}")

def demo_statistics(kb_manager):
    """Display system statistics"""
    print_section("SYSTEM STATISTICS", "📊")
    
    stats = kb_manager.get_statistics()
    
    print(f"📈 Total Knowledge Bases: {stats['total_knowledge_bases']}")
    
    print_subsection("Domain Distribution")
    for domain, count in stats['domain_distribution'].items():
        print(f"   {domain}: {count} knowledge bases")
    
    print_subsection("Format Distribution")
    for format_type, count in stats['format_distribution'].items():
        print(f"   {format_type}: {count} knowledge bases")
    
    print_subsection("Content Statistics")
    content_stats = stats['content_statistics']
    print(f"   Total Facts: {content_stats['total_facts']}")
    print(f"   Total Rules: {content_stats['total_rules']}")
    print(f"   Total Entities: {content_stats['total_entities']}")
    print(f"   Total Relations: {content_stats['total_relations']}")
    print(f"   Average Facts per KB: {content_stats['average_facts_per_kb']}")
    print(f"   Average Rules per KB: {content_stats['average_rules_per_kb']}")
    
    print_subsection("Storage Information")
    storage = stats['storage']
    print(f"   Data Size: {storage['data_size_bytes']} bytes")
    print(f"   Backup Size: {storage['backup_size_bytes']} bytes")
    print(f"   Total Size: {storage['total_size_bytes']} bytes")
    
    print_subsection("Cache Status")
    cache = stats['cache_status']
    print(f"   Loaded KBs in Cache: {cache['loaded_kbs']}")
    print(f"   Cache Hit Potential: {cache['cache_hit_potential']:.2%}")

def cleanup_demo_files():
    """Clean up demo files"""
    print_section("CLEANUP", "🧹")
    
    import shutil
    demo_path = Path("demo_knowledge_bases")
    
    if demo_path.exists():
        try:
            shutil.rmtree(demo_path)
            print(f"🗑️  Cleaned up demo directory: {demo_path}")
        except Exception as e:
            print(f"⚠️  Could not clean up {demo_path}: {e}")
    else:
        print("✅ No cleanup needed - demo directory doesn't exist")

async def main():
    """Run the complete knowledge base management demonstration"""
    print("🧠 OMNI AI - KNOWLEDGE BASE MANAGEMENT SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the comprehensive knowledge base management capabilities")
    print("including creation, storage, import/export, and integration features.")
    
    try:
        # Demo domain templates
        demo_domain_templates()
        
        # Demo knowledge base creation
        kb_manager, kb_ids = await demo_knowledge_base_creation()
        
        # Demo operations
        await demo_knowledge_base_operations(kb_manager, kb_ids)
        
        # Demo import/export
        await demo_import_export(kb_manager, kb_ids)
        
        # Demo reasoner integration
        await demo_integration_with_reasoner(kb_manager)
        
        # Demo knowledge graph integration
        await demo_integration_with_knowledge_graph(kb_manager)
        
        # Show statistics
        demo_statistics(kb_manager)
        
        print_section("DEMO COMPLETED SUCCESSFULLY", "🎉")
        print("The knowledge base management system is fully operational!")
        print("All features have been demonstrated:")
        print("✅ Domain-specific templates")
        print("✅ Multiple storage formats (JSON, YAML, SQLite, Text, RDF)")
        print("✅ Import/Export capabilities")
        print("✅ Backup and restore")
        print("✅ Search and filtering")
        print("✅ Integration with symbolic reasoner")
        print("✅ Integration with knowledge graph")
        print("✅ Comprehensive statistics")
        
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ask user if they want to keep demo files
        response = input("\n🤔 Keep demo files for inspection? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            cleanup_demo_files()
        else:
            print("📁 Demo files preserved in 'demo_knowledge_bases' directory")

if __name__ == "__main__":
    asyncio.run(main())