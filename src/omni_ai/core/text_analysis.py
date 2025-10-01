"""Text analysis utilities for safe, ethical processing."""

import re
from typing import Dict, Any

import textstat
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def analyze_text(text: str, detailed: bool = False) -> Dict[str, Any]:
    """
    Analyze text for various metrics including readability and complexity.
    
    Args:
        text: The text to analyze
        detailed: Whether to include detailed analysis
    
    Returns:
        Dictionary containing analysis results
    """
    # Basic statistics
    word_count = len(text.split())
    char_count = len(text)
    char_count_no_spaces = len(text.replace(' ', ''))
    sentence_count = len(re.findall(r'[.!?]+', text))
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    # Readability scores
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    coleman_liau_index = textstat.coleman_liau_index(text)
    
    results = {
        'word_count': word_count,
        'character_count': char_count,
        'character_count_no_spaces': char_count_no_spaces,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0,
        'flesch_reading_ease': flesch_reading_ease,
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'coleman_liau_index': coleman_liau_index,
        'reading_level': _get_reading_level(flesch_reading_ease),
    }
    
    if detailed:
        # Additional detailed analysis
        syllable_count = textstat.syllable_count(text)
        lexicon_count = textstat.lexicon_count(text)
        automated_readability_index = textstat.automated_readability_index(text)
        
        results.update({
            'syllable_count': syllable_count,
            'lexicon_count': lexicon_count,
            'automated_readability_index': automated_readability_index,
            'avg_syllables_per_word': syllable_count / word_count if word_count > 0 else 0,
            'complex_words': textstat.difficult_words(text),
        })
    
    return results


def _get_reading_level(flesch_score: float) -> str:
    """Convert Flesch reading ease score to reading level description."""
    if flesch_score >= 90:
        return "Very Easy (5th grade)"
    elif flesch_score >= 80:
        return "Easy (6th grade)"
    elif flesch_score >= 70:
        return "Fairly Easy (7th grade)"
    elif flesch_score >= 60:
        return "Standard (8th-9th grade)"
    elif flesch_score >= 50:
        return "Fairly Difficult (10th-12th grade)"
    elif flesch_score >= 30:
        return "Difficult (College level)"
    else:
        return "Very Difficult (Graduate level)"


def display_analysis(results: Dict[str, Any], console: Console) -> None:
    """Display text analysis results in a formatted table."""
    
    # Basic statistics table
    basic_table = Table(title="📊 Text Statistics", border_style="cyan")
    basic_table.add_column("Metric", style="bold cyan")
    basic_table.add_column("Value", style="green")
    
    basic_table.add_row("Word Count", str(results['word_count']))
    basic_table.add_row("Character Count", str(results['character_count']))
    basic_table.add_row("Character Count (no spaces)", str(results['character_count_no_spaces']))
    basic_table.add_row("Sentence Count", str(results['sentence_count']))
    basic_table.add_row("Paragraph Count", str(results['paragraph_count']))
    basic_table.add_row("Avg Words per Sentence", f"{results['avg_words_per_sentence']:.1f}")
    
    console.print(basic_table)
    
    # Readability table
    readability_table = Table(title="📖 Readability Analysis", border_style="green")
    readability_table.add_column("Metric", style="bold green")
    readability_table.add_column("Score", style="yellow")
    readability_table.add_column("Interpretation", style="cyan")
    
    readability_table.add_row(
        "Flesch Reading Ease",
        f"{results['flesch_reading_ease']:.1f}",
        results['reading_level']
    )
    readability_table.add_row(
        "Flesch-Kincaid Grade",
        f"{results['flesch_kincaid_grade']:.1f}",
        f"Grade {results['flesch_kincaid_grade']:.0f} level"
    )
    readability_table.add_row(
        "Coleman-Liau Index",
        f"{results['coleman_liau_index']:.1f}",
        f"Grade {results['coleman_liau_index']:.0f} level"
    )
    
    console.print(readability_table)
    
    # Detailed analysis if available
    if 'syllable_count' in results:
        detailed_table = Table(title="🔍 Detailed Analysis", border_style="magenta")
        detailed_table.add_column("Metric", style="bold magenta")
        detailed_table.add_column("Value", style="green")
        
        detailed_table.add_row("Syllable Count", str(results['syllable_count']))
        detailed_table.add_row("Lexicon Count", str(results['lexicon_count']))
        detailed_table.add_row("Complex Words", str(results['complex_words']))
        detailed_table.add_row("Avg Syllables per Word", f"{results['avg_syllables_per_word']:.2f}")
        detailed_table.add_row("Automated Readability Index", f"{results['automated_readability_index']:.1f}")
        
        console.print(detailed_table)


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """
    Extract key terms from text using simple frequency analysis.
    
    Args:
        text: Text to analyze
        max_keywords: Maximum number of keywords to return
    
    Returns:
        List of keywords sorted by relevance
    """
    # Simple keyword extraction - count word frequency
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'over'
    }
    
    # Clean and split text
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]