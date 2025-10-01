"""Tests for text analysis functionality."""

import pytest
from omni_ai.core import text_analysis


class TestTextAnalysis:
    """Test cases for text analysis functions."""
    
    def test_analyze_text_basic(self):
        """Test basic text analysis functionality."""
        text = "This is a simple test. It has two sentences."
        result = text_analysis.analyze_text(text)
        
        assert result['word_count'] == 9
        assert result['sentence_count'] == 2
        assert result['character_count'] == len(text)
        assert 'flesch_reading_ease' in result
        assert 'reading_level' in result
    
    def test_analyze_text_detailed(self):
        """Test detailed text analysis."""
        text = "This is a more complex test with multiple sentences. We want to test detailed analysis."
        result = text_analysis.analyze_text(text, detailed=True)
        
        assert 'syllable_count' in result
        assert 'lexicon_count' in result
        assert 'complex_words' in result
        assert 'automated_readability_index' in result
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "Python is a programming language. Python is widely used for data analysis and machine learning."
        keywords = text_analysis.extract_keywords(text, max_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert 'python' in keywords  # Should be case-insensitive
    
    def test_reading_level_classification(self):
        """Test reading level classification."""
        # Very simple text should be easy to read
        simple_text = "The cat sat on the mat. It was a big cat."
        result = text_analysis.analyze_text(simple_text)
        
        assert result['flesch_reading_ease'] > 60  # Should be relatively easy
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = text_analysis.analyze_text("")
        
        # Should handle empty text gracefully
        assert result['word_count'] == 0
        assert result['character_count'] == 0
    
    def test_single_word(self):
        """Test analysis of single word."""
        result = text_analysis.analyze_text("Hello")
        
        assert result['word_count'] == 1
        assert result['character_count'] == 5
        assert result['sentence_count'] == 0  # No sentence punctuation


class TestKeywordExtraction:
    """Test cases for keyword extraction."""
    
    def test_stop_words_filtered(self):
        """Test that stop words are properly filtered."""
        text = "The quick brown fox jumps over the lazy dog."
        keywords = text_analysis.extract_keywords(text)
        
        # Stop words should not be in keywords
        stop_words = {'the', 'over'}
        for word in stop_words:
            assert word not in keywords
    
    def test_frequency_sorting(self):
        """Test that keywords are sorted by frequency."""
        text = "test test test important important less"
        keywords = text_analysis.extract_keywords(text)
        
        # 'test' appears 3 times, should be first
        # 'important' appears 2 times, should be second
        assert keywords[0] == 'test'
        assert keywords[1] == 'important'
    
    def test_max_keywords_limit(self):
        """Test that max_keywords parameter is respected."""
        text = "one two three four five six seven eight nine ten eleven twelve"
        keywords = text_analysis.extract_keywords(text, max_keywords=5)
        
        assert len(keywords) <= 5


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Artificial intelligence and machine learning are transforming the world.
    These technologies enable computers to learn from data and make decisions.
    Applications include natural language processing, computer vision, and robotics.
    The future of AI holds great promise for solving complex problems.
    """


def test_analysis_with_fixture(sample_text):
    """Test analysis using sample text fixture."""
    result = text_analysis.analyze_text(sample_text.strip())
    
    assert result['word_count'] > 30
    assert result['sentence_count'] >= 4
    assert 'flesch_reading_ease' in result
    assert result['paragraph_count'] >= 1