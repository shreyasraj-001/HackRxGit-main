"""
Enhanced Document Chunking Framework
Implements multiple advanced chunking strategies for policy documents
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import (
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter
)

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Enumeration of available chunking strategies"""
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    POLICY_AWARE = "policy_aware"
    TOKEN_BASED = "token_based"
    SENTENCE_BASED = "sentence_based"
    SLIDING_WINDOW = "sliding_window"

@dataclass
class ChunkingConfig:
    """Configuration for chunking parameters"""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE
    preserve_sections: bool = True
    use_embeddings: bool = True
    timeout_seconds: int = 10
    similarity_threshold: float = 0.8
    window_size: int = 3

@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: str
    start_position: int
    end_position: int
    length: int
    section_type: str
    importance_score: float
    semantic_coherence: float
    overlap_with_prev: int
    contains_keywords: List[str]

class PolicyDocumentAnalyzer:
    """Analyzes policy documents to identify structure and important sections"""
    
    def __init__(self):
        self.section_patterns = {
            'coverage': r'(?i)\b(coverage|benefits?|insured|policy holder)\b',
            'exclusions': r'(?i)\b(exclusion|not covered|does not cover|excluded)\b',
            'definitions': r'(?i)\b(definition|means|shall mean|defined as)\b',
            'claims': r'(?i)\b(claim|claims process|filing|notification)\b',
            'premiums': r'(?i)\b(premium|payment|cost|fee|deductible)\b',
            'terms': r'(?i)\b(term|condition|requirement|obligation)\b',
            'limitations': r'(?i)\b(limitation|limited to|maximum|minimum)\b',
            'procedures': r'(?i)\b(procedure|process|step|requirement)\b'
        }
        
        self.header_patterns = [
            r'^[A-Z\s]+(?:COVERAGE|BENEFITS|EXCLUSIONS|DEFINITIONS|CLAIMS|TERMS)[A-Z\s]*$',
            r'^\d+\.\s+[A-Z][^.]+$',
            r'^[IVX]+\.\s+[A-Z][^.]+$',
            r'^[A-Z]\.\s+[A-Z][^.]+$',
        ]
        
        self.important_keywords = [
            'policy', 'coverage', 'benefit', 'exclusion', 'claim', 'premium',
            'deductible', 'liability', 'insured', 'policyholder', 'beneficiary',
            'effective date', 'expiration', 'renewal', 'cancellation'
        ]

    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and identify important sections"""
        analysis = {
            'sections': {},
            'headers': [],
            'keywords_density': {},
            'structure_score': 0.0,
            'complexity_score': 0.0
        }
        
        # Find section boundaries
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if len(line_clean) > 5:
                # Check for headers
                for pattern in self.header_patterns:
                    if re.match(pattern, line_clean):
                        analysis['headers'].append({
                            'line': i,
                            'text': line_clean,
                            'position': text.find(line_clean)
                        })
                        break
        
        # Analyze section types
        for section_type, pattern in self.section_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            positions = [match.start() for match in matches]
            analysis['sections'][section_type] = {
                'count': len(positions),
                'positions': positions,
                'density': len(positions) / len(text) * 1000  # per 1000 chars
            }
        
        # Calculate keyword density
        text_lower = text.lower()
        for keyword in self.important_keywords:
            count = text_lower.count(keyword.lower())
            analysis['keywords_density'][keyword] = count / len(text) * 1000
        
        # Calculate complexity scores
        analysis['structure_score'] = len(analysis['headers']) / len(lines) * 100
        analysis['complexity_score'] = sum(section['count'] for section in analysis['sections'].values()) / len(text) * 1000
        
        return analysis

class HierarchicalChunker:
    """Implements hierarchical chunking with parent-child relationships"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.analyzer = PolicyDocumentAnalyzer()
    
    def create_hierarchical_chunks(self, text: str) -> Tuple[List[str], List[ChunkMetadata]]:
        """Create hierarchical chunks with parent-child relationships"""
        analysis = self.analyzer.analyze_document_structure(text)
        
        # Create large parent chunks first
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size * 2,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n"]
        )
        
        parent_chunks = parent_splitter.split_text(text)
        
        # Create smaller child chunks
        child_chunks = []
        chunk_metadata = []
        
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap // 2,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
            )
            
            children = child_splitter.split_text(parent_chunk)
            
            for child_idx, child_chunk in enumerate(children):
                chunk_id = f"p{parent_idx}_c{child_idx}"
                start_pos = text.find(child_chunk)
                
                # Analyze chunk content
                section_type = self._identify_section_type(child_chunk, analysis)
                importance = self._calculate_importance_score(child_chunk, analysis)
                keywords = self._extract_keywords(child_chunk)
                
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    start_position=start_pos,
                    end_position=start_pos + len(child_chunk),
                    length=len(child_chunk),
                    section_type=section_type,
                    importance_score=importance,
                    semantic_coherence=0.0,  # Will be calculated later if embeddings available
                    overlap_with_prev=0,
                    contains_keywords=keywords
                )
                
                child_chunks.append(child_chunk)
                chunk_metadata.append(metadata)
        
        return child_chunks, chunk_metadata
    
    def _identify_section_type(self, chunk: str, analysis: Dict) -> str:
        """Identify the type of section this chunk belongs to"""
        max_score = 0
        section_type = "general"
        
        for section, pattern in self.analyzer.section_patterns.items():
            matches = len(re.findall(pattern, chunk, re.IGNORECASE))
            score = matches / len(chunk) * 1000
            if score > max_score:
                max_score = score
                section_type = section
        
        return section_type
    
    def _calculate_importance_score(self, chunk: str, analysis: Dict) -> float:
        """Calculate importance score based on content analysis"""
        score = 0.0
        chunk_lower = chunk.lower()
        
        # Score based on keywords
        for keyword in self.analyzer.important_keywords:
            if keyword.lower() in chunk_lower:
                score += 1.0
        
        # Score based on section type indicators
        for section_type, pattern in self.analyzer.section_patterns.items():
            matches = len(re.findall(pattern, chunk, re.IGNORECASE))
            score += matches * 0.5
        
        # Normalize by chunk length
        return min(score / len(chunk) * 1000, 10.0)
    
    def _extract_keywords(self, chunk: str) -> List[str]:
        """Extract important keywords from chunk"""
        found_keywords = []
        chunk_lower = chunk.lower()
        
        for keyword in self.analyzer.important_keywords:
            if keyword.lower() in chunk_lower:
                found_keywords.append(keyword)
        
        return found_keywords

class SlidingWindowChunker:
    """Implements sliding window chunking with configurable overlap"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    def create_sliding_chunks(self, text: str) -> List[str]:
        """Create chunks using sliding window approach"""
        chunks = []
        step_size = self.config.chunk_size - self.config.chunk_overlap
        
        for i in range(0, len(text), step_size):
            chunk = text[i:i + self.config.chunk_size]
            
            # Ensure we don't break words
            if i + self.config.chunk_size < len(text) and not chunk.endswith(' '):
                last_space = chunk.rfind(' ')
                if last_space > self.config.chunk_size * 0.8:  # Only adjust if space is near end
                    chunk = chunk[:last_space]
            
            if len(chunk.strip()) >= self.config.min_chunk_size:
                chunks.append(chunk.strip())
                
            if i + self.config.chunk_size >= len(text):
                break
        
        return chunks

class AdaptiveChunker:
    """Implements adaptive chunking that adjusts strategy based on content"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.analyzer = PolicyDocumentAnalyzer()
        self.hierarchical_chunker = HierarchicalChunker(config)
        self.sliding_chunker = SlidingWindowChunker(config)
    
    def chunk_adaptively(self, text: str) -> Tuple[List[str], List[ChunkMetadata]]:
        """Choose and apply the best chunking strategy based on content analysis"""
        analysis = self.analyzer.analyze_document_structure(text)
        
        # Decide on strategy based on document characteristics
        strategy = self._choose_strategy(text, analysis)
        logger.info(f"Adaptive chunker selected strategy: {strategy}")
        
        if strategy == "hierarchical":
            return self.hierarchical_chunker.create_hierarchical_chunks(text)
        elif strategy == "sliding":
            chunks = self.sliding_chunker.create_sliding_chunks(text)
            metadata = [self._create_basic_metadata(i, chunk, text) for i, chunk in enumerate(chunks)]
            return chunks, metadata
        else:
            return self._create_policy_aware_chunks(text, analysis)
    
    def _choose_strategy(self, text: str, analysis: Dict) -> str:
        """Choose the best chunking strategy based on document analysis"""
        # Factors to consider
        text_length = len(text)
        header_count = len(analysis['headers'])
        section_complexity = analysis['complexity_score']
        structure_score = analysis['structure_score']
        
        # Decision logic
        if header_count > 10 and structure_score > 2.0:
            return "hierarchical"
        elif section_complexity > 5.0:
            return "policy_aware"
        elif text_length > 50000:
            return "sliding"
        else:
            return "policy_aware"
    
    def _create_policy_aware_chunks(self, text: str, analysis: Dict) -> Tuple[List[str], List[ChunkMetadata]]:
        """Create policy-aware chunks that respect document structure"""
        chunks = []
        metadata = []
        
        # Use headers as natural boundaries
        boundaries = [0] + [header['position'] for header in analysis['headers']] + [len(text)]
        boundaries = sorted(set(boundaries))
        
        for i in range(len(boundaries) - 1):
            section_text = text[boundaries[i]:boundaries[i + 1]].strip()
            
            if len(section_text) < self.config.min_chunk_size:
                continue
            
            if len(section_text) <= self.config.max_chunk_size:
                # Section fits in one chunk
                chunks.append(section_text)
                metadata.append(self._create_basic_metadata(len(chunks) - 1, section_text, text))
            else:
                # Split large section
                section_chunks = self._split_large_section(section_text)
                for chunk in section_chunks:
                    chunks.append(chunk)
                    metadata.append(self._create_basic_metadata(len(chunks) - 1, chunk, text))
        
        return chunks, metadata
    
    def _split_large_section(self, section_text: str) -> List[str]:
        """Split a large section into smaller chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        )
        return splitter.split_text(section_text)
    
    def _create_basic_metadata(self, chunk_id: int, chunk: str, full_text: str) -> ChunkMetadata:
        """Create basic metadata for a chunk"""
        start_pos = full_text.find(chunk)
        section_type = "general"
        
        # Simple section type detection
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in ['coverage', 'benefit', 'insured']):
            section_type = "coverage"
        elif any(word in chunk_lower for word in ['exclusion', 'not covered', 'excluded']):
            section_type = "exclusions"
        elif any(word in chunk_lower for word in ['definition', 'means', 'defined']):
            section_type = "definitions"
        elif any(word in chunk_lower for word in ['claim', 'filing', 'notification']):
            section_type = "claims"
        
        return ChunkMetadata(
            chunk_id=str(chunk_id),
            start_position=start_pos,
            end_position=start_pos + len(chunk),
            length=len(chunk),
            section_type=section_type,
            importance_score=self._calculate_simple_importance(chunk),
            semantic_coherence=0.0,
            overlap_with_prev=0,
            contains_keywords=[]
        )
    
    def _calculate_simple_importance(self, chunk: str) -> float:
        """Calculate a simple importance score"""
        important_terms = ['policy', 'coverage', 'exclusion', 'claim', 'premium', 'deductible']
        score = sum(1 for term in important_terms if term in chunk.lower())
        return min(score / len(chunk) * 1000, 10.0)

class EnhancedChunkingFramework:
    """Main framework that coordinates all chunking strategies"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.analyzer = PolicyDocumentAnalyzer()
        self.adaptive_chunker = AdaptiveChunker(self.config)
        
    def chunk_document(self, text: str, strategy: Optional[ChunkingStrategy] = None) -> Dict[str, Any]:
        """
        Chunk document using specified or adaptive strategy
        
        Returns:
            Dictionary containing chunks, metadata, and performance metrics
        """
        start_time = time.time()
        
        if strategy is None:
            strategy = self.config.strategy
        
        logger.info(f"Starting chunking with strategy: {strategy.value}")
        
        try:
            # Document analysis
            analysis_start = time.time()
            doc_analysis = self.analyzer.analyze_document_structure(text)
            analysis_time = time.time() - analysis_start
            
            # Chunking
            chunk_start = time.time()
            
            if strategy == ChunkingStrategy.ADAPTIVE:
                chunks, metadata = self.adaptive_chunker.chunk_adaptively(text)
            elif strategy == ChunkingStrategy.HIERARCHICAL:
                chunks, metadata = self.adaptive_chunker.hierarchical_chunker.create_hierarchical_chunks(text)
            elif strategy == ChunkingStrategy.SLIDING_WINDOW:
                chunks = self.adaptive_chunker.sliding_chunker.create_sliding_chunks(text)
                metadata = [self.adaptive_chunker._create_basic_metadata(i, chunk, text) for i, chunk in enumerate(chunks)]
            elif strategy == ChunkingStrategy.SEMANTIC:
                chunks, metadata = self._semantic_chunking_with_metadata(text)
            elif strategy == ChunkingStrategy.TOKEN_BASED:
                chunks, metadata = self._token_based_chunking(text)
            elif strategy == ChunkingStrategy.SENTENCE_BASED:
                chunks, metadata = self._sentence_based_chunking(text)
            else:
                # Default to policy-aware chunking
                chunks, metadata = self.adaptive_chunker._create_policy_aware_chunks(text, doc_analysis)
            
            chunk_time = time.time() - chunk_start
            
            # Quality assessment
            quality_start = time.time()
            quality_metrics = self._assess_chunk_quality(chunks, text)
            quality_time = time.time() - quality_start
            
            total_time = time.time() - start_time
            
            result = {
                'chunks': chunks,
                'metadata': metadata,
                'document_analysis': doc_analysis,
                'quality_metrics': quality_metrics,
                'performance': {
                    'total_time': total_time,
                    'analysis_time': analysis_time,
                    'chunking_time': chunk_time,
                    'quality_assessment_time': quality_time,
                    'chunks_per_second': len(chunks) / chunk_time if chunk_time > 0 else 0
                },
                'config': self.config
            }
            
            logger.info(f"Chunking completed: {len(chunks)} chunks in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            # Fallback to simple splitting
            return self._fallback_chunking(text, start_time)
    
    def _semantic_chunking_with_metadata(self, text: str) -> Tuple[List[str], List[ChunkMetadata]]:
        """Perform semantic chunking with metadata generation"""
        try:
            # Try to create embeddings
            try:
                embeddings = OllamaEmbeddings(model="all-minilm:33m")
            except:
                embeddings = OpenAIEmbeddings(
                    base_url="http://127.0.0.1:11434/",
                    model="all-minilm:33m",
                    api_key="ollama",
                )
            
            # Create semantic chunker
            semantic_chunker = SemanticChunker(embeddings=embeddings)
            doc = LangChainDocument(page_content=text)
            chunk_docs = semantic_chunker.split_documents([doc])
            
            chunks = [chunk.page_content for chunk in chunk_docs]
            metadata = [self.adaptive_chunker._create_basic_metadata(i, chunk, text) for i, chunk in enumerate(chunks)]
            
            return chunks, metadata
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}")
            # Fallback to recursive splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = splitter.split_text(text)
            metadata = [self.adaptive_chunker._create_basic_metadata(i, chunk, text) for i, chunk in enumerate(chunks)]
            return chunks, metadata
    
    def _token_based_chunking(self, text: str) -> Tuple[List[str], List[ChunkMetadata]]:
        """Perform token-based chunking"""
        try:
            splitter = TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = splitter.split_text(text)
            metadata = [self.adaptive_chunker._create_basic_metadata(i, chunk, text) for i, chunk in enumerate(chunks)]
            return chunks, metadata
        except Exception as e:
            logger.warning(f"Token-based chunking failed: {e}")
            return self._fallback_to_recursive(text)
    
    def _sentence_based_chunking(self, text: str) -> Tuple[List[str], List[ChunkMetadata]]:
        """Perform sentence-based chunking using NLTK"""
        try:
            splitter = NLTKTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = splitter.split_text(text)
            metadata = [self.adaptive_chunker._create_basic_metadata(i, chunk, text) for i, chunk in enumerate(chunks)]
            return chunks, metadata
        except Exception as e:
            logger.warning(f"Sentence-based chunking failed: {e}")
            return self._fallback_to_recursive(text)
    
    def _fallback_to_recursive(self, text: str) -> Tuple[List[str], List[ChunkMetadata]]:
        """Fallback to recursive character text splitter"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        chunks = splitter.split_text(text)
        metadata = [self.adaptive_chunker._create_basic_metadata(i, chunk, text) for i, chunk in enumerate(chunks)]
        return chunks, metadata
    
    def _assess_chunk_quality(self, chunks: List[str], original_text: str) -> Dict[str, float]:
        """Assess the quality of the chunking result"""
        if not chunks:
            return {'error': 'No chunks created'}
        
        # Basic quality metrics
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        quality_metrics = {
            'total_chunks': len(chunks),
            'avg_chunk_length': np.mean(chunk_lengths),
            'std_chunk_length': np.std(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'coverage_ratio': sum(chunk_lengths) / len(original_text),
            'length_consistency': 1.0 - (np.std(chunk_lengths) / np.mean(chunk_lengths)) if np.mean(chunk_lengths) > 0 else 0,
            'size_distribution_score': self._calculate_size_distribution_score(chunk_lengths),
            'content_preservation_score': self._calculate_content_preservation_score(chunks, original_text)
        }
        
        return quality_metrics
    
    def _calculate_size_distribution_score(self, chunk_lengths: List[int]) -> float:
        """Calculate how well distributed the chunk sizes are"""
        if not chunk_lengths:
            return 0.0
        
        target_size = self.config.chunk_size
        score = 0.0
        
        for length in chunk_lengths:
            # Score based on how close to target size
            ratio = min(length, target_size) / max(length, target_size)
            score += ratio
        
        return score / len(chunk_lengths)
    
    def _calculate_content_preservation_score(self, chunks: List[str], original_text: str) -> float:
        """Calculate how well the chunking preserves important content"""
        # Simple heuristic: check if important keywords are preserved
        original_keywords = set(re.findall(r'\b(?:policy|coverage|exclusion|claim|premium|deductible|benefit|insured)\b', 
                                         original_text.lower()))
        
        preserved_keywords = set()
        for chunk in chunks:
            chunk_keywords = set(re.findall(r'\b(?:policy|coverage|exclusion|claim|premium|deductible|benefit|insured)\b', 
                                          chunk.lower()))
            preserved_keywords.update(chunk_keywords)
        
        if not original_keywords:
            return 1.0
        
        return len(preserved_keywords) / len(original_keywords)
    
    def _fallback_chunking(self, text: str, start_time: float) -> Dict[str, Any]:
        """Fallback chunking method when all else fails"""
        logger.warning("Using fallback chunking method")
        
        # Simple character-based splitting
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text) and not chunk.endswith(' '):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8:
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap
        
        metadata = [self.adaptive_chunker._create_basic_metadata(i, chunk, text) for i, chunk in enumerate(chunks)]
        
        total_time = time.time() - start_time
        
        return {
            'chunks': chunks,
            'metadata': metadata,
            'document_analysis': {},
            'quality_metrics': {'error': 'Fallback method used'},
            'performance': {
                'total_time': total_time,
                'fallback_used': True
            },
            'config': self.config
        }

# Convenience functions for backward compatibility
def enhanced_semantic_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Enhanced semantic splitting with fallbacks"""
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=ChunkingStrategy.SEMANTIC
    )
    framework = EnhancedChunkingFramework(config)
    result = framework.chunk_document(text, ChunkingStrategy.SEMANTIC)
    return result['chunks']

def adaptive_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Adaptive splitting that chooses the best strategy"""
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=ChunkingStrategy.ADAPTIVE
    )
    framework = EnhancedChunkingFramework(config)
    result = framework.chunk_document(text, ChunkingStrategy.ADAPTIVE)
    return result['chunks']

def hierarchical_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Hierarchical splitting with parent-child relationships"""
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=ChunkingStrategy.HIERARCHICAL
    )
    framework = EnhancedChunkingFramework(config)
    result = framework.chunk_document(text, ChunkingStrategy.HIERARCHICAL)
    return result['chunks']

def policy_aware_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Policy-aware splitting that understands document structure"""
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=ChunkingStrategy.POLICY_AWARE
    )
    framework = EnhancedChunkingFramework(config)
    result = framework.chunk_document(text, ChunkingStrategy.POLICY_AWARE)
    return result['chunks']
