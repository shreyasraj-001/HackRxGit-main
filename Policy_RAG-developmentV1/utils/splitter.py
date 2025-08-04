"""
Optimized semantic text splitter utility for efficient chunking of documents
Enhanced with multiple chunking strategies and intelligent fallbacks
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LangChainDocument
from typing import List, Dict, Any, Optional
import logging
import time
import re

# Import enhanced chunking framework
try:
    from .enhanced_chunking import (
        EnhancedChunkingFramework, 
        ChunkingConfig, 
        ChunkingStrategy,
        adaptive_split as enhanced_adaptive_split,
        policy_aware_split,
        hierarchical_split
    )
    ENHANCED_CHUNKING_AVAILABLE = True
except ImportError:
    ENHANCED_CHUNKING_AVAILABLE = False
    logging.warning("Enhanced chunking framework not available, falling back to basic chunking")

logger = logging.getLogger(__name__)

def fast_rule_based_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Fast rule-based splitting optimized for policy documents
    Uses document structure and patterns for intelligent chunking
    """
    logger.info("Using fast rule-based chunking...")
    
    # Define policy document separators in order of priority
    separators = [
        "\n\n\n",  # Major section breaks
        "\n\n",    # Paragraph breaks
        "\n• ",    # Bullet points
        "\n- ",    # Dash points
        r"\n\d+\.", # Numbered points
        ". ",      # Sentence boundaries
        "! ",      # Exclamation boundaries
        "? ",      # Question boundaries
        "; ",      # Semicolon boundaries
        ", ",      # Comma boundaries
        " ",       # Space boundaries
        ""         # Character boundaries
    ]
    
    # Create optimized text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(text)
    
    # Post-process chunks for better quality
    processed_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and len(chunk) > 20:  # Filter very short chunks
            # Clean up chunk boundaries
            chunk = re.sub(r'\s+', ' ', chunk)  # Normalize whitespace
            processed_chunks.append(chunk)
    
    logger.info(f"Fast rule-based chunking complete: {len(processed_chunks)} chunks created")
    return processed_chunks

def smart_policy_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Smart policy document splitting that preserves important structure
    """
    logger.info("Using smart policy document chunking...")
    
    # Identify policy-specific patterns
    section_patterns = [
        r'\n[A-Z\s]+(?:POLICY|COVERAGE|BENEFITS|EXCLUSIONS|DEFINITIONS)[A-Z\s]*\n',
        r'\n\d+\.\s+[A-Z][^.]+[:.]',  # Section headers
        r'\n[A-Z][a-z\s]+:\s*',       # Definition patterns
        r'\nNOTE[:.]',                # Important notes
        r'\nIMPORTANT[:.]',           # Important sections
    ]
    
    # Find important boundaries
    important_boundaries = []
    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        important_boundaries.extend([match.start() for match in matches])
    
    important_boundaries = sorted(set(important_boundaries))
    
    if not important_boundaries:
        # No special structure found, use fast rule-based splitting
        return fast_rule_based_split(text, chunk_size, chunk_overlap)
    
    # Create chunks respecting important boundaries
    chunks = []
    start = 0
    
    for boundary in important_boundaries + [len(text)]:
        if boundary - start > chunk_size * 2:
            # Section is too large, split it
            section_text = text[start:boundary]
            section_chunks = fast_rule_based_split(section_text, chunk_size, chunk_overlap)
            chunks.extend(section_chunks)
        else:
            # Keep section intact if it's reasonable size
            section_text = text[start:boundary].strip()
            if section_text and len(section_text) > 20:
                chunks.append(section_text)
        start = boundary
    
    logger.info(f"Smart policy chunking complete: {len(chunks)} chunks created")
    return chunks

def semantic_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Enhanced semantic chunking with multiple strategies and intelligent fallbacks
    """
    start_time = time.time()
    
    # Quick check: if text is small, don't bother with complex chunking
    if len(text) < chunk_size * 1.5:
        logger.info("Text is small, returning as single chunk")
        return [text.strip()]
    
    # Try enhanced chunking framework first if available
    if ENHANCED_CHUNKING_AVAILABLE:
        try:
            logger.info("Using enhanced chunking framework with adaptive strategy")
            config = ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=ChunkingStrategy.ADAPTIVE,
                timeout_seconds=15
            )
            framework = EnhancedChunkingFramework(config)
            result = framework.chunk_document(text)
            
            chunks = result['chunks']
            if chunks and len(chunks) > 0:
                elapsed = time.time() - start_time
                logger.info(f"Enhanced chunking succeeded: {len(chunks)} chunks in {elapsed:.2f}s")
                logger.info(f"Quality metrics: {result.get('quality_metrics', {})}")
                return chunks
        except Exception as e:
            logger.warning(f"Enhanced chunking failed: {e}")
    
    # Strategy 1: Try smart policy-aware chunking first (fastest)
    try:
        chunks = smart_policy_split(text, chunk_size, chunk_overlap)
        if chunks and len(chunks) > 1:
            elapsed = time.time() - start_time
            logger.info(f"Smart policy chunking succeeded in {elapsed:.2f}s")
            return chunks
    except Exception as e:
        logger.warning(f"Smart policy chunking failed: {e}")
    
    # Strategy 2: Try semantic chunking with timeout (medium speed)
    try:
        logger.info("Attempting semantic chunking with timeout...")
        
        # Set a timeout for semantic chunking
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Semantic chunking timed out")
        
        # Try Ollama embeddings with timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        try:
            embeddings = OllamaEmbeddings(model="all-minilm:33m")
            logger.info("Using Ollama embeddings for semantic chunking")
        except Exception as e:
            logger.warning(f"Ollama embeddings failed: {e}")
            embeddings = OpenAIEmbeddings(
                base_url="http://127.0.0.1:11434/",
                model="all-minilm:33m",
                api_key="ollama",
            )
            logger.info("Using OpenAI-compatible embeddings")
        
        # Create semantic chunker
        semantic_chunker = SemanticChunker(embeddings=embeddings)
        doc = LangChainDocument(page_content=text)
        chunks = semantic_chunker.split_documents([doc])
        
        signal.alarm(0)  # Cancel timeout
        
        # Extract text content
        text_chunks = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
        
        if text_chunks:
            elapsed = time.time() - start_time
            logger.info(f"Semantic chunking successful: {len(text_chunks)} chunks in {elapsed:.2f}s")
            return text_chunks
            
    except (TimeoutError, Exception) as e:
        signal.alarm(0)  # Cancel timeout
        logger.warning(f"Semantic chunking failed or timed out: {e}")
    
    # Strategy 3: Fallback to fast rule-based splitting
    logger.info("Using fallback rule-based chunking")
    chunks = fast_rule_based_split(text, chunk_size, chunk_overlap)
    
    elapsed = time.time() - start_time
    logger.info(f"Total chunking time: {elapsed:.2f}s")
    return chunks

# New enhanced functions that utilize the enhanced chunking framework
def adaptive_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100, max_time: float = 10.0) -> List[str]:
    """
    Adaptive splitting that intelligently chooses the best chunking strategy
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        max_time: Maximum time to spend on chunking
        
    Returns:
        List of text chunks
    """
    if ENHANCED_CHUNKING_AVAILABLE:
        try:
            return enhanced_adaptive_split(text, chunk_size, chunk_overlap)
        except Exception as e:
            logger.warning(f"Enhanced adaptive split failed: {e}")
    
    # Fallback to smart policy split
    return smart_policy_split(text, chunk_size, chunk_overlap)

def lightning_fast_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Ultra-fast chunking prioritizing speed over sophistication
    """
    logger.info("Using lightning-fast chunking...")
    
    # Simple but fast splitting using basic patterns
    separators = ["\n\n\n", "\n\n", "\n", ". ", " "]
    
    chunks = []
    current_chunk = ""
    
    # Split by sentences first for better boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Apply overlap if specified
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) > chunk_overlap else prev_chunk
                overlapped_chunks.append(overlap_text + " " + chunk)
        chunks = overlapped_chunks
    
    logger.info(f"Lightning-fast chunking complete: {len(chunks)} chunks created")
    return chunks

def smart_fast_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Fast chunking with policy document awareness
    """
    if ENHANCED_CHUNKING_AVAILABLE:
        try:
            return policy_aware_split(text, chunk_size, chunk_overlap)
        except Exception as e:
            logger.warning(f"Enhanced policy-aware split failed: {e}")
    
    # Fallback to original smart policy split
    return smart_policy_split(text, chunk_size, chunk_overlap)

def contextual_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Context-aware chunking that maintains semantic coherence
    """
    if ENHANCED_CHUNKING_AVAILABLE:
        try:
            return hierarchical_split(text, chunk_size, chunk_overlap)
        except Exception as e:
            logger.warning(f"Enhanced hierarchical split failed: {e}")
    
    # Fallback to semantic split
    return semantic_split(text, chunk_size, chunk_overlap)
