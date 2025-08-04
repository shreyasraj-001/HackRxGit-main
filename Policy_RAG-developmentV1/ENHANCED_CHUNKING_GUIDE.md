# Enhanced Document Chunking Framework - Integration Guide

## 🚀 Overview

The Enhanced Document Chunking Framework provides multiple advanced chunking strategies specifically optimized for policy documents. It includes intelligent strategy selection, comprehensive quality metrics, and robust fallback mechanisms.

## 🧠 Available Chunking Strategies

### 1. **Adaptive Chunking** (Recommended)
- **Best for**: General use, automatic optimization
- **Features**: Analyzes document structure and selects optimal strategy
- **Performance**: Balanced speed and quality
```python
from utils.enhanced_chunking import adaptive_split
chunks = adaptive_split(text, chunk_size=1000, chunk_overlap=100)
```

### 2. **Hierarchical Chunking**
- **Best for**: Complex documents with clear structure
- **Features**: Parent-child relationships, maintains document hierarchy
- **Performance**: High quality, moderate speed
```python
from utils.enhanced_chunking import hierarchical_split
chunks = hierarchical_split(text, chunk_size=1000, chunk_overlap=100)
```

### 3. **Policy-Aware Chunking**
- **Best for**: Insurance policy documents
- **Features**: Recognizes policy sections (coverage, exclusions, definitions)
- **Performance**: High relevance for policy documents
```python
from utils.enhanced_chunking import policy_aware_split
chunks = policy_aware_split(text, chunk_size=1000, chunk_overlap=100)
```

### 4. **Semantic Chunking**
- **Best for**: Content coherence
- **Features**: Uses embeddings to maintain semantic similarity
- **Performance**: High quality, slower speed
```python
from utils.enhanced_chunking import enhanced_semantic_split
chunks = enhanced_semantic_split(text, chunk_size=1000, chunk_overlap=100)
```

### 5. **Lightning Fast Chunking**
- **Best for**: High-volume processing, real-time applications
- **Features**: Ultra-fast rule-based splitting
- **Performance**: Maximum speed, basic quality
```python
from utils.splitter import lightning_fast_split
chunks = lightning_fast_split(text, chunk_size=1000, chunk_overlap=100)
```

## 🔧 Advanced Configuration

### Using the Complete Framework
```python
from utils.enhanced_chunking import (
    EnhancedChunkingFramework, 
    ChunkingConfig, 
    ChunkingStrategy
)

# Create configuration
config = ChunkingConfig(
    chunk_size=1000,
    chunk_overlap=100,
    min_chunk_size=50,
    max_chunk_size=2000,
    strategy=ChunkingStrategy.ADAPTIVE,
    preserve_sections=True,
    use_embeddings=True,
    timeout_seconds=15,
    similarity_threshold=0.8
)

# Initialize framework
framework = EnhancedChunkingFramework(config)

# Process document
result = framework.chunk_document(text)

# Access results
chunks = result['chunks']
metadata = result['metadata']
quality_metrics = result['quality_metrics']
performance = result['performance']
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Target size for each chunk (characters) |
| `chunk_overlap` | 100 | Overlap between consecutive chunks |
| `min_chunk_size` | 50 | Minimum acceptable chunk size |
| `max_chunk_size` | 2000 | Maximum chunk size before splitting |
| `strategy` | ADAPTIVE | Chunking strategy to use |
| `preserve_sections` | True | Respect document section boundaries |
| `use_embeddings` | True | Enable semantic analysis |
| `timeout_seconds` | 15 | Maximum time for complex operations |
| `similarity_threshold` | 0.8 | Semantic similarity threshold |

## 📊 Quality Metrics

The framework provides comprehensive quality assessment:

```python
quality_metrics = result['quality_metrics']

print(f"Total chunks: {quality_metrics['total_chunks']}")
print(f"Average length: {quality_metrics['avg_chunk_length']:.1f}")
print(f"Size consistency: {quality_metrics['length_consistency']:.3f}")
print(f"Content preservation: {quality_metrics['content_preservation_score']:.3f}")
```

### Quality Metrics Explained

- **Length Consistency**: How uniform chunk sizes are (0-1, higher is better)
- **Content Preservation**: How well important keywords are preserved (0-1)
- **Coverage Ratio**: Proportion of original text covered by chunks
- **Size Distribution Score**: How well chunks match target size

## 🔄 Integration with Existing Code

### Replace Existing Chunking
```python
# OLD WAY
from utils.splitter import semantic_split
chunks = semantic_split(text, chunk_size=1000, chunk_overlap=100)

# NEW WAY (Drop-in replacement)
from utils.splitter import adaptive_split
chunks = adaptive_split(text, chunk_size=1000, chunk_overlap=100)
```

### Update Main Processing Loop
```python
# In main.py or your processing file
from utils.enhanced_chunking import EnhancedChunkingFramework, ChunkingConfig, ChunkingStrategy

class Chunker:
    def __init__(self, text: str):
        self.text = text
        self.chunks = []
        self.chunk_size = 1000
        self.chunk_overlap = 100
        
        # Use enhanced chunking framework
        try:
            config = ChunkingConfig(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                strategy=ChunkingStrategy.ADAPTIVE
            )
            framework = EnhancedChunkingFramework(config)
            result = framework.chunk_document(text)
            
            self.chunks = result['chunks']
            self.metadata = result['metadata']
            self.quality_metrics = result['quality_metrics']
            
            print(f"✅ Enhanced chunking complete: {len(self.chunks)} chunks created")
            print(f"📊 Quality score: {self.quality_metrics.get('content_preservation_score', 0):.3f}")
            
        except Exception as e:
            print(f"❌ Enhanced chunking failed: {e}")
            # Fallback to original method
            from utils.splitter import semantic_split
            self.chunks = semantic_split(text, self.chunk_size, self.chunk_overlap)
```

## 🧪 Performance Testing

Run comprehensive performance tests:

```bash
# Test all chunking strategies
python test_enhanced_chunking.py

# This will test:
# - All basic strategies (fast, smart, adaptive, etc.)
# - Enhanced framework strategies
# - Generate performance report
# - Save detailed analysis
```

## 📈 Performance Optimization Tips

### 1. **Choose Strategy Based on Use Case**
- **Real-time processing**: Use `lightning_fast_split`
- **Batch processing**: Use `adaptive_split`
- **High-quality retrieval**: Use `hierarchical_split`
- **Policy documents**: Use `policy_aware_split`

### 2. **Optimize Configuration**
```python
# For speed-optimized processing
config = ChunkingConfig(
    chunk_size=1200,  # Larger chunks = fewer chunks to process
    chunk_overlap=50,  # Less overlap = faster processing
    timeout_seconds=5,  # Shorter timeout
    use_embeddings=False  # Disable heavy embedding computation
)

# For quality-optimized processing
config = ChunkingConfig(
    chunk_size=800,   # Smaller chunks = better granularity
    chunk_overlap=150, # More overlap = better context preservation
    timeout_seconds=30, # Allow more time for complex operations
    use_embeddings=True # Enable semantic analysis
)
```

### 3. **Monitor Performance**
```python
result = framework.chunk_document(text)
performance = result['performance']

print(f"Total time: {performance['total_time']:.2f}s")
print(f"Chunks/second: {performance['chunks_per_second']:.1f}")
print(f"Analysis time: {performance['analysis_time']:.2f}s")
print(f"Chunking time: {performance['chunking_time']:.2f}s")
```

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **ImportError: Enhanced chunking not available**
   ```python
   # Check if numpy is installed
   pip install numpy
   
   # Ensure all dependencies are available
   pip install langchain langchain-experimental langchain-ollama
   ```

2. **Timeout errors with semantic chunking**
   ```python
   # Increase timeout or disable embeddings
   config.timeout_seconds = 30
   config.use_embeddings = False
   ```

3. **Poor chunk quality**
   ```python
   # Try different strategy
   config.strategy = ChunkingStrategy.HIERARCHICAL
   
   # Adjust chunk size
   config.chunk_size = 800
   config.chunk_overlap = 150
   ```

4. **Slow performance**
   ```python
   # Use faster strategy
   config.strategy = ChunkingStrategy.POLICY_AWARE
   
   # Or use direct function
   from utils.splitter import lightning_fast_split
   chunks = lightning_fast_split(text)
   ```

## 🎯 Best Practices

1. **Test Different Strategies**: Use the performance test to find the best strategy for your use case
2. **Monitor Quality**: Check quality metrics to ensure chunking meets your requirements
3. **Use Appropriate Timeouts**: Set reasonable timeouts based on your performance requirements
4. **Cache Results**: Store chunking results for repeated processing of the same documents
5. **Fallback Handling**: Always implement fallback chunking for production use

## 📋 Migration Checklist

- [ ] Install required dependencies (`numpy`, updated langchain packages)
- [ ] Update import statements to use enhanced chunking
- [ ] Test performance with your documents
- [ ] Configure strategy and parameters for your use case
- [ ] Implement error handling and fallbacks
- [ ] Monitor performance and quality metrics
- [ ] Update documentation and team training

## 🔮 Future Enhancements

The framework is designed to be extensible. Planned improvements include:

- **Custom Strategy Registration**: Add your own chunking algorithms
- **Machine Learning Optimization**: Automatically tune parameters based on document type
- **Parallel Processing**: Process multiple documents simultaneously
- **Advanced Quality Metrics**: More sophisticated content analysis
- **Database Integration**: Store and retrieve chunking configurations
