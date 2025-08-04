# Enhanced RAG System with Advanced Retrieval 🚀

## Overview

This enhanced RAG (Retrieval-Augmented Generation) system provides state-of-the-art document retrieval and answer generation capabilities, specifically optimized for policy documents and complex question-answering scenarios.

## 🆕 Enhanced Features

### 1. Advanced Retrieval Techniques

#### Multi-Modal Fusion Retrieval
- **Semantic Search**: Dense vector retrieval using cosine similarity
- **BM25 Search**: Sparse keyword-based retrieval for exact matches
- **TF-IDF Search**: Statistical relevance scoring
- **Keyword Matching**: Exact term and phrase matching
- **Fusion Scoring**: Intelligent combination of all retrieval methods

#### Query Enhancement
- **Query Expansion**: Automatic synonym and related term expansion
- **Intent Recognition**: Identifies query type (definition, procedure, eligibility, etc.)
- **Context Awareness**: Uses conversation history for better retrieval

#### Smart Document Processing
- **Section-Aware Retrieval**: Understands document structure
- **Policy-Specific Patterns**: Optimized for insurance and policy documents
- **Confidence Scoring**: Provides retrieval confidence metrics

### 2. Enhanced Answer Generation

#### Intelligent Answer Structuring
- **Intent-Based Templates**: Different answer formats for different query types
- **Structured Responses**: Organized, professional answer formatting
- **Confidence Indicators**: Shows answer reliability
- **Source Attribution**: Clear references to document sections

#### Query Type Recognition
- **Definitions**: "What is..." questions
- **Procedures**: "How to..." questions  
- **Eligibility**: "Who can..." questions
- **Amounts**: "How much..." questions
- **Time**: "When..." and grace period questions
- **Coverage**: "What is covered..." questions
- **Exclusions**: "What is not covered..." questions

## 🔧 Installation

### Standard Installation
```bash
pip install -r requirements.txt
```

### Enhanced Features Installation
```bash
pip install -r requirements_enhanced.txt
```

### Additional Dependencies for Advanced Features
```bash
# For better text processing
python -m spacy download en_core_web_sm

# For NLTK (if using enhanced text processing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🚀 Usage

### 1. Basic Initialization
```python
# Start the server
python main.py

# Initialize the system
GET http://localhost:8000/
```

### 2. Standard Query
```python
# POST /query
{
    "question": "What is the grace period for premium payment?",
    "k": 5,
    "use_cosine_only": false,
    "similarity_threshold": 0.0,
    "use_advanced": true
}
```

### 3. Advanced Query with Context
```python
# POST /query/advanced
{
    "question": "How much is the premium?",
    "k": 7,
    "similarity_threshold": 0.3,
    "conversation_history": [
        "What is covered under this policy?",
        "Who is eligible for this insurance?"
    ],
    "explain_retrieval": true
}
```

## 📊 API Endpoints

### Enhanced Endpoints

#### `POST /query` - Standard Enhanced Query
- **Enhanced Parameters**:
  - `use_advanced`: Enable advanced retrieval (default: true)
  - All standard parameters supported

#### `POST /query/advanced` - Advanced Query with Full Features
- **New Parameters**:
  - `conversation_history`: Array of previous questions for context
  - `explain_retrieval`: Include retrieval explanation in response
  - Higher default similarity threshold (0.3)

#### `GET /query/{question}` - Quick Query
- **Enhanced Parameters**:
  - `use_advanced`: Enable advanced features (default: true)

### Response Format

#### Standard Response
```json
{
    "query": "What is the grace period?",
    "answer": "Based on the policy document, the grace period is defined as...",
    "confidence": 0.85,
    "intent": "definition",
    "num_chunks_retrieved": 5,
    "success": true,
    "processing_time": 0.234,
    "retrieval_mode": "Advanced Fusion"
}
```

#### Advanced Response
```json
{
    "query": "What is the grace period?",
    "answer": "Based on the policy document, the grace period is defined as: The time period...\n\n**Confidence Level:** 🟢 High (85%)\n**Query Type:** Definition",
    "confidence": 0.85,
    "intent": "definition",
    "num_chunks_retrieved": 5,
    "success": true,
    "processing_time": 0.234,
    "retrieval_mode": "Advanced Fusion with Context Awareness",
    "retrieval_explanation": "Retrieved based on semantic similarity, keyword matching...",
    "chunk_scores": [
        {"content": "Grace period means...", "score": 0.92},
        {"content": "Payment of premium...", "score": 0.87}
    ]
}
```

## 🎯 Retrieval Accuracy Improvements

### 1. Multi-Signal Fusion
- Combines 4 different retrieval methods
- Weighted scoring for optimal results
- Adaptive thresholds based on query type

### 2. Context Understanding
- Policy-specific keyword recognition
- Document structure awareness
- Conversation history integration

### 3. Quality Metrics
- Confidence scoring for all responses
- Retrieval explanation capabilities
- Performance monitoring and logging

### 4. Answer Quality
- Intent-based answer structuring
- Professional formatting
- Clear source attribution
- Confidence indicators

## 🔍 Advanced Configuration

### Environment Variables
```bash
# Enhanced features
ENABLE_ADVANCED_RETRIEVAL=true
SIMILARITY_THRESHOLD_DEFAULT=0.3
MAX_FUSION_CANDIDATES=10

# LangSmith monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=enhanced-policy-rag
LANGSMITH_STORE_DOCUMENTS=true
LANGSMITH_STORE_QUESTIONS=true
```

### Retrieval Parameters Tuning
```python
# In advanced_retrieval.py
FUSION_WEIGHTS = {
    'semantic': 0.4,    # Dense vector similarity
    'tfidf': 0.3,       # Statistical relevance
    'keyword': 0.2,     # Exact matches
    'bm25': 0.1         # Sparse retrieval
}

# Query expansion settings
MAX_EXPANDED_QUERIES = 5
EXPANSION_CONFIDENCE_THRESHOLD = 0.7
```

## 📈 Performance Improvements

### Speed Optimizations
- Parallel retrieval processing
- Efficient vector operations
- Smart caching strategies
- Optimized chunking algorithms

### Accuracy Improvements
- Multi-modal retrieval fusion
- Intent-aware answer generation
- Context-sensitive processing
- Confidence-based filtering

### Monitoring
- Comprehensive LangSmith integration
- Detailed performance metrics
- Error tracking and logging
- Query analysis and optimization

## 🛠️ Troubleshooting

### Common Issues

#### Advanced Features Not Available
```bash
# Check dependencies
pip install scikit-learn numpy scipy

# Verify installation
python -c "from utils.advanced_retrieval import get_advanced_retriever; print('✅ Advanced retrieval available')"
```

#### Low Confidence Scores
- Increase `k` parameter (more chunks)
- Lower `similarity_threshold`
- Check document quality and chunking
- Verify query clarity and specificity

#### Slow Performance
- Reduce `k` parameter
- Increase `similarity_threshold`
- Use `use_cosine_only=true` for faster retrieval
- Check system resources

## 📋 Best Practices

### Query Optimization
1. **Be Specific**: Use clear, specific questions
2. **Use Keywords**: Include relevant policy terms
3. **Context Matters**: Provide conversation history for better results
4. **Question Types**: Frame questions according to supported types

### System Configuration
1. **Chunking**: Optimize chunk size for your documents
2. **Thresholds**: Tune similarity thresholds based on your needs
3. **Monitoring**: Enable LangSmith for production monitoring
4. **Caching**: Implement caching for frequently asked questions

### Production Deployment
1. **Resources**: Ensure adequate memory for vector operations
2. **Scaling**: Use load balancing for high traffic
3. **Monitoring**: Set up comprehensive logging and monitoring
4. **Backup**: Implement fallback mechanisms for advanced features

## 🔬 Technical Details

### Retrieval Algorithm
1. **Query Analysis**: Intent recognition and keyword extraction
2. **Multi-Modal Search**: Parallel execution of all retrieval methods
3. **Fusion Scoring**: Weighted combination of retrieval scores
4. **Re-ranking**: Context-aware result ordering
5. **Filtering**: Confidence-based result filtering

### Answer Generation Pipeline
1. **Intent Classification**: Determine query type
2. **Information Extraction**: Extract relevant details from chunks
3. **Template Selection**: Choose appropriate answer format
4. **Response Generation**: Create structured, professional answer
5. **Confidence Assessment**: Calculate and display confidence score

---

## 🤝 Contributing

To contribute to the enhanced RAG system:

1. Fork the repository
2. Install enhanced dependencies: `pip install -r requirements_enhanced.txt`
3. Test your changes thoroughly
4. Submit a pull request with detailed description

## 📄 License

This enhanced RAG system is built on top of the existing codebase and follows the same licensing terms.

---

**For technical support or questions about the enhanced features, please refer to the comprehensive logging and monitoring capabilities built into the system.**
