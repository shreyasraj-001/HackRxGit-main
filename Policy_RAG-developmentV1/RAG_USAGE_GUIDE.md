# 🚀 RAG Retrieval System - Usage Guide

## 📋 Overview
Your RAG system now has complete retrieval and answer generation capabilities! Here's how to use it:

## 🔧 Components Added

### 1. **QueryEngine Class**
- **Purpose**: Main class for handling question-answering
- **Features**:
  - Retrieves relevant chunks from vector database
  - Generates coherent context from multiple chunks
  - Provides rule-based answer extraction
  - Ready for LLM integration

### 2. **FastAPI Endpoints**

#### **GET /** - Initialize System
- Loads and processes the PDF document
- Creates vector embeddings and stores them
- Returns system status and test query result

#### **POST /query** - Query with JSON
```json
{
    "question": "What is the grace period for premium payment?",
    "k": 5
}
```

#### **GET /query/{question}** - Simple Query
- Direct URL-based querying
- Example: `GET /query/What is the grace period?`

## 🚀 Getting Started

### 1. **Start the Server**
```bash
cd "c:\Users\tanoj\OneDrive - K L University\Desktop\Bajaj_Hackthon\Policy_RAG"
uvicorn main:app --reload --port 8000
```

### 2. **Initialize the System**
```bash
curl http://localhost:8000/
```

### 3. **Ask Questions**
```bash
# Using GET endpoint
curl "http://localhost:8000/query/What is the grace period for premium payment?"

# Using POST endpoint
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the coverage benefits?", "k": 3}'
```

## 🎯 Example Queries

### **Policy Information**
- "What is the grace period for premium payment?"
- "What are the coverage benefits under this policy?"
- "What is the sum insured amount?"
- "What is the policy term and renewal process?"

### **Claims Related**
- "How do I make a claim?"
- "What documents are required for claims?"
- "What is the claim settlement process?"
- "Are there any claim exclusions?"

### **Premium & Payment**
- "How much is the premium amount?"
- "What are the payment modes available?"
- "What happens if I miss a premium payment?"

## 📊 Response Format

```json
{
    "query": "What is the grace period?",
    "answer": "Based on the document: The grace period for premium payment is 30 days...",
    "context": "[Context 1]\nRelevant document content...\n\n[Context 2]\nMore relevant content...",
    "num_chunks_retrieved": 3,
    "success": true
}
```

## 🧠 How It Works

### **Step 1: Query Processing**
1. User asks a question
2. Question is converted to embeddings
3. Vector similarity search finds relevant chunks

### **Step 2: Context Generation**
1. Retrieved chunks are combined into coherent context
2. Context is formatted with clear sections
3. Most relevant information is prioritized

### **Step 3: Answer Generation**
1. **Current**: Rule-based extraction using keyword matching
2. **Future**: LLM-based answer generation using context
3. Provides specific, relevant answers from document content

## 🔧 Testing

### **Run the Test Suite**
```bash
python test_retrieval.py
```

### **Options**:
1. **Server-based testing**: Tests the FastAPI endpoints
2. **Local component testing**: Tests individual classes
3. **Both**: Comprehensive testing

## 🎯 Key Features

### **Intelligent Chunking**
- Uses semantic chunking for better context preservation
- Falls back to recursive splitting if needed
- Optimal chunk size for retrieval

### **Robust Retrieval**
- Vector similarity search using Pinecone
- Configurable number of chunks to retrieve
- Context combination from multiple sources

### **Smart Answer Extraction**
- Rule-based patterns for common policy questions
- Keyword matching for specific information
- Fallback to most relevant context

### **Error Handling**
- Graceful fallbacks at every step
- Detailed error messages and logging
- System state validation

## 🚀 Production Enhancements

### **Add LLM Integration**
```python
# Future enhancement: Replace rule-based extraction
def llm_answer_generation(self, query: str, context: str) -> str:
    # Use Ollama or OpenAI for better answer generation
    prompt = self.create_prompt(query, context)
    # Call LLM API here
    return llm_response
```

### **Caching**
- Cache frequently asked questions
- Store embeddings for faster retrieval
- Session-based context memory

### **Analytics**
- Track popular queries
- Monitor answer quality
- Performance metrics

## 🎉 Ready to Use!

Your RAG system is now complete with:
- ✅ Document processing and chunking
- ✅ Vector storage and retrieval  
- ✅ Question answering pipeline
- ✅ REST API endpoints
- ✅ Comprehensive testing
- ✅ Error handling and fallbacks

Start the server and begin asking questions about your policy documents! 🚀
