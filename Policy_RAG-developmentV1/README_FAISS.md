# 🚀 Policy RAG with FAISS Vector Store

## 📋 Overview
A complete Retrieval-Augmented Generation (RAG) system for policy documents using **FAISS in-memory vector store** instead of Pinecone. This eliminates the need for external cloud services and API keys!

## 🎯 Key Features
- ✅ **No External Dependencies**: Uses FAISS in-memory vector store
- ✅ **Semantic Chunking**: Intelligent text splitting using LangChain Experimental
- ✅ **Local Embeddings**: Ollama-based embeddings (with OpenAI fallback)
- ✅ **FastAPI Web Service**: REST API for querying documents
- ✅ **Real-time Processing**: Process documents and answer questions instantly
- ✅ **Robust Error Handling**: Fallback mechanisms at every step

## 🛠️ Installation

### Option 1: Automatic Installation
```bash
cd "c:\Users\tanoj\OneDrive - K L University\Desktop\Bajaj_Hackthon\Policy_RAG"
python install_faiss_deps.py
```

### Option 2: Manual Installation
```bash
pip install fastapi uvicorn requests pdfplumber langchain langchain-openai langchain-experimental langchain-text-splitters langchain-ollama langchain-community langchain-core python-dotenv faiss-cpu
```

## 🚀 Quick Start

### 1. Start the Server
```bash
uvicorn main:app --reload --port 8000
```

### 2. Initialize the System
```bash
curl http://localhost:8000/
```

### 3. Ask Questions
```bash
# Simple GET query
curl "http://localhost:8000/query/What is the grace period for premium payment?"

# POST query with JSON
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the coverage benefits?", "k": 3}'
```

## 🏗️ System Architecture

```
PDF Document → Text Extraction → Semantic Chunking → FAISS Vector Store → Query Engine → Answers
```

### Components:
1. **PDFDocument**: Downloads and extracts text from PDF URLs
2. **Chunker**: Uses SemanticChunker for intelligent text splitting
3. **Embeddings**: Ollama-based embeddings (local, no API needed)
4. **VectorStore**: FAISS in-memory vector database
5. **QueryEngine**: Retrieval and answer generation pipeline
6. **FastAPI**: REST API endpoints

## 📊 API Endpoints

### GET `/`
Initialize the RAG system and process the policy document.

**Response:**
```json
{
    "message": "RAG system initialized successfully with FAISS vector store",
    "embeddings_size": 384,
    "chunks_count": 42,
    "vector_store_type": "FAISS (In-Memory)",
    "vector_store_success": true,
    "system_ready": true,
    "test_query": {
        "query": "What is the grace period?",
        "answer": "Based on the document: ...",
        "success": true
    }
}
```

### POST `/query`
Query the document with JSON payload.

**Request:**
```json
{
    "question": "What is the premium amount?",
    "k": 5
}
```

**Response:**
```json
{
    "query": "What is the premium amount?",
    "answer": "Based on the document: The premium amount is...",
    "context": "[Context 1]\nRelevant content...",
    "num_chunks_retrieved": 3,
    "success": true
}
```

### GET `/query/{question}`
Direct URL-based querying.

**Example:** `GET /query/What are the exclusions?`

## 🧪 Testing

### Run the Test Suite
```bash
python test_retrieval.py
```

### Test Individual Components
```bash
python test_text_content.py
```

## 🎯 Advantages of FAISS Implementation

### ✅ **No External Services**
- No Pinecone account or API keys required
- No network dependencies for vector operations
- Faster setup and deployment

### ✅ **In-Memory Performance**
- Ultra-fast similarity search
- No network latency
- Ideal for development and small-scale deployments

### ✅ **Simplicity**
- Single-file deployment
- No configuration files needed
- Easy to understand and modify

### ✅ **Cost-Effective**
- No subscription fees
- No usage limits
- Perfect for prototyping and demos

## 🔧 How It Works

### 1. **Document Processing**
```python
document = PDFDocument(pdf_url)
text = document.parse_pdf()
```

### 2. **Semantic Chunking**
```python
chunker = Chunker(text)  # Creates ~42 intelligent chunks
```

### 3. **Vector Storage**
```python
vector_store = VectorStore(chunks, embeddings)
vector_store.add_chunks()  # Creates FAISS index
```

### 4. **Query Processing**
```python
query_engine = QueryEngine(vector_store, embeddings)
response = query_engine.answer_query("Your question here")
```

## 🎪 Example Queries

### Policy Information
- "What is the grace period for premium payment?"
- "What are the coverage benefits?"
- "What is the sum insured amount?"
- "What is the policy term?"

### Claims & Procedures
- "How do I make a claim?"
- "What documents are required?"
- "What is the claim settlement process?"
- "Are there any exclusions?"

### Premium & Payment
- "How much is the premium?"
- "What are the payment modes?"
- "What happens if I miss a payment?"

## 🚀 Production Deployment

### For Production Use:
1. **Consider FAISS with Disk Storage** for persistence
2. **Add Caching** for frequently asked questions
3. **Implement User Sessions** for context memory
4. **Add Analytics** for query tracking
5. **Scale with Load Balancing** for multiple instances

### FAISS Persistence (Optional Enhancement):
```python
# Save FAISS index to disk
vector_store.vector_store.save_local("faiss_index")

# Load from disk
vector_store = FAISS.load_local("faiss_index", embeddings)
```

## 🎉 Ready to Use!

Your FAISS-based RAG system is now:
- ✅ **Self-contained** (no external services)
- ✅ **Fast** (in-memory operations)
- ✅ **Simple** (easy setup and deployment)
- ✅ **Scalable** (can handle multiple documents)
- ✅ **Production-ready** (with proper error handling)

Start asking questions about your policy documents! 🚀

## 🔍 Troubleshooting

### Common Issues:
1. **ModuleNotFoundError**: Run the installation script
2. **Ollama Connection Error**: Install Ollama or use OpenAI fallback
3. **PDF Loading Error**: Check the PDF URL accessibility
4. **Empty Results**: Verify document processing and chunking

### Debug Mode:
The system provides detailed logging for each step. Watch the console output to track the processing pipeline.

---

**Happy Querying!** 🎯
