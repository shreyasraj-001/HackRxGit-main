"""
Quick fix main.py to start the server without advanced features
This is a minimal working version for testing
"""

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
import io
import os
import time
import fitz  # PyMuPDF - much faster than pdfplumber
import requests
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import json
from dotenv import load_dotenv
import uvicorn
from langchain_core.documents import Document as LangChainDocument  
from langchain_community.vectorstores import DocArrayInMemorySearch
import asyncio
import uuid
import traceback
from datetime import datetime

# Import our utilities
from utils.splitter import semantic_split
from utils.llm_chain import process_chunk_with_llm_async, llm_processor
from utils.hybrid_retrieval import get_hybrid_retriever

# Load environment variables from .env file if it exists
load_dotenv()

# Simple LangSmith manager (disabled by default)
class LangSmithManager:
    def __init__(self):
        self.enabled = False
        print("ℹ️ LangSmith disabled for quick testing")
    
    def log_document_processing(self, **kwargs):
        pass
    
    def log_question_answer(self, **kwargs):
        pass
    
    def log_batch_questions_answers(self, **kwargs):
        pass
    
    def get_tracer(self):
        return None

# Initialize global LangSmith manager
langsmith_manager = LangSmithManager()

# Pydantic models for the API endpoints
class QueryRequest(BaseModel):
    documents: str 
    questions: List[str]

class SingleQueryRequest(BaseModel):
    question: str
    k: int = 5
    use_cosine_only: bool = False
    similarity_threshold: float = 0.0
    use_advanced: bool = False  # Disabled for quick testing

class BatchQueryResponse(BaseModel):
    answers: List[str]

class PDFDocument: 
    def __init__(self, link: str):
        self.link = link
        self.response = None
        self.parsed_data = ""    
    
    def load_pdf(self):
        """Load PDF from URL with optimized download using PyMuPDF"""
        if not self.response:
            # Optimized download with streaming and timeout
            self.response = requests.get(
                self.link, 
                timeout=10,  # 10 second timeout
                stream=True,  # Stream for large files
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            self.response.raise_for_status()
        
        # Use PyMuPDF (fitz) which is much faster than pdfplumber
        pdf_document = fitz.open(stream=self.response.content, filetype="pdf")
        return pdf_document

    def parse_pdf(self):
        """Parse PDF and extract text content with PyMuPDF - much faster"""
        pdf_document = self.load_pdf()
        text_parts = []
        
        try:
            total_pages = len(pdf_document)
            print(f"📄 Processing {total_pages} pages with PyMuPDF...")
            
            for page_num in range(total_pages):
                try:
                    page = pdf_document[page_num]
                    # Extract text - PyMuPDF is much faster than pdfplumber
                    page_text = page.get_text()
                    
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                        
                except Exception as e:
                    print(f"⚠️ Warning: Failed to extract text from page {page_num + 1}: {e}")
                    continue
            
            self.parsed_data = '\n'.join(text_parts)
            print(f"✅ PyMuPDF extracted text from {len(text_parts)} pages")
            
        finally:
            # Always close the PDF document
            pdf_document.close()
        
        return self.parsed_data

class Embeddings:
    def __init__(self, text: str):
        self.text = text
        # Try Ollama first, fallback to OpenAI if needed
        try:
            self.embeddings = OllamaEmbeddings(
                model="all-minilm:33m",
            )
        except Exception as e:
            print(f"Ollama embeddings failed: {e}")
            # Fallback to OpenAI or use a simpler approach
            self.embeddings = OpenAIEmbeddings(
                base_url="http://127.0.0.1:11434/",
                model="all-minilm:33m",
                api_key="ollama",
            )

    def openai_embed(self):
        return self.embeddings

    def embed_text(self, text: str):
        if self.embeddings:
            return self.embeddings.embed_query(text)
        else:
            return None

class Chunker:
    def __init__(self, text: str):
        self.text = text
        self.chunks = []
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.metadata = []
        self.quality_metrics = {}
        
        # Use enhanced chunking framework with fallbacks
        try:
            print("🧠 Enhanced chunking in progress...")
            
            # Try enhanced adaptive chunking first
            try:
                from utils.enhanced_chunking import (
                    EnhancedChunkingFramework, 
                    ChunkingConfig, 
                    ChunkingStrategy
                )
                
                config = ChunkingConfig(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    strategy=ChunkingStrategy.ADAPTIVE,
                    timeout_seconds=15
                )
                
                framework = EnhancedChunkingFramework(config)
                result = framework.chunk_document(text)
                
                self.chunks = result['chunks']
                self.metadata = result.get('metadata', [])
                self.quality_metrics = result.get('quality_metrics', {})
                
                print(f"✅ Enhanced chunking complete: {len(self.chunks)} chunks created")
                print(f"📊 Quality score: {self.quality_metrics.get('content_preservation_score', 0):.3f}")
                print(f"⏱️ Processing time: {result.get('performance', {}).get('total_time', 0):.2f}s")
                
            except ImportError:
                print("⚠️ Enhanced chunking not available, using adaptive chunking...")
                from utils.splitter import adaptive_split
                self.chunks = adaptive_split(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                print(f"✅ Adaptive chunking complete: {len(self.chunks)} chunks created")
                
            except Exception as e:
                print(f"⚠️ Enhanced chunking failed ({e}), falling back to semantic chunking...")
                self.chunks = semantic_split(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                print(f"✅ Semantic chunking complete: {len(self.chunks)} chunks created")
            
            if self.chunks:
                # Debug: Check chunk quality
                first_chunk = self.chunks[0] if self.chunks else ""
                print(f"📄 First chunk length: {len(first_chunk)}")
                print(f"📄 First chunk preview: {first_chunk[:100]}...")
                
                # Check for empty chunks
                non_empty_chunks = sum(1 for chunk in self.chunks if chunk.strip())
                print(f"📊 Non-empty chunks: {non_empty_chunks}/{len(self.chunks)}")
                
                # Display quality metrics if available
                if self.quality_metrics:
                    print(f"📈 Average chunk length: {self.quality_metrics.get('avg_chunk_length', 0):.1f}")
                    print(f"📏 Length consistency: {self.quality_metrics.get('length_consistency', 0):.3f}")
            else:
                print("⚠️ No chunks created!")
                
        except Exception as e:
            print(f"❌ All chunking methods failed: {e}")
            raise e
    
    def save_chunks(self):
        """Save chunks and metadata to files"""
        # Save chunks
        with open("chunks.json", "w") as f:
            json.dump(self.chunks, f)
        
        # Save metadata if available
        if self.metadata:
            with open("chunks_metadata.json", "w") as f:
                metadata_serializable = []
                for m in self.metadata:
                    if hasattr(m, '__dict__'):
                        metadata_serializable.append(m.__dict__)
                    else:
                        metadata_serializable.append(m)
                json.dump(metadata_serializable, f)
        
        # Save quality metrics if available
        if self.quality_metrics:
            with open("chunks_quality.json", "w") as f:
                json.dump(self.quality_metrics, f)

class VectorStore:
    def __init__(self, chunks: list, embeddings, sig: str = "default"):
        """
        Initialize VectorStore with chunks and embeddings using LangChain in-memory vector store with cosine similarity
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.sig = sig
        self.vector_store = None
        self.distance_strategy = "COSINE"  # Explicitly set cosine similarity
        
        # Convert chunks to Document objects if they're strings
        if chunks and isinstance(chunks[0], str):
            self.documents = [LangChainDocument(page_content=chunk) for chunk in chunks]
        elif chunks and hasattr(chunks[0], 'page_content'):
            # Chunks are already Document objects
            self.documents = chunks
        else:
            # Fallback: convert to strings then to documents
            self.documents = [LangChainDocument(page_content=str(chunk)) for chunk in chunks]
        
        print(f"✅ VectorStore initialized with {len(self.documents)} documents")
        print(f"📏 Distance strategy: {self.distance_strategy} (cosine similarity)")

    def add_chunks(self):
        """Add chunks to the in-memory vector store with cosine similarity"""
        try:
            print(f"➕ Creating in-memory vector store with {len(self.documents)} documents using cosine similarity")
            
            if not self.documents:
                print("❌ No documents to add to vector store")
                return None
            
            # Create DocArrayInMemorySearch vector store from documents with cosine similarity
            self.vector_store = DocArrayInMemorySearch.from_documents(
                documents=self.documents,
                embedding=self.embeddings
            )
            
            print(f"✅ Successfully created in-memory vector store with {len(self.documents)} chunks using cosine similarity")
            print(f"🔍 Distance strategy: COSINE (cosine similarity)")
            return self.vector_store
            
        except Exception as e:
            print(f"❌ Error creating in-memory vector store: {e}")
            traceback.print_exc()
            return None

    def query_chunks(self, query: str, k: int = 5):
        """Query chunks from the in-memory vector store using cosine similarity"""
        try:
            if not self.vector_store:
                print("❌ Vector store not initialized. Call add_chunks() first.")
                return []
            
            print(f"🔍 Querying in-memory vector store with cosine similarity:")
            print(f"   📝 Query: {query[:50]}...")
            print(f"   🔢 k: {k}")
            print(f"   📏 Distance strategy: COSINE")
            
            # Perform similarity search using cosine similarity
            results = self.vector_store.similarity_search(query, k=k)
            print(f"🔍 Query results: {len(results)} documents found using cosine similarity")
            
            if results:
                for i, result in enumerate(results[:2]):
                    content = result.page_content if hasattr(result, 'page_content') else str(result)
                    print(f"📄 Result {i+1}: {content[:100]}...")
            else:
                print("⚠️ No results found")
            
            return results
            
        except Exception as e:
            print(f"❌ Error querying in-memory vector store: {e}")
            traceback.print_exc()
            return []
    
    def similarity_search_with_scores(self, query: str, k: int = 5):
        """Query chunks with cosine similarity scores"""
        try:
            if not self.vector_store:
                print("❌ Vector store not initialized. Call add_chunks() first.")
                return []
            
            print(f"🔍 Querying with cosine similarity scores: {query[:50]}...")
            results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            print(f"📊 Found {len(results_with_scores)} results with cosine similarity scores:")
            for i, (doc, score) in enumerate(results_with_scores[:3]):
                content = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                # DocArrayInMemorySearch uses cosine distance - lower is better
                # Convert to similarity score for better interpretation
                similarity_score = 1 - score if score <= 1 else 1 / (1 + score)
                print(f"   {i+1}. Cosine Distance: {score:.4f} | Similarity: {similarity_score:.4f} | Content: {content}...")
            
            return results_with_scores
            
        except Exception as e:
            print(f"❌ Error querying with scores: {e}")
            return []
    
    def cosine_similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.0):
        """
        Perform cosine similarity search with optional score filtering
        """
        try:
            if not self.vector_store:
                print("❌ Vector store not initialized. Call add_chunks() first.")
                return []
            
            print(f"🎯 Cosine similarity search with threshold {score_threshold}")
            results_with_scores = self.similarity_search_with_scores(query, k=k)
            
            # Filter results by similarity threshold
            filtered_results = []
            for doc, distance in results_with_scores:
                # Convert DocArrayInMemorySearch cosine distance to similarity score
                similarity = 1 - distance if distance <= 1 else 1 / (1 + distance)
                
                if similarity >= score_threshold:
                    filtered_results.append((doc, similarity))
                    
            print(f"🔍 Found {len(filtered_results)} results above threshold {score_threshold}")
            
            # Return just the documents, sorted by similarity (highest first)
            return [doc for doc, _ in sorted(filtered_results, key=lambda x: x[1], reverse=True)]
            
        except Exception as e:
            print(f"❌ Error in cosine similarity search: {e}")
            return []
    
    def get_distance_strategy(self):
        """Get the current distance strategy being used"""
        return getattr(self, 'distance_strategy', 'COSINE')
    
    def verify_cosine_setup(self):
        """Verify that cosine similarity is properly configured"""
        try:
            if not self.vector_store:
                return False, "Vector store not initialized"
            
            # DocArrayInMemorySearch uses cosine similarity by default
            print(f"✅ In-memory vector store distance strategy: {self.distance_strategy}")
            return True, f"In-memory vector store with cosine similarity"
                
        except Exception as e:
            return False, f"Error verifying setup: {e}"

class QueryEngine:
    def __init__(self, vector_store: VectorStore, embeddings, chunks: list):
        """
        Initialize QueryEngine with vector store, embeddings, and chunks for basic retrieval using cosine similarity.
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.chunks = chunks
        
        # Verify cosine similarity setup
        is_cosine, message = self.vector_store.verify_cosine_setup()
        print(f"🔍 Cosine similarity verification: {message}")
        
        # Initialize standard hybrid retriever
        try:
            self.hybrid_retriever = get_hybrid_retriever(
                self.vector_store.vector_store, self.chunks
            )
            print("✅ Hybrid retriever initialized")
        except Exception as e:
            print(f"⚠️ Hybrid retriever initialization failed: {e}")
            self.hybrid_retriever = None
        
        print(f"✅ QueryEngine initialized with standard cosine similarity-based retrieval")
        
    def retrieve_relevant_chunks(self, query: str, k: int = 5, use_cosine_only: bool = False, 
                               similarity_threshold: float = 0.0):
        """
        Retrieve relevant chunks from vector store using standard retrieval methods.
        """
        try:
            print(f"🔍 Retrieving relevant chunks for query: {query[:50]}...")
            print(f"🎯 Mode: {'Pure cosine similarity' if use_cosine_only else 'Hybrid (BM25 + cosine)'}")
            if similarity_threshold > 0:
                print(f"📊 Similarity threshold: {similarity_threshold}")
            
            if use_cosine_only:
                # Use pure cosine similarity search
                relevant_chunks = self.vector_store.cosine_similarity_search(
                    query, k=k, score_threshold=similarity_threshold
                )
            elif self.hybrid_retriever:
                # Use hybrid retriever (BM25 + in-memory vector store cosine similarity)
                self.hybrid_retriever.k = k
                relevant_chunks = self.hybrid_retriever.get_relevant_documents(query)
            else:
                # Fallback to simple vector store query with cosine similarity
                relevant_chunks = self.vector_store.query_chunks(query, k=k)
            
            if relevant_chunks:
                print(f"✅ Found {len(relevant_chunks)} relevant chunks")
                return relevant_chunks
            else:
                print("⚠️ No relevant chunks found")
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving chunks: {e}")
            return []
    
    def generate_context(self, relevant_chunks: list) -> str:
        """
        Combine relevant chunks into a coherent context
        """
        if not relevant_chunks:
            return "No relevant information found in the document."
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            context_parts.append(f"[Context {i}]\n{content.strip()}")
        
        return "\n\n".join(context_parts)
    
    def simple_answer_extraction(self, query: str, context: str) -> str:
        """
        Simple rule-based answer extraction
        """
        if not context or context == "No relevant information found in the document.":
            return "I couldn't find relevant information in the document to answer your question."
        
        # Simple keyword-based extraction for common policy questions
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Look for specific patterns in policy documents
        if "grace period" in query_lower:
            # Look for grace period information
            lines = context.split('\n')
            for line in lines:
                if "grace period" in line.lower() or "days" in line.lower():
                    return f"Based on the document: {line.strip()}"
        
        elif "premium" in query_lower and ("payment" in query_lower or "pay" in query_lower):
            # Look for premium payment information
            lines = context.split('\n')
            for line in lines:
                if "premium" in line.lower() and ("payment" in line.lower() or "pay" in line.lower()):
                    return f"Based on the document: {line.strip()}"
        
        elif "coverage" in query_lower or "cover" in query_lower:
            # Look for coverage information
            lines = context.split('\n')
            for line in lines:
                if "coverage" in line.lower() or "covered" in line.lower():
                    return f"Based on the document: {line.strip()}"
        
        # Default: return the most relevant context chunk
        context_parts = context.split('[Context')
        if len(context_parts) > 1:
            first_context = context_parts[1].split('\n', 1)
            if len(first_context) > 1:
                return f"Based on the document: {first_context[1].strip()[:500]}..."
        
        return f"Based on the retrieved information: {context[:500]}..."
    
    def answer_query(self, query: str, k: int = 5, use_llm: bool = False, use_cosine_only: bool = False, 
                   similarity_threshold: float = 0.0, use_advanced: bool = False) -> dict:
        """
        Complete pipeline to answer a user query using basic retrieval
        """
        start_time = time.time()
        try:
            print(f"🤖 Processing query with standard retrieval: {query}")
            retrieval_mode = f"{'Pure cosine' if use_cosine_only else 'Hybrid (BM25 + cosine)'}"
            print(f"🎯 Retrieval mode: {retrieval_mode}")
            
            # Step 1: Retrieve relevant chunks
            retrieval_start = time.time()
            relevant_chunks = self.retrieve_relevant_chunks(
                query, k=k, use_cosine_only=use_cosine_only, 
                similarity_threshold=similarity_threshold
            )
            retrieval_time = time.time() - retrieval_start
            
            # Step 2: Generate context
            context_start = time.time()
            context = self.generate_context(relevant_chunks)
            context_time = time.time() - context_start
            
            # Step 3: Generate answer
            answer_start = time.time()
            answer = self.simple_answer_extraction(query, context)
            answer_time = time.time() - answer_start
            
            total_time = time.time() - start_time
            
            # Prepare response
            response = {
                "query": query,
                "answer": answer,
                "context": context,
                "num_chunks_retrieved": len(relevant_chunks),
                "success": True,
                "processing_time": total_time,
                "retrieval_mode": retrieval_mode,
                "performance": {
                    "retrieval_time": retrieval_time,
                    "context_time": context_time,
                    "answer_time": answer_time,
                    "total_time": total_time
                }
            }
            
            return response
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ Error processing query: {e}")
            
            error_response = {
                "query": query,
                "answer": "An error occurred while processing your query.",
                "context": "",
                "num_chunks_retrieved": 0,
                "success": False,
                "error": str(e),
                "processing_time": total_time
            }
            
            return error_response

# FastAPI app
app = FastAPI()

# Global variables to store the initialized components
global_vector_store = None
global_embeddings = None
global_query_engine = None

@app.get("/")
def root():
    """Initialize the RAG system with the policy document"""
    global global_vector_store, global_embeddings, global_query_engine
    
    start_time = time.time()
    
    file_link = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        print("📄 Loading and processing PDF document...")
        document_start = time.time()
        document = PDFDocument(file_link)
        data = document.parse_pdf()
        document_time = time.time() - document_start
        print(f"✅ PDF processed: {len(data)} characters")

        print("✂️ Chunking document...")
        chunking_start = time.time()
        chunker = Chunker(data)
        chunking_time = time.time() - chunking_start
        print(f"✅ Chunking complete: {len(chunker.chunks)} chunks")
        
        print("🧠 Creating embeddings...")
        embedding_start = time.time()
        embed = Embeddings(data)
        embeddings_data = embed.embed_text(data)
        embedding_time = time.time() - embedding_start
        print(f"✅ Embeddings created: {len(embeddings_data) if embeddings_data else 0} dimensions")

        # Store global references
        print("🗄️ Initializing vector store...")
        vectorstore_start = time.time()
        global_vector_store = VectorStore(chunker.chunks, embed.embeddings, "policy_document")
        global_embeddings = embed.embeddings
        
        print("📚 Adding documents to vector store...")
        vector_store_result = global_vector_store.add_chunks()
        vectorstore_time = time.time() - vectorstore_start

        # Initialize QueryEngine with chunks for hybrid retrieval
        global_query_engine = QueryEngine(global_vector_store, global_embeddings, global_vector_store.documents)

        print("🔍 Testing with sample query...")
        # Test query using the new QueryEngine
        test_response = global_query_engine.answer_query(
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
        )

        total_time = time.time() - start_time

        response = {
            "message": "RAG system initialized successfully with in-memory vector store using cosine similarity",
            "embeddings_size": len(embeddings_data) if embeddings_data else 0,
            "chunks_count": len(chunker.chunks),
            "vector_store_type": "DocArrayInMemorySearch (In-Memory, Cosine Similarity)",
            "distance_strategy": global_vector_store.get_distance_strategy(),
            "vector_store_success": vector_store_result is not None,
            "system_ready": True,
            "cosine_similarity": True,
            "test_query": test_response,
            "processing_times": {
                "document_parsing": document_time,
                "chunking": chunking_time,
                "embeddings": embedding_time,
                "vector_store": vectorstore_time,
                "total": total_time
            }
        }
        
        return response
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Failed to initialize RAG system: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "message": error_msg,
            "embeddings_size": 0,
            "chunks_count": 0,
            "vector_store_type": "None",
            "distance_strategy": "None",
            "vector_store_success": False,
            "system_ready": False,
            "cosine_similarity": False,
            "test_query": None,
            "error": str(e),
            "processing_time": total_time
        }

@app.post("/query")
def query_document(request: dict):
    """
    Query the policy document with a user question using enhanced retrieval
    """
    global global_query_engine
    
    if not global_query_engine:
        return {
            "error": "RAG system not initialized. Please call the root endpoint (/) first.",
            "success": False
        }
    
    question = request.get("question", "")
    k = request.get("k", 5)
    use_cosine_only = request.get("use_cosine_only", False)
    similarity_threshold = request.get("similarity_threshold", 0.0)
    use_advanced = request.get("use_advanced", False)  # Disabled for now
    
    if not question:
        return {
            "error": "Please provide a question in the request body.",
            "success": False
        }
    
    # Validate similarity threshold
    if not (0.0 <= similarity_threshold <= 1.0):
        return {
            "error": "similarity_threshold must be between 0.0 and 1.0",
            "success": False
        }
    
    response = global_query_engine.answer_query(
        question, k=k, use_cosine_only=use_cosine_only, 
        similarity_threshold=similarity_threshold, use_advanced=use_advanced
    )
    return response

@app.get("/query/{question}")
def query_document_get(question: str, k: int = 5, use_advanced: bool = False):
    """
    Query the policy document with a user question (GET endpoint)
    """
    global global_query_engine
    
    if not global_query_engine:
        return {
            "error": "RAG system not initialized. Please call the root endpoint (/) first.",
            "success": False
        }
    
    response = global_query_engine.answer_query(question, k=k, use_advanced=use_advanced)
    return response

if __name__ == "__main__":
    print("🚀 Starting Enhanced RAG System...")
    print("📍 Access the system at: http://localhost:8000")
    print("📍 Initialize the system: GET http://localhost:8000/")
    print("📍 Query the system: POST http://localhost:8000/query")
    uvicorn.run(app, host="0.0.0.0", port=8000)
