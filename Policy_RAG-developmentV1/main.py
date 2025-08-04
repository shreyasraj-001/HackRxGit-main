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

# LangSmith imports for monitoring and tracing
try:
    from langsmith import Client as LangSmithClient
    from langchain.callbacks import LangChainTracer
    from langchain_core.tracers.langchain import LangChainTracer
    LANGSMITH_AVAILABLE = True
except ImportError:
    print("⚠️ LangSmith not available. Install with: pip install langsmith")
    LANGSMITH_AVAILABLE = False

import uuid
from datetime import datetime

# Import async CSV logger
try:
    from utils.async_csv_logger import csv_logger
    CSV_LOGGING_AVAILABLE = True
    print("✅ Async CSV Logger initialized")
except ImportError:
    CSV_LOGGING_AVAILABLE = False
    print("⚠️ Async CSV Logger not available")

# Import our utilities
from utils.splitter import semantic_split
from utils.llm_chain import process_chunk_with_llm_async, llm_processor
from utils.hybrid_retrieval import get_hybrid_retriever

# Import advanced retrieval and answer generation (with fallbacks)
try:
    from utils.advanced_retrieval import get_advanced_retriever
    ADVANCED_RETRIEVAL_AVAILABLE = True
    print("✅ Full advanced retrieval features loaded")
except ImportError:
    try:
        from utils.simple_advanced_retrieval import get_advanced_retriever
        ADVANCED_RETRIEVAL_AVAILABLE = True
        print("✅ Simplified advanced retrieval features loaded")
    except ImportError as e:
        print(f"⚠️ Advanced retrieval not available: {e}")
        ADVANCED_RETRIEVAL_AVAILABLE = False

try:
    from utils.advanced_answer_generation import get_advanced_answer_generator
    ADVANCED_ANSWER_AVAILABLE = True
    print("✅ Advanced answer generation loaded")
except ImportError as e:
    print(f"⚠️ Advanced answer generation not available: {e}")
    ADVANCED_ANSWER_AVAILABLE = False

ADVANCED_FEATURES_AVAILABLE = ADVANCED_RETRIEVAL_AVAILABLE or ADVANCED_ANSWER_AVAILABLE

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize LangSmith if available
class LangSmithManager:
    def __init__(self):
        self.enabled = LANGSMITH_AVAILABLE and os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.client = None
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "policy-rag-cosine-similarity")
        self.store_documents = os.getenv("LANGSMITH_STORE_DOCUMENTS", "true").lower() == "true"
        self.store_questions = os.getenv("LANGSMITH_STORE_QUESTIONS", "true").lower() == "true"
        
        if self.enabled:
            try:
                self.client = LangSmithClient()
                print(f"✅ LangSmith initialized for project: {self.project_name}")
            except Exception as e:
                print(f"⚠️ LangSmith initialization failed: {e}")
                self.enabled = False
        else:
            print("ℹ️ LangSmith disabled or not available")
    
    def log_document_processing(self, document_url: str, document_type: str, chunks_created: int, 
                              embeddings_dimension: int, processing_time: float, metadata: dict = None):
        """Log document processing to LangSmith and CSV"""
        # LangSmith logging
        if not self.enabled or not self.store_documents:
            pass
        else:
            try:
                run_id = str(uuid.uuid4())
                self.client.create_run(
                    name="document_processing",
                    run_type="chain",
                    inputs={
                        "document_url": document_url,
                        "document_type": document_type,
                        "embeddings_dimension": embeddings_dimension,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata or {}
                    },
                    outputs={
                        "success": True,
                        "chunks_created": chunks_created,
                        "processing_time": processing_time,
                        "embeddings_dimension": embeddings_dimension
                    },
                    project_name=self.project_name,
                    run_id=run_id
                )
                print(f"📊 LangSmith: Document processing logged (run_id: {run_id[:8]}...)")
            except Exception as e:
                print(f"⚠️ LangSmith document logging failed: {e}")
        
        # Async CSV logging (fire-and-forget)
        if CSV_LOGGING_AVAILABLE:
            csv_logger.log_document_background(
                document_url=document_url,
                document_type=document_type,
                document_length=metadata.get('total_document_length', 0) if metadata else 0,
                chunks_created=chunks_created,
                embeddings_dimension=embeddings_dimension,
                processing_time=processing_time,
                metadata=metadata
            )
    
    def log_question_answer(self, question: str, answer: str, context: str, metadata: dict = None):
        """Log question-answer pairs to LangSmith and CSV"""
        # LangSmith logging
        if not self.enabled or not self.store_questions:
            pass
        else:
            try:
                run_id = str(uuid.uuid4())
                self.client.create_run(
                    name="question_answering",
                    run_type="chain",
                    inputs={
                        "question": question,
                        "context_length": len(context),
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata or {}
                    },
                    outputs={
                        "answer": answer,
                        "context": context[:1000] + "..." if len(context) > 1000 else context,
                        "success": True,
                        "chunks_used": metadata.get("chunks_used") if metadata else None,
                        "retrieval_mode": metadata.get("retrieval_mode") if metadata else None,
                        "similarity_threshold": metadata.get("similarity_threshold") if metadata else None
                    },
                    project_name=self.project_name,
                    run_id=run_id
                )
                print(f"📊 LangSmith: Q&A logged (run_id: {run_id[:8]}...)")
            except Exception as e:
                print(f"⚠️ LangSmith Q&A logging failed: {e}")
        
        # Async CSV logging (fire-and-forget)
        if CSV_LOGGING_AVAILABLE:
            csv_logger.log_question_background(
                question=question,
                answer=answer,
                context_length=len(context),
                chunks_used=metadata.get("chunks_used") if metadata else None,
                retrieval_mode=metadata.get("retrieval_mode") if metadata else None,
                similarity_threshold=metadata.get("similarity_threshold") if metadata else None,
                processing_time=metadata.get("processing_time") if metadata else None,
                success=metadata.get("success", True) if metadata else True,
                metadata=metadata
            )
    
    def log_batch_questions_answers(self, questions: List[str], answers: List[str], 
                                  document_context: str, metadata: dict = None):
        """Log batch Q&A processing to LangSmith and CSV"""
        # LangSmith logging
        if not self.enabled:
            pass
        else:
            try:
                run_id = str(uuid.uuid4())
                self.client.create_run(
                    name="batch_questions_answers",
                    run_type="chain",
                    inputs={
                        "questions_count": len(questions),
                        "questions": questions,
                        "context_length": len(document_context),
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata or {}
                    },
                    outputs={
                        "answers": answers,
                        "success": True,
                        "context": document_context[:1000] + "..." if len(document_context) > 1000 else document_context,
                        "processing_time": metadata.get("total_processing_time") if metadata else None,
                        "questions_per_second": metadata.get("questions_per_second") if metadata else None,
                        "avg_time_per_question": metadata.get("avg_time_per_question") if metadata else None
                    },
                    project_name=self.project_name,
                    run_id=run_id
                )
                print(f"📊 LangSmith: Batch Q&A logged (run_id: {run_id[:8]}...)")
            except Exception as e:
                print(f"⚠️ LangSmith batch logging failed: {e}")
        
        # Async CSV logging (fire-and-forget)
        if CSV_LOGGING_AVAILABLE:
            csv_logger.log_batch_background(
                document_url=metadata.get("document_url", "unknown") if metadata else "unknown",
                questions=questions,
                answers=answers,
                total_processing_time=metadata.get("total_processing_time", 0) if metadata else 0,
                questions_per_second=metadata.get("questions_per_second", 0) if metadata else 0,
                success_count=metadata.get("success_count", len(answers)) if metadata else len(answers),
                error_count=metadata.get("error_count", 0) if metadata else 0,
                metadata={
                    **metadata,
                    "context_length": len(document_context)
                } if metadata else {"context_length": len(document_context)}
            )
    
    def get_tracer(self):
        """Get LangChain tracer for automatic tracing"""
        if self.enabled and LANGSMITH_AVAILABLE:
            try:
                return LangChainTracer(project_name=self.project_name)
            except:
                return None
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
    use_advanced: bool = True  # New parameter for advanced retrieval

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
                # dimensions=1536,
            )

        except Exception as e:
            print(f"Ollama embeddings failed: {e}")
            # Fallback to OpenAI or use a simpler approach
            self.embeddings = OpenAIEmbeddings(
                base_url="http://127.0.0.1:11434/",
                model="all-minilm:33m",
                api_key="ollama",
                # dimensions=1536,
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
            import traceback
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
            import traceback
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
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            score_threshold (float): Minimum similarity score threshold (0.0 to 1.0)
            
        Returns:
            list: Filtered results based on cosine similarity
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
        Initialize QueryEngine with vector store, embeddings, and chunks for advanced hybrid retrieval using cosine similarity.
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.chunks = chunks
        
        # Verify cosine similarity setup
        is_cosine, message = self.vector_store.verify_cosine_setup()
        print(f"🔍 Cosine similarity verification: {message}")
        
        # Initialize standard hybrid retriever
        self.hybrid_retriever = get_hybrid_retriever(
            self.vector_store.vector_store, self.chunks
        )
        
        # Initialize advanced retriever if available
        if ADVANCED_RETRIEVAL_AVAILABLE:
            try:
                self.advanced_retriever = get_advanced_retriever(
                    self.vector_store.vector_store, 
                    self.chunks, 
                    self.embeddings
                )
                print("✅ Advanced retrieval system initialized")
            except Exception as e:
                print(f"⚠️ Advanced retrieval initialization failed: {e}")
                self.advanced_retriever = None
        else:
            self.advanced_retriever = None
        
        # Initialize advanced answer generator if available
        if ADVANCED_ANSWER_AVAILABLE:
            try:
                self.advanced_answer_generator = get_advanced_answer_generator()
                print("✅ Advanced answer generation initialized")
            except Exception as e:
                print(f"⚠️ Advanced answer generation initialization failed: {e}")
                self.advanced_answer_generator = None
        else:
            self.advanced_answer_generator = None
        
        print(f"✅ QueryEngine initialized with {'advanced' if self.advanced_retriever else 'standard'} cosine similarity-based retrieval")
        
    def retrieve_relevant_chunks(self, query: str, k: int = 5, use_cosine_only: bool = False, 
                               similarity_threshold: float = 0.0, use_advanced: bool = True):
        """
        Retrieve relevant chunks from vector store using advanced or standard retrieval methods.
        
        Args:
            query (str): User's question
            k (int): Number of relevant chunks to retrieve
            use_cosine_only (bool): If True, use only vector store cosine similarity (no BM25)
            similarity_threshold (float): Minimum cosine similarity threshold (0.0 to 1.0)
            use_advanced (bool): If True, use advanced retrieval techniques
            
        Returns:
            list: List of relevant document chunks with scores
        """
        try:
            print(f"🔍 Retrieving relevant chunks for query: {query[:50]}...")
            
            # Use advanced retrieval if available and requested
            if use_advanced and self.advanced_retriever:
                print(f"🚀 Using advanced fusion retrieval")
                results_with_scores = self.advanced_retriever.retrieve_with_fusion(
                    query, k=k, similarity_threshold=similarity_threshold
                )
                print(f"✅ Advanced retrieval found {len(results_with_scores)} results")
                return [doc for doc, score in results_with_scores]
            
            # Fallback to standard retrieval methods
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
        
        Args:
            relevant_chunks (list): List of relevant document chunks
            
        Returns:
            str: Combined context string
        """
        if not relevant_chunks:
            return "No relevant information found in the document."
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            context_parts.append(f"[Context {i}]\n{content.strip()}")
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for answer generation
        
        Args:
            query (str): User's question
            context (str): Retrieved context from documents
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"""You are an expert AI assistant for analyzing policy documents. Your goal is to provide clear, accurate, and helpful answers based on the provided text.

**Context from the policy document:**
---
{context}
---

**User's Question:**
{query}

**Instructions for Generating the Answer:**

1.  **Analyze the Context:** Carefully read the provided context. Your answer **must** be based *only* on this information. Do not use any external knowledge.
2.  **Directly Answer the Question:** Start your response with a direct answer to the user's question.
3.  **Provide Supporting Details:** After the direct answer, provide key details, definitions, and relevant clauses from the context that support your answer. Use bullet points or short paragraphs for clarity.
4.  **Cite Sources:** When you extract specific information, mention the context block it came from (e.g., "As stated in [Context 1]...").
5.  **Handle Missing Information:** If the answer cannot be found in the provided context, you **must** state: "Based on the provided document, the information to answer this question is not available." Do not try to guess or infer information that isn't present.
6.  **Maintain a Professional Tone:** Be formal, clear, and concise in your language.

**Answer:**
"""
        
        return prompt
    
    def generate_enhanced_answer(self, query: str, relevant_chunks: list) -> str:
        """
        Generate enhanced answer using advanced answer generation techniques
        
        Args:
            query (str): User's question
            relevant_chunks (list): List of relevant document chunks
            
        Returns:
            str: Enhanced structured answer
        """
        if self.advanced_answer_generator and relevant_chunks:
            try:
                # Calculate average confidence from retrieval
                retrieval_confidence = 0.8 if len(relevant_chunks) >= 3 else 0.6
                
                # Generate structured answer
                answer_data = self.advanced_answer_generator.generate_structured_answer(
                    query, relevant_chunks, retrieval_confidence
                )
                
                enhanced_answer = answer_data['answer']
                confidence = answer_data['confidence']
                intent = answer_data.get('intent', 'general')
                
                # Add confidence indicator
                confidence_indicator = "🟢 High" if confidence > 0.7 else "🟡 Medium" if confidence > 0.4 else "🔴 Low"
                enhanced_answer += f"\n\n**Confidence Level:** {confidence_indicator} ({confidence:.1%})"
                enhanced_answer += f"\n**Query Type:** {intent.title()}"
                
                print(f"✅ Enhanced answer generated with {confidence:.1%} confidence ({intent} type)")
                return enhanced_answer
                
            except Exception as e:
                print(f"⚠️ Enhanced answer generation failed: {e}")
        # Fallback to simple extraction
        return self.simple_answer_extraction(query, self.generate_context(relevant_chunks))
    
    def simple_answer_extraction(self, query: str, context: str) -> str:
        """
        Simple rule-based answer extraction (fallback when no LLM is available)
        
        Args:
            query (str): User's question
            context (str): Retrieved context
            
        Returns:
            str: Extracted answer
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
                   similarity_threshold: float = 0.0, use_advanced: bool = True) -> dict:
        """
        Complete pipeline to answer a user query using enhanced retrieval with LangSmith logging
        
        Args:
            query (str): User's question
            k (int): Number of chunks to retrieve
            use_llm (bool): Whether to use LLM for answer generation
            use_cosine_only (bool): If True, use only vector store cosine similarity (no BM25)
            similarity_threshold (float): Minimum cosine similarity threshold (0.0 to 1.0)
            use_advanced (bool): If True, use advanced retrieval and answer generation
            
        Returns:
            dict: Complete response with query, context, and answer
        """
        start_time = time.time()
        try:
            print(f"🤖 Processing query with {'advanced' if use_advanced else 'standard'} retrieval: {query}")
            retrieval_mode = f"{'Advanced fusion' if use_advanced and self.advanced_retriever else 'Pure cosine' if use_cosine_only else 'Hybrid (BM25 + cosine)'}"
            print(f"🎯 Retrieval mode: {retrieval_mode}")
            
            # Step 1: Retrieve relevant chunks using enhanced retrieval
            retrieval_start = time.time()
            relevant_chunks = self.retrieve_relevant_chunks(
                query, k=k, use_cosine_only=use_cosine_only, 
                similarity_threshold=similarity_threshold, use_advanced=use_advanced
            )
            retrieval_time = time.time() - retrieval_start
            
            # Step 2: Generate context
            context_start = time.time()
            context = self.generate_context(relevant_chunks)
            context_time = time.time() - context_start
            
            # Step 3: Generate enhanced answer
            answer_start = time.time()
            if use_llm:
                # TODO: Implement LLM-based answer generation using Ollama
                prompt = self.create_prompt(query, context)
                answer = "LLM integration not implemented yet. Using enhanced rule-based extraction."
                answer = self.generate_enhanced_answer(query, relevant_chunks)
            else:
                # Use enhanced answer generation
                answer = self.generate_enhanced_answer(query, relevant_chunks)
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
            
            # Log to LangSmith
            langsmith_metadata = {
                "chunks_used": len(relevant_chunks),
                "retrieval_mode": retrieval_mode,
                "similarity_threshold": similarity_threshold,
                "k": k,
                "processing_time": total_time,
                "retrieval_time": retrieval_time,
                "context_time": context_time,
                "answer_time": answer_time,
                "use_llm": use_llm,
                "use_advanced": use_advanced and self.advanced_retriever is not None
            }
            
            langsmith_manager.log_question_answer(
                question=query,
                answer=answer,
                context=context,
                metadata=langsmith_metadata
            )
            
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
            
            # Log error to LangSmith
            langsmith_manager.log_question_answer(
                question=query,
                answer=error_response["answer"],
                context="",
                metadata={
                    "error": str(e),
                    "success": False,
                    "processing_time": total_time,
                    "similarity_threshold": similarity_threshold,
                    "k": k,
                    "use_llm": use_llm
                }
            )
            
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
    
    # Extract a simple identifier from URL (optional, for logging purposes)
    sig = "policy_document"  # Simple identifier for vector store (no namespace needed)
    
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
        global_vector_store = VectorStore(chunker.chunks, embed.embeddings, sig)
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
        
        # Log document processing to LangSmith
        langsmith_manager.log_document_processing(
            document_url=file_link,
            document_type="policy_pdf",
            chunks_created=len(chunker.chunks),
            embeddings_dimension=len(embeddings_data) if embeddings_data else 0,
            processing_time=total_time,
            metadata={
                "document_parsing_time": document_time,
                "chunking_time": chunking_time,
                "embedding_time": embedding_time,
                "vectorstore_time": vectorstore_time,
                "total_document_length": len(data),
                "vector_store_type": "DocArrayInMemorySearch",
                "distance_strategy": "cosine",
                "test_query_success": test_response.get("success", False),
                "system_initialization": "successful"
            }
        )

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
        
        # Log error to LangSmith
        langsmith_manager.log_document_processing(
            document_url=file_link,
            document_type="policy_pdf",
            chunks_created=0,
            embeddings_dimension=0,
            processing_time=total_time,
            metadata={
                "error": str(e),
                "success": False,
                "system_initialization": "failed",
                "error_type": type(e).__name__
            }
        )
        
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
    
    Expected request format:
    {
        "question": "Your question here",
        "k": 5 (optional, number of chunks to retrieve),
        "use_cosine_only": false (optional, use only cosine similarity without BM25),
        "similarity_threshold": 0.0 (optional, minimum cosine similarity threshold 0.0-1.0),
        "use_advanced": true (optional, use advanced retrieval and answer generation)
    }
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
    use_advanced = request.get("use_advanced", True)
    
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
def query_document_get(question: str, k: int = 5, use_advanced: bool = True):
    """
    Query the policy document with a user question (GET endpoint)
    
    Args:
        question (str): The question to ask
        k (int): Number of chunks to retrieve (default: 5)
        use_advanced (bool): Use advanced retrieval techniques (default: True)
    """
    global global_query_engine
    
    if not global_query_engine:
        return {
            "error": "RAG system not initialized. Please call the root endpoint (/) first.",
            "success": False
        }
    
    response = global_query_engine.answer_query(question, k=k, use_advanced=use_advanced)
    return response


@app.post("/query/advanced")
def query_document_advanced(request: dict):
    """
    Advanced query endpoint with enhanced retrieval and answer generation
    
    Expected request format:
    {
        "question": "Your question here",
        "k": 5 (optional, number of chunks to retrieve),
        "similarity_threshold": 0.3 (optional, minimum similarity threshold),
        "conversation_history": ["previous question 1", "previous question 2"] (optional),
        "explain_retrieval": false (optional, include retrieval explanation)
    }
    """
    global global_query_engine
    
    if not global_query_engine:
        return {
            "error": "RAG system not initialized. Please call the root endpoint (/) first.",
            "success": False
        }
    
    if not global_query_engine.advanced_retriever:
        return {
            "error": "Advanced retrieval features not available. Install required dependencies.",
            "success": False
        }
    
    question = request.get("question", "")
    k = request.get("k", 7)  # Slightly higher default for advanced retrieval
    similarity_threshold = request.get("similarity_threshold", 0.3)
    conversation_history = request.get("conversation_history", [])
    explain_retrieval = request.get("explain_retrieval", False)
    
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
    
    start_time = time.time()
    
    try:
        # Use context-aware retrieval if conversation history is provided
        if conversation_history:
            relevant_chunks_with_scores = global_query_engine.advanced_retriever.retrieve_with_context_awareness(
                question, k=k, conversation_history=conversation_history
            )
        else:
            relevant_chunks_with_scores = global_query_engine.advanced_retriever.retrieve_with_fusion(
                question, k=k, similarity_threshold=similarity_threshold
            )
        
        relevant_chunks = [doc for doc, score in relevant_chunks_with_scores]
        
        # Generate enhanced answer
        if global_query_engine.advanced_answer_generator:
            retrieval_confidence = sum(score for _, score in relevant_chunks_with_scores) / len(relevant_chunks_with_scores) if relevant_chunks_with_scores else 0.0
            answer_data = global_query_engine.advanced_answer_generator.generate_structured_answer(
                question, relevant_chunks, retrieval_confidence
            )
            answer = answer_data['answer']
            confidence = answer_data['confidence']
            intent = answer_data.get('intent', 'general')
        else:
            # Fallback
            context = global_query_engine.generate_context(relevant_chunks)
            answer = global_query_engine.simple_answer_extraction(question, context)
            confidence = 0.5
            intent = 'general'
        
        processing_time = time.time() - start_time
        
        response = {
            "query": question,
            "answer": answer,
            "confidence": confidence,
            "intent": intent,
            "num_chunks_retrieved": len(relevant_chunks),
            "success": True,
            "processing_time": processing_time,
            "retrieval_mode": "Advanced Fusion with Context Awareness" if conversation_history else "Advanced Fusion"
        }
        
        # Add retrieval explanation if requested
        if explain_retrieval:
            explanation = global_query_engine.advanced_retriever.get_explanation(question, relevant_chunks_with_scores)
            response["retrieval_explanation"] = explanation
        
        # Add chunk scores for transparency
        response["chunk_scores"] = [
            {"content": doc.page_content[:100] + "...", "score": score}
            for doc, score in relevant_chunks_with_scores[:3]
        ]
        
        # Log to LangSmith
        langsmith_manager.log_question_answer(
            question=question,
            answer=answer,
            context=global_query_engine.generate_context(relevant_chunks),
            metadata={
                "chunks_used": len(relevant_chunks),
                "retrieval_mode": "Advanced Fusion",
                "similarity_threshold": similarity_threshold,
                "confidence": confidence,
                "intent": intent,
                "processing_time": processing_time,
                "conversation_history_provided": bool(conversation_history),
                "advanced_features": True
            }
        )
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_response = {
            "query": question,
            "answer": "An error occurred during advanced processing.",
            "success": False,
            "error": str(e),
            "processing_time": processing_time
        }
        
        # Log error
        langsmith_manager.log_question_answer(
            question=question,
            answer=error_response["answer"],
            context="",
            metadata={
                "error": str(e),
                "success": False,
                "processing_time": processing_time,
                "advanced_features": True
            }
        )
        
        return error_response


@app.post("/hackrx/run")
async def run_rag(request: Request, body: QueryRequest) -> BatchQueryResponse:
    """
    Enhanced RAG endpoint for batch processing multiple questions with detailed timing
    
    This endpoint processes multiple questions against a document URL using:
    - PyMuPDF for fast and reliable PDF processing
    - Semantic chunking for intelligent text splitting
    - Parallel async processing for maximum LLM efficiency
    - Comprehensive timing instrumentation
    
    Args:
        request (Request): FastAPI request object
        body (QueryRequest): Request body containing document URL and questions
        
    Returns:
        BatchQueryResponse: Response containing answers to all questions
    """
    overall_start = time.time()
    
    try:
        # Optional: Token-based authentication (uncomment if needed)
        # token = request.headers.get("Authorization", "")
        # if token != "Bearer 2d42fd7d38f866414d839e960974157a2da00333865223973f728105760fe343":
        #     raise HTTPException(status_code=401, detail="Unauthorized")
        
        print(f"🚀 Processing batch RAG request with {len(body.questions)} questions")
        print(f"📄 Document URL: {body.documents[:50]}...")
        
        # Step 1: Download and extract PDF content using PyMuPDF
        step1_start = time.time()
        try:
            print("📥 Downloading and extracting PDF document with PyMuPDF...")
            document = PDFDocument(body.documents)
            pdf_text = document.parse_pdf()
            step1_time = time.time() - step1_start
            print(f"✅ PDF processed: {len(pdf_text)} characters in {step1_time:.3f}s")
            
            if not pdf_text.strip():
                raise HTTPException(status_code=400, detail="No text content found in PDF")
                
        except Exception as e:
            step1_time = time.time() - step1_start
            print(f"❌ PDF extraction failed in {step1_time:.3f}s: {e}")
            raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {e}")

        # Step 2: Semantic chunking for intelligent text splitting
        step2_start = time.time()
        try:
            print("✂️ Performing semantic chunking...")
            chunks = semantic_split(pdf_text, chunk_size=1000, chunk_overlap=100)
            step2_time = time.time() - step2_start
            print(f"✅ Semantic chunking complete: {len(chunks)} chunks created in {step2_time:.3f}s")
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No valid chunks created from document")
                
        except Exception as e:
            step2_time = time.time() - step2_start
            print(f"❌ Semantic chunking failed in {step2_time:.3f}s: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing document chunks: {e}")

        # Step 3: Prepare context for questions
        step3_start = time.time()
        full_context = " ".join(chunks)
        step3_time = time.time() - step3_start
        print(f"📝 Full context prepared: {len(full_context)} characters in {step3_time:.3f}s")

        # Step 4: Parallel LLM processing for maximum speed
        step4_start = time.time()
        try:
            print(f"🚀 Starting parallel LLM processing for {len(body.questions)} questions...")
            
            # Prepare all prompts at once
            prompt_prep_start = time.time()
            prompts = [
                f"Based on the following document, answer the question:\n\nDocument:\n{full_context}\n\nQuestion:\n{q}"
                for q in body.questions
            ]
            prompt_prep_time = time.time() - prompt_prep_start
            print(f"📋 Prepared {len(prompts)} prompts in {prompt_prep_time:.3f}s")
            
            # Create ALL async tasks simultaneously (no batching)
            task_creation_start = time.time()
            tasks = [process_chunk_with_llm_async(prompt) for prompt in prompts]
            task_creation_time = time.time() - task_creation_start
            print(f"🔧 Created {len(tasks)} async tasks in {task_creation_time:.3f}s")
            
            # Execute ALL tasks in parallel and wait for completion
            parallel_start = time.time()
            print(f"⚡ Executing all {len(tasks)} questions in parallel...")
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            parallel_time = time.time() - parallel_start
            
            # Handle any exceptions
            final_responses = []
            error_count = 0
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    error_msg = f"Error processing question {i+1}: {str(response)}"
                    print(f"❌ {error_msg}")
                    final_responses.append(error_msg)
                    error_count += 1
                else:
                    final_responses.append(response)
            
            step4_time = time.time() - step4_start
            success_count = len(final_responses) - error_count
            
            print(f"✅ Parallel processing complete in {parallel_time:.3f}s")
            print(f"📊 Results: {success_count} successful, {error_count} errors")
            print(f"⚡ Average time per question: {parallel_time/len(body.questions):.3f}s")
            print(f"🎯 Total LLM step time: {step4_time:.3f}s")
            
            responses = final_responses
            
        except Exception as e:
            step4_time = time.time() - step4_start
            print(f"❌ Parallel LLM processing failed in {step4_time:.3f}s: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing questions: {e}")

        # Step 5: Validate and return responses
        step5_start = time.time()
        if len(responses) != len(body.questions):
            print(f"⚠️ Response count mismatch: {len(responses)} responses for {len(body.questions)} questions")
            # Pad with error messages if needed
            while len(responses) < len(body.questions):
                responses.append("Error: Could not generate response for this question")

        step5_time = time.time() - step5_start
        overall_time = time.time() - overall_start
        
        print(f"🎉 Successfully processed {len(body.questions)} questions")
        print(f"⏱️ TIMING BREAKDOWN:")
        print(f"   📥 PyMuPDF PDF Parse: {step1_time:.3f}s ({step1_time/overall_time*100:.1f}%)")
        print(f"   🧠 Semantic Chunking: {step2_time:.3f}s ({step2_time/overall_time*100:.1f}%)")
        print(f"   📝 Context Prep: {step3_time:.3f}s ({step3_time/overall_time*100:.1f}%)")
        print(f"   ⚡ Parallel LLM Processing: {step4_time:.3f}s ({step4_time/overall_time*100:.1f}%)")
        print(f"   ✅ Response Validation: {step5_time:.3f}s ({step5_time/overall_time*100:.1f}%)")
        print(f"   🔄 TOTAL TIME: {overall_time:.3f}s")
        print(f"🎯 PERFORMANCE METRICS:")
        print(f"   ⚡ Questions per second: {len(body.questions)/overall_time:.2f}")
        print(f"   📊 Avg time per question: {overall_time/len(body.questions):.3f}s")
        if step4_time > 0:
            print(f"   🚀 LLM efficiency: {len(body.questions)/step4_time:.2f} q/s")
        
        # Log document processing to LangSmith
        langsmith_manager.log_document_processing(
            document_url=body.documents,
            document_type="batch_pdf",
            chunks_created=len(chunks),
            embeddings_dimension=0,  # No embeddings in batch processing
            processing_time=step1_time + step2_time,
            metadata={
                "pdf_parsing_time": step1_time,
                "semantic_chunking_time": step2_time,
                "context_prep_time": step3_time,
                "document_length": len(pdf_text),
                "chunk_count": len(chunks),
                "processing_mode": "batch",
                "chunking_strategy": "semantic",
                "chunk_size": 1000,
                "chunk_overlap": 100
            }
        )
        
        # Log batch Q&A to LangSmith
        langsmith_manager.log_batch_questions_answers(
            questions=body.questions,
            answers=responses,
            document_context=full_context,
            metadata={
                "total_processing_time": overall_time,
                "pdf_parsing_time": step1_time,
                "semantic_chunking_time": step2_time,
                "context_prep_time": step3_time,
                "llm_processing_time": step4_time,
                "response_validation_time": step5_time,
                "questions_per_second": len(body.questions)/overall_time,
                "avg_time_per_question": overall_time/len(body.questions),
                "llm_efficiency": len(body.questions)/step4_time if step4_time > 0 else 0,
                "success_count": success_count,
                "error_count": error_count,
                "processing_mode": "parallel_batch",
                "chunk_count": len(chunks),
                "document_url": body.documents
            }
        )
        
        return BatchQueryResponse(answers=responses)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        overall_time = time.time() - overall_start
        print(f"❌ Unexpected error in run_rag after {overall_time:.3f}s: {e}")
        
        # Log error to LangSmith
        langsmith_manager.log_batch_questions_answers(
            questions=body.questions if 'body' in locals() else [],
            answers=[f"Error: {str(e)}"] * len(body.questions) if 'body' in locals() else ["Error: System failure"],
            document_context="",
            metadata={
                "error": str(e),
                "success": False,
                "processing_time": overall_time,
                "error_type": type(e).__name__,
                "processing_mode": "batch_error",
                "document_url": body.documents if 'body' in locals() else "unknown"
            }
        )
        
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, port=8000)