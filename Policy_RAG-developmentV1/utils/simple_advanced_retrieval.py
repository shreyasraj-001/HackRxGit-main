"""
Simplified advanced retrieval with basic dependencies only
Provides enhanced retrieval without requiring scikit-learn or numpy
"""

import re
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.retrievers import BM25Retriever
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class SimpleAdvancedRetriever:
    """
    Simplified advanced retriever using only basic dependencies
    Focuses on query enhancement and intelligent chunking
    """
    
    def __init__(self, vector_store: DocArrayInMemorySearch, documents: List[Document], embeddings_model):
        self.vector_store = vector_store
        self.documents = documents
        self.embeddings_model = embeddings_model
        
        # Initialize basic retrievers
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Build keyword index
        self._build_keyword_index()
        
        logger.info(f"✅ SimpleAdvancedRetriever initialized with {len(documents)} documents")
    
    def _build_keyword_index(self):
        """Build simple keyword index for exact matching"""
        self.keyword_index = defaultdict(list)
        
        for i, doc in enumerate(self.documents):
            content = doc.page_content.lower()
            
            # Policy-specific terms
            policy_terms = re.findall(r'\b(?:policy|premium|coverage|benefit|claim|deductible|exclusion|grace period)\b', content)
            
            # Numbers and amounts
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', content)
            
            # Store terms
            for term in policy_terms + numbers:
                self.keyword_index[term].append(i)
    
    def expand_query(self, query: str) -> List[str]:
        """Simple query expansion with policy-specific synonyms"""
        expanded_queries = [query]
        
        expansions = {
            'premium': ['payment', 'cost', 'fee'],
            'coverage': ['benefit', 'protection'],
            'claim': ['reimbursement', 'payout'],
            'grace period': ['payment period', 'late payment'],
            'policyholder': ['insured', 'member'],
        }
        
        query_lower = query.lower()
        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        return expanded_queries[:3]  # Limit to top 3
    
    def retrieve_with_fusion(self, query: str, k: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[Document, float]]:
        """
        Simple fusion retrieval combining multiple methods
        """
        logger.info(f"🔍 Simple fusion retrieval for: {query[:100]}...")
        
        all_candidates = {}  # doc_id -> (doc, score)
        
        # Semantic search
        try:
            semantic_results = self.vector_store.similarity_search_with_score(query, k=k*2)
            for doc, distance in semantic_results:
                # Convert distance to similarity
                similarity = 1 - distance if distance <= 1 else 1 / (1 + distance)
                doc_id = id(doc)
                all_candidates[doc_id] = (doc, similarity * 0.6)  # Weight: 60%
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
        
        # BM25 search
        try:
            self.bm25_retriever.k = k*2
            bm25_results = self.bm25_retriever.get_relevant_documents(query)
            for rank, doc in enumerate(bm25_results):
                doc_id = id(doc)
                bm25_score = 1.0 / (rank + 1) * 0.3  # Weight: 30%
                if doc_id in all_candidates:
                    all_candidates[doc_id] = (all_candidates[doc_id][0], all_candidates[doc_id][1] + bm25_score)
                else:
                    all_candidates[doc_id] = (doc, bm25_score)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
        
        # Keyword matching
        query_terms = re.findall(r'\b\w+\b', query.lower())
        for term in query_terms:
            if term in self.keyword_index:
                for doc_idx in self.keyword_index[term]:
                    if doc_idx < len(self.documents):
                        doc = self.documents[doc_idx]
                        doc_id = id(doc)
                        keyword_score = 0.1  # Weight: 10%
                        if doc_id in all_candidates:
                            all_candidates[doc_id] = (all_candidates[doc_id][0], all_candidates[doc_id][1] + keyword_score)
                        else:
                            all_candidates[doc_id] = (doc, keyword_score)
        
        # Filter and sort results
        scored_results = [(doc, score) for doc, score in all_candidates.values() if score >= similarity_threshold]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        final_results = scored_results[:k]
        logger.info(f"✅ Simple fusion: {len(final_results)} results above threshold {similarity_threshold}")
        
        return final_results
    
    def retrieve_with_context_awareness(self, query: str, k: int = 5, conversation_history: List[str] = None) -> List[Tuple[Document, float]]:
        """Simple context-aware retrieval"""
        if conversation_history:
            # Extract keywords from recent context
            recent_context = ' '.join(conversation_history[-2:])
            context_keywords = set(re.findall(r'\b\w+\b', recent_context.lower()))
            query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
            shared_keywords = context_keywords.intersection(query_keywords)
            
            if shared_keywords:
                enhanced_query = query + " " + " ".join(shared_keywords)
                return self.retrieve_with_fusion(enhanced_query, k=k)
        
        return self.retrieve_with_fusion(query, k=k)
    
    def get_explanation(self, query: str, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """Simple explanation of retrieval"""
        if not retrieved_docs:
            return "No relevant documents found."
        
        explanations = []
        for i, (doc, score) in enumerate(retrieved_docs[:3]):
            content_preview = doc.page_content[:100].replace('\n', ' ')
            explanations.append(f"Result {i+1} (score: {score:.2f}): {content_preview}...")
        
        return "Retrieved using semantic similarity and keyword matching:\n" + "\n".join(explanations)


def get_advanced_retriever(vector_store: DocArrayInMemorySearch, documents: List[Document], 
                          embeddings_model, chunk_size: int = 1000):
    """
    Factory function that tries advanced retriever first, falls back to simple version
    """
    try:
        # Try to import the full advanced retriever
        from utils.advanced_retrieval import AdvancedRetriever
        return AdvancedRetriever(vector_store, documents, embeddings_model, chunk_size)
    except ImportError:
        # Fall back to simple version
        logger.info("📎 Using simplified advanced retriever (some dependencies not available)")
        return SimpleAdvancedRetriever(vector_store, documents, embeddings_model)
