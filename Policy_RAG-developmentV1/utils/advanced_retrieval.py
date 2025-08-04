"""
Advanced retrieval techniques for maximum accuracy and relevance
Implements multiple state-of-the-art retrieval strategies
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.retrievers import BM25Retriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from collections import defaultdict, Counter
import math

logger = logging.getLogger(__name__)

class AdvancedRetriever:
    """
    Advanced retrieval system combining multiple techniques for maximum accuracy:
    1. Hybrid retrieval (BM25 + Vector similarity)
    2. Query expansion and reformulation
    3. Contextual re-ranking
    4. Semantic clustering
    5. Confidence scoring
    """
    
    def __init__(self, vector_store: DocArrayInMemorySearch, documents: List[Document], 
                 embeddings_model, chunk_size: int = 1000):
        self.vector_store = vector_store
        self.documents = documents
        self.embeddings_model = embeddings_model
        self.chunk_size = chunk_size
        
        # Initialize retrievers
        self._init_retrievers()
        
        # Build additional indexes
        self._build_keyword_index()
        self._build_tfidf_index()
        self._analyze_document_structure()
        
        logger.info(f"🚀 AdvancedRetriever initialized with {len(documents)} documents")
    
    def _init_retrievers(self):
        """Initialize various retriever components"""
        try:
            # BM25 retriever for sparse retrieval
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 10  # Get more candidates for re-ranking
            
            # Vector retriever for dense retrieval
            self.vector_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 10, "score_threshold": 0.1}
            )
            
            logger.info("✅ Base retrievers initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing retrievers: {e}")
            raise
    
    def _build_keyword_index(self):
        """Build keyword-based index for exact matching"""
        self.keyword_index = defaultdict(list)
        
        for i, doc in enumerate(self.documents):
            content = doc.page_content.lower()
            
            # Extract important keywords and phrases
            # Policy-specific terms
            policy_terms = re.findall(r'\b(?:policy|premium|coverage|benefit|claim|deductible|exclusion)\b', content)
            
            # Numbers and amounts
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', content)
            
            # Dates
            dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', content)
            
            # Percentages
            percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', content)
            
            # Store all terms
            for term in policy_terms + numbers + dates + percentages:
                self.keyword_index[term].append(i)
        
        logger.info(f"📚 Keyword index built with {len(self.keyword_index)} terms")
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for statistical relevance"""
        try:
            doc_texts = [doc.page_content for doc in self.documents]
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
                max_df=0.95,
                min_df=2
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
            logger.info(f"📊 TF-IDF index built with {self.tfidf_matrix.shape[1]} features")
        except Exception as e:
            logger.error(f"❌ Error building TF-IDF index: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def _analyze_document_structure(self):
        """Analyze document structure for section-aware retrieval"""
        self.section_map = {}
        self.heading_patterns = [
            r'^[A-Z][A-Z\s]+:',  # ALL CAPS headings
            r'^\d+\.\s+[A-Z]',    # Numbered sections
            r'^[IVX]+\.\s+',      # Roman numerals
            r'^[A-Z]\.\s+',       # Letter sections
        ]
        
        for i, doc in enumerate(self.documents):
            content = doc.page_content
            
            # Detect section headings
            for pattern in self.heading_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    self.section_map[i] = matches[0]
                    break
            
            # Check if document contains definitions
            if 'definition' in content.lower() or 'means' in content.lower():
                self.section_map[i] = self.section_map.get(i, '') + ' [DEFINITIONS]'
            
            # Check if document contains procedures
            if any(word in content.lower() for word in ['procedure', 'process', 'step', 'how to']):
                self.section_map[i] = self.section_map.get(i, '') + ' [PROCEDURES]'
        
        logger.info(f"📋 Document structure analyzed: {len(self.section_map)} sections identified")
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms for better retrieval
        """
        expanded_queries = [query]
        
        # Policy-specific expansions
        expansions = {
            'premium': ['payment', 'cost', 'fee', 'amount due'],
            'coverage': ['benefit', 'protection', 'insurance'],
            'claim': ['reimbursement', 'payout', 'settlement'],
            'deductible': ['excess', 'out-of-pocket'],
            'exclusion': ['limitation', 'restriction', 'not covered'],
            'grace period': ['payment period', 'late payment'],
            'policyholder': ['insured', 'member', 'subscriber'],
            'network': ['provider', 'hospital', 'doctor'],
            'pre-existing': ['existing condition', 'prior condition'],
            'renewal': ['continuation', 'extension'],
        }
        
        query_lower = query.lower()
        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        # Add question variations
        if '?' not in query:
            expanded_queries.append(f"What is {query}?")
            expanded_queries.append(f"How does {query} work?")
            expanded_queries.append(f"When is {query} applicable?")
        
        logger.info(f"🔍 Query expanded to {len(expanded_queries)} variations")
        return expanded_queries[:5]  # Limit to top 5 expansions
    
    def _tfidf_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search using TF-IDF similarity"""
        if not self.tfidf_vectorizer or not self.tfidf_matrix:
            return []
        
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            results = []
            
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append((self.documents[idx], float(similarities[idx])))
            
            return results
        except Exception as e:
            logger.error(f"❌ TF-IDF search error: {e}")
            return []
    
    def _keyword_search(self, query: str) -> List[int]:
        """Search for exact keyword matches"""
        query_lower = query.lower()
        matched_docs = set()
        
        # Extract potential keywords from query
        keywords = re.findall(r'\b\w+\b', query_lower)
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', query)
        percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', query)
        
        all_terms = keywords + numbers + percentages
        
        for term in all_terms:
            if term in self.keyword_index:
                matched_docs.update(self.keyword_index[term])
        
        return list(matched_docs)
    
    def _semantic_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Enhanced semantic search with multiple query variations"""
        all_results = []
        
        # Search with original query
        try:
            results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            for doc, score in results_with_scores:
                # Convert distance to similarity score
                similarity = 1 - score if score <= 1 else 1 / (1 + score)
                all_results.append((doc, similarity))
        except Exception as e:
            logger.error(f"❌ Semantic search error: {e}")
        
        return all_results
    
    def _bm25_search(self, query: str, k: int = 5) -> List[Document]:
        """BM25 search for sparse retrieval"""
        try:
            self.bm25_retriever.k = k
            return self.bm25_retriever.get_relevant_documents(query)
        except Exception as e:
            logger.error(f"❌ BM25 search error: {e}")
            return []
    
    def _calculate_relevance_score(self, doc: Document, query: str, 
                                 semantic_score: float = 0.0, 
                                 tfidf_score: float = 0.0, 
                                 keyword_match: bool = False,
                                 bm25_rank: int = 0) -> float:
        """
        Calculate comprehensive relevance score using multiple signals
        """
        # Base scores
        semantic_weight = 0.4
        tfidf_weight = 0.3
        keyword_weight = 0.2
        bm25_weight = 0.1
        
        # Calculate weighted score
        score = (semantic_score * semantic_weight + 
                tfidf_score * tfidf_weight + 
                (1.0 if keyword_match else 0.0) * keyword_weight +
                (1.0 / (bm25_rank + 1)) * bm25_weight)
        
        # Boost scores for certain document types
        content = doc.page_content.lower()
        
        # Boost for definitions if query asks "what is"
        if "what is" in query.lower() and "definition" in content:
            score *= 1.3
        
        # Boost for procedures if query asks "how to"
        if "how" in query.lower() and any(word in content for word in ["step", "process", "procedure"]):
            score *= 1.2
        
        # Boost for exact phrase matches
        query_phrases = re.findall(r'"([^"]*)"', query)
        for phrase in query_phrases:
            if phrase.lower() in content:
                score *= 1.5
        
        # Penalize very short or very long chunks
        content_length = len(doc.page_content)
        if content_length < 100:
            score *= 0.8
        elif content_length > 3000:
            score *= 0.9
        
        return min(score, 1.0)  # Cap at 1.0
    
    def retrieve_with_fusion(self, query: str, k: int = 5, 
                           similarity_threshold: float = 0.3) -> List[Tuple[Document, float]]:
        """
        Advanced retrieval using fusion of multiple techniques
        """
        logger.info(f"🔍 Advanced fusion retrieval for: {query[:100]}...")
        
        # Step 1: Expand query
        expanded_queries = self.expand_query(query)
        
        # Step 2: Multi-modal retrieval
        all_candidates = {}  # doc_id -> (doc, scores_dict)
        
        # Semantic search with expanded queries
        for exp_query in expanded_queries:
            semantic_results = self._semantic_search(exp_query, k=k*2)
            for doc, score in semantic_results:
                doc_id = id(doc)
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = (doc, {'semantic': 0, 'tfidf': 0, 'keyword': False, 'bm25_rank': 999})
                all_candidates[doc_id][1]['semantic'] = max(all_candidates[doc_id][1]['semantic'], score)
        
        # TF-IDF search
        tfidf_results = self._tfidf_search(query, k=k*2)
        for doc, score in tfidf_results:
            doc_id = id(doc)
            if doc_id not in all_candidates:
                all_candidates[doc_id] = (doc, {'semantic': 0, 'tfidf': 0, 'keyword': False, 'bm25_rank': 999})
            all_candidates[doc_id][1]['tfidf'] = score
        
        # Keyword search
        keyword_matches = self._keyword_search(query)
        for idx in keyword_matches:
            if idx < len(self.documents):
                doc = self.documents[idx]
                doc_id = id(doc)
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = (doc, {'semantic': 0, 'tfidf': 0, 'keyword': False, 'bm25_rank': 999})
                all_candidates[doc_id][1]['keyword'] = True
        
        # BM25 search
        bm25_results = self._bm25_search(query, k=k*2)
        for rank, doc in enumerate(bm25_results):
            doc_id = id(doc)
            if doc_id not in all_candidates:
                all_candidates[doc_id] = (doc, {'semantic': 0, 'tfidf': 0, 'keyword': False, 'bm25_rank': 999})
            all_candidates[doc_id][1]['bm25_rank'] = rank
        
        # Step 3: Calculate fusion scores
        scored_results = []
        for doc_id, (doc, scores) in all_candidates.items():
            fusion_score = self._calculate_relevance_score(
                doc, query,
                semantic_score=scores['semantic'],
                tfidf_score=scores['tfidf'],
                keyword_match=scores['keyword'],
                bm25_rank=scores['bm25_rank']
            )
            
            if fusion_score >= similarity_threshold:
                scored_results.append((doc, fusion_score))
        
        # Step 4: Sort by fusion score and return top k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        final_results = scored_results[:k]
        
        logger.info(f"✅ Fusion retrieval: {len(final_results)} results above threshold {similarity_threshold}")
        for i, (doc, score) in enumerate(final_results[:3]):
            content_preview = doc.page_content[:100].replace('\n', ' ')
            logger.info(f"   {i+1}. Score: {score:.3f} | {content_preview}...")
        
        return final_results
    
    def retrieve_with_context_awareness(self, query: str, k: int = 5,
                                      conversation_history: List[str] = None) -> List[Tuple[Document, float]]:
        """
        Context-aware retrieval considering conversation history
        """
        # If we have conversation history, modify query
        if conversation_history:
            # Extract key terms from recent conversation
            recent_context = ' '.join(conversation_history[-3:])  # Last 3 interactions
            context_keywords = set(re.findall(r'\b\w+\b', recent_context.lower()))
            
            # Boost query with context
            query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
            shared_keywords = context_keywords.intersection(query_keywords)
            
            if shared_keywords:
                enhanced_query = query + " " + " ".join(shared_keywords)
                logger.info(f"🧠 Context-enhanced query: {enhanced_query}")
                return self.retrieve_with_fusion(enhanced_query, k=k)
        
        return self.retrieve_with_fusion(query, k=k)
    
    def get_explanation(self, query: str, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """
        Generate explanation of why these documents were retrieved
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        explanations = []
        for i, (doc, score) in enumerate(retrieved_docs[:3]):
            content_preview = doc.page_content[:150].replace('\n', ' ')
            explanations.append(f"Result {i+1} (relevance: {score:.1%}): {content_preview}...")
        
        return "Retrieved based on semantic similarity, keyword matching, and document structure analysis:\n" + "\n".join(explanations)


def get_advanced_retriever(vector_store: DocArrayInMemorySearch, documents: List[Document], 
                          embeddings_model, chunk_size: int = 1000) -> AdvancedRetriever:
    """
    Factory function to create an advanced retriever
    """
    return AdvancedRetriever(vector_store, documents, embeddings_model, chunk_size)
