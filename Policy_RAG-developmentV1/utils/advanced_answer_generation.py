"""
Advanced answer generation and context processing for maximum accuracy
"""

import re
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class AdvancedAnswerGenerator:
    """
    Advanced answer generation system that:
    1. Analyzes query intent and type
    2. Processes context intelligently
    3. Generates structured, accurate answers
    4. Provides confidence scoring
    """
    
    def __init__(self):
        self.query_patterns = self._init_query_patterns()
        self.answer_templates = self._init_answer_templates()
    
    def _init_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for different query types"""
        return {
            'definition': [
                r'what is\s+(.+)',
                r'define\s+(.+)',
                r'(.+)\s+means?',
                r'meaning of\s+(.+)',
                r'explain\s+(.+)'
            ],
            'procedure': [
                r'how to\s+(.+)',
                r'how do i\s+(.+)',
                r'steps to\s+(.+)',
                r'process for\s+(.+)',
                r'procedure to\s+(.+)'
            ],
            'eligibility': [
                r'who is eligible\s+(.+)',
                r'eligibility for\s+(.+)',
                r'can i\s+(.+)',
                r'am i eligible\s+(.+)',
                r'qualifying for\s+(.+)'
            ],
            'amount': [
                r'how much\s+(.+)',
                r'what is the cost\s+(.+)',
                r'price of\s+(.+)',
                r'amount for\s+(.+)',
                r'premium for\s+(.+)'
            ],
            'time': [
                r'when\s+(.+)',
                r'how long\s+(.+)',
                r'duration of\s+(.+)',
                r'time period\s+(.+)',
                r'grace period\s+(.+)'
            ],
            'coverage': [
                r'what is covered\s+(.+)',
                r'coverage for\s+(.+)',
                r'benefits of\s+(.+)',
                r'what does\s+(.+)\s+cover',
                r'included in\s+(.+)'
            ],
            'exclusion': [
                r'what is not covered\s+(.+)',
                r'exclusions\s+(.+)',
                r'limitations of\s+(.+)',
                r'not included\s+(.+)',
                r'excluded from\s+(.+)'
            ],
            'comparison': [
                r'difference between\s+(.+)',
                r'compare\s+(.+)',
                r'(.+)\s+vs\s+(.+)',
                r'which is better\s+(.+)',
                r'advantages of\s+(.+)'
            ]
        }
    
    def _init_answer_templates(self) -> Dict[str, str]:
        """Initialize answer templates for different query types"""
        return {
            'definition': "Based on the policy document, {term} is defined as: {definition}.\n\nKey details:\n{details}",
            'procedure': "To {action}, follow these steps according to the policy:\n\n{steps}\n\nImportant notes:\n{notes}",
            'eligibility': "Eligibility for {subject}:\n\n{criteria}\n\nAdditional requirements:\n{requirements}",
            'amount': "The cost/amount for {subject} is:\n\n{amount}\n\nAdditional details:\n{details}",
            'time': "Regarding the timing for {subject}:\n\n{timeframe}\n\nImportant deadlines:\n{deadlines}",
            'coverage': "{subject} coverage includes:\n\n{benefits}\n\nConditions and limitations:\n{conditions}",
            'exclusion': "{subject} exclusions and limitations:\n\n{exclusions}\n\nAlternative options:\n{alternatives}",
            'comparison': "Comparison of {subjects}:\n\n{comparison_table}\n\nRecommendation:\n{recommendation}",
            'general': "Based on the policy document:\n\n{answer}\n\nRelevant details:\n{supporting_info}"
        }
    
    def analyze_query_intent(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Analyze the query to determine intent and extract key terms
        """
        query_lower = query.lower().strip()
        
        for intent, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    key_term = match.group(1) if match.groups() else None
                    logger.info(f"🎯 Query intent: {intent} | Key term: {key_term}")
                    return intent, key_term
        
        logger.info("🎯 Query intent: general")
        return 'general', None
    
    def extract_key_information(self, documents: List[Document], intent: str, key_term: str = None) -> Dict[str, str]:
        """
        Extract key information from documents based on query intent
        """
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        
        if intent == 'definition':
            return self._extract_definition(combined_text, key_term)
        elif intent == 'procedure':
            return self._extract_procedure(combined_text, key_term)
        elif intent == 'eligibility':
            return self._extract_eligibility(combined_text, key_term)
        elif intent == 'amount':
            return self._extract_amount(combined_text, key_term)
        elif intent == 'time':
            return self._extract_time(combined_text, key_term)
        elif intent == 'coverage':
            return self._extract_coverage(combined_text, key_term)
        elif intent == 'exclusion':
            return self._extract_exclusions(combined_text, key_term)
        else:
            return self._extract_general(combined_text, key_term)
    
    def _extract_definition(self, text: str, term: str) -> Dict[str, str]:
        """Extract definition-related information"""
        if not term:
            return {'definition': 'No specific term identified', 'details': ''}
        
        # Look for explicit definitions
        definition_patterns = [
            rf'{re.escape(term)}\s+means?\s+([^.]+)',
            rf'{re.escape(term)}\s+is\s+defined\s+as\s+([^.]+)',
            rf'{re.escape(term)}[:\-]\s*([^.]+)',
            rf'definition\s+of\s+{re.escape(term)}[:\-]\s*([^.]+)'
        ]
        
        definition = ""
        for pattern in definition_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                definition = match.group(1).strip()
                break
        
        # Extract additional details
        details = self._extract_related_sentences(text, term, max_sentences=3)
        
        return {
            'term': term,
            'definition': definition or f"Definition for '{term}' not explicitly found in the document",
            'details': details
        }
    
    def _extract_procedure(self, text: str, action: str) -> Dict[str, str]:
        """Extract procedure-related information"""
        # Look for step-by-step procedures
        step_patterns = [
            r'(?:step\s+\d+[:\-.]?\s*)([^.]+)',
            r'(?:\d+\.\s*)([^.]+)',
            r'(?:first|second|third|next|then|finally)[:\-,]\s*([^.]+)'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            steps.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        # Extract notes and important information
        note_patterns = [
            r'(?:note|important|warning|caution)[:\-]\s*([^.]+)',
            r'(?:please note|remember|ensure)[:\-]\s*([^.]+)'
        ]
        
        notes = []
        for pattern in note_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            notes.extend([match.strip() for match in matches])
        
        return {
            'action': action or 'the requested action',
            'steps': '\n'.join([f"• {step}" for step in steps[:5]]) or "Specific steps not clearly outlined in the document",
            'notes': '\n'.join([f"• {note}" for note in notes[:3]]) or "No additional notes found"
        }
    
    def _extract_eligibility(self, text: str, subject: str) -> Dict[str, str]:
        """Extract eligibility-related information"""
        eligibility_patterns = [
            r'(?:eligible|eligibility)[^.]*?([^.]+)',
            r'(?:qualify|qualification)[^.]*?([^.]+)',
            r'(?:criteria|requirements)[^.]*?([^.]+)'
        ]
        
        criteria = []
        for pattern in eligibility_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            criteria.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        # Extract age, income, or other specific requirements
        requirement_patterns = [
            r'(?:age|minimum age|maximum age)[^.]*?(\d+[^.]*)',
            r'(?:income|salary|annual income)[^.]*?([^.]+)',
            r'(?:must|should|required to)[^.]*?([^.]+)'
        ]
        
        requirements = []
        for pattern in requirement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            requirements.extend([match.strip() for match in matches])
        
        return {
            'subject': subject or 'this benefit',
            'criteria': '\n'.join([f"• {criterion}" for criterion in criteria[:5]]) or "Specific eligibility criteria not clearly stated",
            'requirements': '\n'.join([f"• {req}" for req in requirements[:3]]) or "No additional requirements specified"
        }
    
    def _extract_amount(self, text: str, subject: str) -> Dict[str, str]:
        """Extract amount/cost-related information"""
        # Look for currency amounts, percentages, and numbers
        amount_patterns = [
            r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?',
            r'\$\s*[\d,]+(?:\.\d+)?',
            r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:rupees?|dollars?)',
            r'\d+(?:\.\d+)?%',
            r'\d+\s*(?:times|x)\s*(?:annual|monthly|daily)'
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)
        
        # Extract context around amounts
        details = []
        for amount in amounts[:3]:
            # Find sentences containing this amount
            sentences = re.findall(rf'[^.]*{re.escape(amount)}[^.]*\.', text, re.IGNORECASE)
            details.extend(sentences[:2])
        
        return {
            'subject': subject or 'this item',
            'amount': '\n'.join([f"• {amount}" for amount in amounts[:5]]) or "Specific amounts not clearly stated in the document",
            'details': '\n'.join([f"• {detail.strip()}" for detail in details[:3]]) or "No additional cost details provided"
        }
    
    def _extract_time(self, text: str, subject: str) -> Dict[str, str]:
        """Extract time-related information"""
        time_patterns = [
            r'\d+\s*(?:days?|weeks?|months?|years?)',
            r'(?:within|after|before)\s+\d+\s*(?:days?|weeks?|months?|years?)',
            r'(?:immediately|instantly|upon)',
            r'(?:annual|monthly|quarterly|daily)',
            r'grace period[^.]*'
        ]
        
        timeframes = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            timeframes.extend([match.strip() for match in matches])
        
        # Extract deadline-related information
        deadline_patterns = [
            r'(?:deadline|due date|expiry)[^.]*',
            r'(?:must be|should be)[^.]*(?:within|by|before)[^.]*',
            r'(?:last date|final date)[^.]*'
        ]
        
        deadlines = []
        for pattern in deadline_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            deadlines.extend([match.strip() for match in matches])
        
        return {
            'subject': subject or 'this process',
            'timeframe': '\n'.join([f"• {time}" for time in timeframes[:5]]) or "Specific timeframes not clearly mentioned",
            'deadlines': '\n'.join([f"• {deadline}" for deadline in deadlines[:3]]) or "No specific deadlines mentioned"
        }
    
    def _extract_coverage(self, text: str, subject: str) -> Dict[str, str]:
        """Extract coverage-related information"""
        coverage_patterns = [
            r'(?:covers?|covered|coverage|includes?|benefits?)[^.]*',
            r'(?:entitled to|eligible for)[^.]*',
            r'(?:reimbursement|payment|compensation)[^.]*'
        ]
        
        benefits = []
        for pattern in coverage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            benefits.extend([match.strip() for match in matches if len(match.strip()) > 15])
        
        # Extract conditions and limitations
        condition_patterns = [
            r'(?:provided|subject to|conditional on)[^.]*',
            r'(?:limitations?|restrictions?|conditions?)[^.]*',
            r'(?:maximum|minimum|up to)[^.]*'
        ]
        
        conditions = []
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditions.extend([match.strip() for match in matches])
        
        return {
            'subject': subject or 'this policy',
            'benefits': '\n'.join([f"• {benefit}" for benefit in benefits[:5]]) or "Specific coverage details not clearly outlined",
            'conditions': '\n'.join([f"• {condition}" for condition in conditions[:3]]) or "No specific conditions mentioned"
        }
    
    def _extract_exclusions(self, text: str, subject: str) -> Dict[str, str]:
        """Extract exclusion-related information"""
        exclusion_patterns = [
            r'(?:exclusions?|excluded|not covered|limitations?)[^.]*',
            r'(?:does not cover|will not pay|not applicable)[^.]*',
            r'(?:except|excluding|other than)[^.]*'
        ]
        
        exclusions = []
        for pattern in exclusion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            exclusions.extend([match.strip() for match in matches if len(match.strip()) > 15])
        
        # Look for alternative options
        alternative_patterns = [
            r'(?:alternatively|instead|option)[^.]*',
            r'(?:may consider|can opt for)[^.]*'
        ]
        
        alternatives = []
        for pattern in alternative_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            alternatives.extend([match.strip() for match in matches])
        
        return {
            'subject': subject or 'this coverage',
            'exclusions': '\n'.join([f"• {exclusion}" for exclusion in exclusions[:5]]) or "Specific exclusions not clearly listed",
            'alternatives': '\n'.join([f"• {alt}" for alt in alternatives[:3]]) or "No alternative options mentioned"
        }
    
    def _extract_general(self, text: str, key_term: str) -> Dict[str, str]:
        """Extract general information"""
        if key_term:
            relevant_sentences = self._extract_related_sentences(text, key_term, max_sentences=5)
        else:
            # Extract the most informative sentences
            sentences = re.split(r'[.!?]+', text)
            relevant_sentences = '\n'.join([f"• {s.strip()}" for s in sentences[:5] if len(s.strip()) > 20])
        
        return {
            'answer': relevant_sentences or "Relevant information found in the document",
            'supporting_info': "Please refer to the complete policy document for comprehensive details"
        }
    
    def _extract_related_sentences(self, text: str, term: str, max_sentences: int = 3) -> str:
        """Extract sentences related to a specific term"""
        if not term:
            return ""
        
        sentences = re.split(r'[.!?]+', text)
        related_sentences = []
        
        for sentence in sentences:
            if term.lower() in sentence.lower() and len(sentence.strip()) > 20:
                related_sentences.append(sentence.strip())
                if len(related_sentences) >= max_sentences:
                    break
        
        return '\n'.join([f"• {sentence}" for sentence in related_sentences])
    
    def generate_structured_answer(self, query: str, documents: List[Document], 
                                 confidence_score: float = 0.0) -> Dict[str, str]:
        """
        Generate a structured answer based on query intent and retrieved documents
        """
        if not documents:
            return {
                'answer': "I couldn't find relevant information in the policy document to answer your question.",
                'confidence': 0.0,
                'sources': 0,
                'explanation': "No relevant documents were retrieved."
            }
        
        # Analyze query intent
        intent, key_term = self.analyze_query_intent(query)
        
        # Extract key information
        extracted_info = self.extract_key_information(documents, intent, key_term)
        
        # Generate structured answer using template
        template = self.answer_templates.get(intent, self.answer_templates['general'])
        
        try:
            answer = template.format(**extracted_info)
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}")
            answer = f"Based on the policy document: {extracted_info.get('answer', 'Information extracted from the retrieved sections.')}"
        
        # Add source information
        answer += f"\n\n**Sources:** Information compiled from {len(documents)} relevant sections of the policy document."
        
        # Calculate answer confidence
        answer_confidence = self._calculate_answer_confidence(query, documents, extracted_info, confidence_score)
        
        return {
            'answer': answer,
            'confidence': answer_confidence,
            'sources': len(documents),
            'intent': intent,
            'key_term': key_term,
            'explanation': f"Answer generated using {intent} analysis pattern with {answer_confidence:.1%} confidence."
        }
    
    def _calculate_answer_confidence(self, query: str, documents: List[Document], 
                                   extracted_info: Dict[str, str], retrieval_confidence: float) -> float:
        """
        Calculate confidence score for the generated answer
        """
        # Base confidence from retrieval
        confidence = retrieval_confidence * 0.4
        
        # Content quality indicators
        total_content_length = sum(len(doc.page_content) for doc in documents)
        if total_content_length > 500:
            confidence += 0.2
        
        # Information extraction quality
        info_quality = 0.0
        for key, value in extracted_info.items():
            if value and len(value) > 20 and "not found" not in value.lower():
                info_quality += 0.1
        
        confidence += min(info_quality, 0.3)
        
        # Query-answer alignment
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        answer_words = set(re.findall(r'\b\w+\b', str(extracted_info).lower()))
        overlap = len(query_words.intersection(answer_words))
        alignment_score = min(overlap / max(len(query_words), 1), 0.1)
        confidence += alignment_score
        
        return min(confidence, 1.0)


def get_advanced_answer_generator() -> AdvancedAnswerGenerator:
    """Factory function to create an advanced answer generator"""
    return AdvancedAnswerGenerator()
