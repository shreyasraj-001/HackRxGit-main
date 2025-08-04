import csv
import asyncio
import aiofiles
import os
from datetime import datetime
from typing import List, Dict, Any
import json

class AsyncCSVLogger:
    """
    Async CSV logger to save documents and questions without affecting main process performance
    """
    
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = base_dir
        self.documents_file = os.path.join(base_dir, "documents_log.csv")
        self.questions_file = os.path.join(base_dir, "questions_log.csv")
        self.batch_file = os.path.join(base_dir, "batch_processing_log.csv")
        self._initialized = False
        
        # Ensure log directory exists
        os.makedirs(base_dir, exist_ok=True)
    
    async def _ensure_initialized(self):
        """Ensure CSV files are initialized (call this before any logging operation)"""
        if self._initialized:
            return
            
        await self._initialize_csv_files()
        self._initialized = True
    
    async def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        try:
            # Documents CSV headers
            if not os.path.exists(self.documents_file):
                async with aiofiles.open(self.documents_file, 'w', newline='', encoding='utf-8') as f:
                    await f.write('timestamp,document_url,document_type,document_length,chunks_created,embeddings_dimension,processing_time,metadata\n')
            
            # Questions CSV headers
            if not os.path.exists(self.questions_file):
                async with aiofiles.open(self.questions_file, 'w', newline='', encoding='utf-8') as f:
                    await f.write('timestamp,question,answer,context_length,chunks_used,retrieval_mode,similarity_threshold,processing_time,success,metadata\n')
            
            # Batch processing CSV headers
            if not os.path.exists(self.batch_file):
                async with aiofiles.open(self.batch_file, 'w', newline='', encoding='utf-8') as f:
                    await f.write('timestamp,document_url,questions_count,total_processing_time,avg_time_per_question,questions_per_second,success_count,error_count,metadata\n')
                    
        except Exception as e:
            print(f"⚠️ CSV Logger initialization error: {e}")
    
    async def log_document_async(self, document_url: str, document_type: str, 
                               document_length: int, chunks_created: int, 
                               embeddings_dimension: int, processing_time: float, 
                               metadata: Dict[str, Any] = None):
        """Log document processing to CSV asynchronously"""
        try:
            await self._ensure_initialized()
            
            timestamp = datetime.now().isoformat()
            metadata_json = json.dumps(metadata or {})
            
            # Prepare CSV row
            row = f'"{timestamp}","{document_url}","{document_type}",{document_length},{chunks_created},{embeddings_dimension},{processing_time:.3f},"{metadata_json}"\n'
            
            # Write to CSV file asynchronously
            async with aiofiles.open(self.documents_file, 'a', encoding='utf-8') as f:
                await f.write(row)
                
            print(f"📊 CSV: Document logged to {self.documents_file}")
            
        except Exception as e:
            print(f"⚠️ Async document CSV logging failed: {e}")
    
    async def log_question_async(self, question: str, answer: str, context_length: int,
                               chunks_used: int = None, retrieval_mode: str = None,
                               similarity_threshold: float = None, processing_time: float = None,
                               success: bool = True, metadata: Dict[str, Any] = None):
        """Log question-answer pairs to CSV asynchronously"""
        try:
            await self._ensure_initialized()
            
            timestamp = datetime.now().isoformat()
            metadata_json = json.dumps(metadata or {})
            
            # Clean text for CSV (escape quotes and newlines)
            question_clean = question.replace('"', '""').replace('\n', ' ').replace('\r', ' ')
            answer_clean = answer.replace('"', '""').replace('\n', ' ').replace('\r', ' ')
            
            # Prepare CSV row
            row = f'"{timestamp}","{question_clean}","{answer_clean}",{context_length},{chunks_used or 0},"{retrieval_mode or ""}",{similarity_threshold or 0.0},{processing_time or 0.0},{success},"{metadata_json}"\n'
            
            # Write to CSV file asynchronously
            async with aiofiles.open(self.questions_file, 'a', encoding='utf-8') as f:
                await f.write(row)
                
            print(f"📊 CSV: Q&A logged to {self.questions_file}")
            
        except Exception as e:
            print(f"⚠️ Async Q&A CSV logging failed: {e}")
    
    async def log_batch_async(self, document_url: str, questions: List[str], answers: List[str],
                            total_processing_time: float, questions_per_second: float,
                            success_count: int, error_count: int, metadata: Dict[str, Any] = None):
        """Log batch processing to CSV asynchronously"""
        try:
            await self._ensure_initialized()
            
            timestamp = datetime.now().isoformat()
            questions_count = len(questions)
            avg_time_per_question = total_processing_time / questions_count if questions_count > 0 else 0
            metadata_json = json.dumps(metadata or {})
            
            # Prepare CSV row
            row = f'"{timestamp}","{document_url}",{questions_count},{total_processing_time:.3f},{avg_time_per_question:.3f},{questions_per_second:.3f},{success_count},{error_count},"{metadata_json}"\n'
            
            # Write to CSV file asynchronously
            async with aiofiles.open(self.batch_file, 'a', encoding='utf-8') as f:
                await f.write(row)
            
            # Also log individual Q&A pairs
            for i, (question, answer) in enumerate(zip(questions, answers)):
                await self.log_question_async(
                    question=question,
                    answer=answer,
                    context_length=metadata.get('context_length', 0) if metadata else 0,
                    chunks_used=metadata.get('chunk_count', 0) if metadata else 0,
                    retrieval_mode="batch_parallel",
                    processing_time=avg_time_per_question,
                    success=i < success_count,
                    metadata={
                        "batch_index": i,
                        "batch_total": questions_count,
                        "document_url": document_url,
                        "processing_mode": "batch"
                    }
                )
                
            print(f"📊 CSV: Batch processing logged to {self.batch_file}")
            
        except Exception as e:
            print(f"⚠️ Async batch CSV logging failed: {e}")
    
    def log_document_background(self, *args, **kwargs):
        """Fire-and-forget document logging"""
        asyncio.create_task(self.log_document_async(*args, **kwargs))
    
    def log_question_background(self, *args, **kwargs):
        """Fire-and-forget question logging"""
        asyncio.create_task(self.log_question_async(*args, **kwargs))
    
    def log_batch_background(self, *args, **kwargs):
        """Fire-and-forget batch logging"""
        asyncio.create_task(self.log_batch_async(*args, **kwargs))

# Global instance
csv_logger = AsyncCSVLogger()
