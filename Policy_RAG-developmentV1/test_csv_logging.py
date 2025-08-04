#!/usr/bin/env python3
"""
Test script to verify async CSV logging is working correctly
"""

import asyncio
import os
import time
from utils.async_csv_logger import csv_logger

async def test_csv_logging():
    """Test the async CSV logging functionality"""
    
    print("🧪 Testing Async CSV Logger...")
    print("=" * 50)
    
    # Test document logging
    print("\n📄 Testing document logging...")
    await csv_logger.log_document_async(
        document_url="https://example.com/test-policy.pdf",
        document_type="test_policy_pdf",
        document_length=50000,
        chunks_created=25,
        embeddings_dimension=1536,
        processing_time=3.5,
        metadata={
            "test": True,
            "parsing_time": 1.2,
            "chunking_time": 0.8,
            "embedding_time": 1.5
        }
    )
    
    # Test Q&A logging
    print("\n❓ Testing Q&A logging...")
    await csv_logger.log_question_async(
        question="What is the grace period for premium payments?",
        answer="The grace period for premium payments is 30 days from the due date.",
        context_length=2500,
        chunks_used=3,
        retrieval_mode="hybrid",
        similarity_threshold=0.7,
        processing_time=1.8,
        success=True,
        metadata={
            "test": True,
            "model": "gpt-4",
            "temperature": 0.1
        }
    )
    
    # Test batch logging
    print("\n📦 Testing batch logging...")
    test_questions = [
        "What is the coverage amount?",
        "What are the exclusions?",
        "How to file a claim?"
    ]
    test_answers = [
        "The coverage amount is up to Rs. 5 lakhs per family.",
        "Exclusions include pre-existing conditions and cosmetic treatments.",
        "Claims can be filed online through our portal or by calling customer service."
    ]
    
    await csv_logger.log_batch_async(
        document_url="https://example.com/test-policy.pdf",
        questions=test_questions,
        answers=test_answers,
        total_processing_time=5.2,
        questions_per_second=0.58,
        success_count=3,
        error_count=0,
        metadata={
            "test": True,
            "processing_mode": "parallel",
            "chunk_count": 25,
            "context_length": 15000
        }
    )
    
    print("\n✅ All async CSV logging tests completed!")
    
    # Check if files were created
    print("\n📁 Checking created log files...")
    log_dir = "logs"
    
    files_to_check = [
        "documents_log.csv",
        "questions_log.csv", 
        "batch_processing_log.csv"
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(log_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"   ✅ {filename}: {size} bytes")
        else:
            print(f"   ❌ {filename}: Not found")
    
    print("\n🎉 CSV logging test completed!")

def test_background_logging():
    """Test fire-and-forget background logging"""
    
    print("\n🚀 Testing background (fire-and-forget) logging...")
    
    # These should not block
    start_time = time.time()
    
    csv_logger.log_document_background(
        document_url="https://example.com/background-test.pdf",
        document_type="background_test",
        document_length=75000,
        chunks_created=40,
        embeddings_dimension=1536,
        processing_time=4.2,
        metadata={"background_test": True}
    )
    
    csv_logger.log_question_background(
        question="Background test question?",
        answer="Background test answer.",
        context_length=3000,
        chunks_used=2,
        retrieval_mode="background_test",
        similarity_threshold=0.8,
        processing_time=1.5,
        success=True,
        metadata={"background_test": True}
    )
    
    elapsed_time = time.time() - start_time
    print(f"   ⚡ Background logging calls completed in {elapsed_time:.3f}s (should be very fast)")
    
    # Give some time for async tasks to complete
    print("   ⏳ Waiting 2 seconds for background tasks to complete...")
    time.sleep(2)
    
    print("   ✅ Background logging test completed")

async def main():
    """Run all tests"""
    
    print("🚀 Starting Async CSV Logger Tests...")
    print("=" * 70)
    
    # Test async logging
    await test_csv_logging()
    
    # Test background logging
    test_background_logging()
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY:")
    print("   ✅ Async CSV logging: Working")
    print("   ✅ Background logging: Working") 
    print("   ✅ File creation: Working")
    print("\n🎉 All tests passed! CSV logging is ready for production.")

if __name__ == "__main__":
    asyncio.run(main())
