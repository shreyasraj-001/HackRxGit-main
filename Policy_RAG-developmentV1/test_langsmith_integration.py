#!/usr/bin/env python3
"""
Test script to verify LangSmith integration is working correctly
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test imports
try:
    from main import LangSmithManager
    print("✅ Successfully imported LangSmithManager")
except ImportError as e:
    print(f"❌ Failed to import LangSmithManager: {e}")
    exit(1)

def test_langsmith_manager():
    """Test LangSmith manager functionality"""
    
    print("\n🧪 Testing LangSmith Manager...")
    
    # Initialize manager
    try:
        manager = LangSmithManager()
        print("✅ LangSmith Manager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LangSmith Manager: {e}")
        return False
    
    # Test document processing logging
    try:
        print("\n📄 Testing document processing logging...")
        manager.log_document_processing(
            document_url="test://example.pdf",
            document_type="test_pdf",
            chunks_created=10,
            embeddings_dimension=1536,
            processing_time=2.5,
            metadata={
                "test": True,
                "framework": "langsmith_integration_test"
            }
        )
        print("✅ Document processing logged successfully")
    except Exception as e:
        print(f"❌ Document processing logging failed: {e}")
        return False
    
    # Test Q&A logging
    try:
        print("\n❓ Testing Q&A logging...")
        manager.log_question_answer(
            question="What is a test question?",
            answer="This is a test answer from the integration test.",
            context="Test context for validation",
            metadata={
                "test": True,
                "response_time": 1.2,
                "chunks_used": 3
            }
        )
        print("✅ Q&A logging successful")
    except Exception as e:
        print(f"❌ Q&A logging failed: {e}")
        return False
    
    # Test batch Q&A logging
    try:
        print("\n📦 Testing batch Q&A logging...")
        manager.log_batch_questions_answers(
            questions=["Test question 1?", "Test question 2?"],
            answers=["Test answer 1", "Test answer 2"],
            document_context="Batch test context",
            metadata={
                "test": True,
                "batch_size": 2,
                "processing_mode": "test_batch"
            }
        )
        print("✅ Batch Q&A logging successful")
    except Exception as e:
        print(f"❌ Batch Q&A logging failed: {e}")
        return False
    
    return True

def test_environment_setup():
    """Test that environment variables are properly configured"""
    
    print("\n🔧 Testing environment configuration...")
    
    # Check LangSmith configuration
    required_vars = [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT", 
        "LANGCHAIN_API_KEY",
        "LANGSMITH_ENABLE_STORAGE"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("💡 Make sure your .env file contains all LangSmith configuration")
        return False
    else:
        print("✅ All required environment variables are set")
        
    # Show current values (without exposing sensitive data)
    print(f"   LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
    print(f"   LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")
    print(f"   LANGSMITH_ENABLE_STORAGE: {os.getenv('LANGSMITH_ENABLE_STORAGE')}")
    print(f"   LANGCHAIN_API_KEY: {'*' * 20}...{os.getenv('LANGCHAIN_API_KEY', '')[-4:]}")
    
    return True

def main():
    """Run all tests"""
    
    print("🚀 Starting LangSmith Integration Tests...")
    print("=" * 50)
    
    # Test environment setup
    env_ok = test_environment_setup()
    
    # Test LangSmith manager
    manager_ok = test_langsmith_manager()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   Environment Setup: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"   LangSmith Manager: {'✅ PASS' if manager_ok else '❌ FAIL'}")
    
    if env_ok and manager_ok:
        print("\n🎉 All tests passed! LangSmith integration is working correctly.")
        print("💡 You can now use the RAG system with full LangSmith monitoring.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
