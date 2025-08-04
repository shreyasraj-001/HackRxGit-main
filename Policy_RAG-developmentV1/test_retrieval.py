#!/usr/bin/env python3
"""
Test script to demonstrate the retrieval and answer generation system
"""

import requests
import json
import time

def test_retrieval_system():
    """Test the complete retrieval system"""
    print("🚀 Testing RAG Retrieval System")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Step 1: Initialize the system
    print("📥 Step 1: Initializing RAG system...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            result = response.json()
            print("✅ System initialized successfully!")
            print(f"   📊 Chunks created: {result.get('chunks_count', 0)}")
            print(f"   🧠 Embeddings size: {result.get('embeddings_size', 0)}")
            print(f"   🗄️ Vector store ready: {result.get('vector_store_success', False)}")
            
            # Show test query result
            test_query = result.get('test_query', {})
            if test_query.get('success'):
                print(f"   🔍 Test query successful!")
                print(f"   📝 Answer: {test_query.get('answer', '')[:200]}...")
            
            time.sleep(2)
        else:
            print(f"❌ Failed to initialize system: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return False
    
    # Step 2: Test various queries
    test_questions = [
        "What is the grace period for premium payment?",
        "What are the coverage benefits?",
        "How much is the premium amount?",
        "What is the policy term?",
        "Are there any exclusions in this policy?",
        "What documents are required for claims?",
        "What is the waiting period for coverage?",
        "Can I renew this policy?"
    ]
    
    print("\n🔍 Step 2: Testing various queries...")
    print("-" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Query {i}: {question}")
        try:
            # Using GET endpoint for simplicity
            encoded_question = requests.utils.quote(question)
            response = requests.get(f"{base_url}/query/{encoded_question}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ Success!")
                    print(f"   📝 Answer: {result.get('answer', '')}")
                    print(f"   📊 Chunks retrieved: {result.get('num_chunks_retrieved', 0)}")
                else:
                    print(f"⚠️ Query failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        time.sleep(1)  # Small delay between queries
    
    # Step 3: Test POST endpoint
    print(f"\n📤 Step 3: Testing POST endpoint...")
    try:
        post_data = {
            "question": "What is the sum insured under this policy?",
            "k": 3
        }
        
        response = requests.post(f"{base_url}/query", json=post_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ POST query successful!")
            print(f"   📝 Answer: {result.get('answer', '')}")
            print(f"   📊 Chunks retrieved: {result.get('num_chunks_retrieved', 0)}")
        else:
            print(f"❌ POST query failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error with POST query: {e}")
    
    print(f"\n🎉 Testing completed!")
    print("=" * 60)

def test_local_components():
    """Test components locally without server"""
    print("🧪 Testing Local Components (No Server Required)")
    print("=" * 60)
    
    try:
        from main import PDFDocument, Chunker, Embeddings, VectorStore, QueryEngine
        
        # Test PDF loading
        print("📄 Testing PDF processing...")
        file_link = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        
        document = PDFDocument(file_link)
        data = document.parse_pdf()
        print(f"✅ PDF processed: {len(data)} characters extracted")
        
        # Test chunking
        print("✂️ Testing chunking...")
        chunker = Chunker(data)
        print(f"✅ Chunking complete: {len(chunker.chunks)} chunks created")
        
        # Test embeddings
        print("🧠 Testing embeddings...")
        embed = Embeddings(data)
        print("✅ Embeddings initialized")
        
        # Test vector store
        print("🗄️ Testing FAISS vector store...")
        sig = "test_document"
        vector_store = VectorStore(chunker.chunks, embed.embeddings, sig)
        print("✅ FAISS vector store initialized")
        
        # Add chunks to vector store
        print("📚 Adding chunks to FAISS...")
        vector_store.add_chunks()
        print("✅ Chunks added to FAISS vector store")
        
        # Test query engine
        print("🤖 Testing query engine...")
        query_engine = QueryEngine(vector_store, embed.embeddings)
        
        test_queries = [
            "What is the grace period?",
            "What are the benefits?",
            "How to make a claim?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: {query}")
            result = query_engine.answer_query(query)
            if result.get('success'):
                print(f"✅ Answer: {result.get('answer', '')[:150]}...")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        print("\n✅ All local components tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing local components: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 RAG System Testing Suite")
    print("=" * 60)
    
    choice = input("Choose test type:\n1. Server-based testing (requires running server)\n2. Local component testing\n3. Both\nEnter choice (1/2/3): ")
    
    if choice == "1":
        test_retrieval_system()
    elif choice == "2":
        test_local_components()
    elif choice == "3":
        test_local_components()
        print("\n" + "="*60)
        input("Press Enter after starting the server (uvicorn main:app --reload)...")
        test_retrieval_system()
    else:
        print("Invalid choice. Running local tests...")
        test_local_components()
