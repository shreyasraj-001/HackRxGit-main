#!/usr/bin/env python3
"""
Debug script to analyze the Policy RAG chunking and retrieval process.
"""

import os
from dotenv import load_dotenv
#!/usr/bin/env python3
"""
Debug script to test RAG system components individually
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_environment():
    """Check environment variables"""
    print("🔧 Environment Variables Debug")
    print("=" * 50)
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX")
    
    print(f"PINECONE_API_KEY: {'✅ Set' if pinecone_api_key else '❌ Not Set'}")
    print(f"PINECONE_INDEX: {pinecone_index}")
    
    if pinecone_api_key:
        print(f"API Key starts with: {pinecone_api_key[:20]}...")
    
    return pinecone_api_key and pinecone_index

def debug_pinecone_connection():
    """Test direct Pinecone connection"""
    print("\n🔗 Pinecone Connection Debug")
    print("=" * 50)
    
    try:
        from pinecone import Pinecone
        from langchain_pinecone.vectorstores import PineconeVectorStore
        from main import Embeddings
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # List indexes
        indexes = pc.list_indexes()
        print(f"📊 Available indexes: {[idx.name for idx in indexes]}")
        
        index_name = os.getenv("PINECONE_INDEX", "hackrx-index")
        print(f"🎯 Using index: {index_name}")
        
        # Check if index exists
        index_names = [idx.name for idx in indexes]
        if index_name not in index_names:
            print(f"❌ Index '{index_name}' not found!")
            print(f"   Available indexes: {index_names}")
            return False
        
        # Connect to index
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        print(f"✅ Connected to index: {index_name}")
        print(f"📈 Total vectors: {stats.total_vector_count}")
        print(f"📂 Namespaces: {list(stats.namespaces.keys())}")
        
        # Show namespace details
        for ns_name, ns_stats in stats.namespaces.items():
            print(f"   📁 {ns_name}: {ns_stats.vector_count} vectors")
        
        return True
        
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_embeddings():
    """Test embeddings"""
    print("\n🧠 Embeddings Debug")
    print("=" * 50)
    
    try:
        from main import Embeddings
        
        test_text = "This is a test text for embeddings"
        embed = Embeddings(test_text)
        
        # Test embedding
        vector = embed.embed_text("test query")
        
        if vector:
            print(f"✅ Embeddings working")
            print(f"📏 Vector dimension: {len(vector)}")
            print(f"🔢 First 5 values: {vector[:5]}")
            return True
        else:
            print("❌ Embeddings failed")
            return False
            
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_complete_flow():
    """Test the complete RAG flow"""
    print("\n🔄 Complete Flow Debug")
    print("=" * 50)
    
    try:
        from main import PDFDocument, Chunker, Embeddings, VectorStore, QueryEngine
        
        # Test PDF processing
        print("📄 Testing PDF processing...")
        file_link = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        
        document = PDFDocument(file_link)
        data = document.parse_pdf()
        print(f"✅ PDF processed: {len(data)} characters")
        
        # Test chunking
        print("✂️ Testing chunking...")
        chunker = Chunker(data)
        print(f"✅ Chunks created: {len(chunker.chunks)}")
        
        # Test embeddings
        print("🧠 Testing embeddings...")
        embed = Embeddings(data)
        print("✅ Embeddings initialized")
        
        # Test vector store
        print("🗄️ Testing vector store...")
        sig_part = file_link.split("sig=")[1]
        sig = sig_part.split("&")[0] if "&" in sig_part else sig_part
        
        vector_store = VectorStore(chunker.chunks, embed.embeddings, sig)
        
        # Try to add chunks (this might connect to existing namespace)
        result = vector_store.add_chunks()
        print(f"✅ Vector store operation completed: {result is not None}")
        
        # Test querying
        print("🔍 Testing query...")
        query_result = vector_store.query_chunks("What is the grace period?", k=3)
        print(f"✅ Query completed: {len(query_result)} results")
        
        if query_result:
            print("🎉 RAG system is working!")
            for i, result in enumerate(query_result[:2]):
                content = result.page_content if hasattr(result, 'page_content') else str(result)
                print(f"📄 Sample result {i+1}: {content[:150]}...")
        else:
            print("⚠️ No query results - might need to check namespace or add documents")
        
        return len(query_result) > 0
        
    except Exception as e:
        print(f"❌ Complete flow error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🐛 RAG System Debug Suite")
    print("=" * 60)
    
    # Step 1: Check environment
    if not debug_environment():
        print("❌ Environment setup issues found!")
        return
    
    # Step 2: Test Pinecone connection
    if not debug_pinecone_connection():
        print("❌ Pinecone connection issues found!")
        return
    
    # Step 3: Test embeddings
    if not debug_embeddings():
        print("❌ Embeddings issues found!")
        return
    
    # Step 4: Test complete flow
    if debug_complete_flow():
        print("\n🎉 All systems working!")
        print("✅ Your RAG system should be functional")
    else:
        print("\n⚠️ Issues found in complete flow")
        print("🔧 Check the detailed logs above")
    
    print("\n💡 Recommendations:")
    print("1. Restart your FastAPI server after fixing any issues")
    print("2. Make sure you're using the correct index name")
    print("3. Check if documents were actually added to the vector store")

if __name__ == "__main__":
    main()

def debug_rag_process():
    """Debug the RAG process step by step"""
    load_dotenv()
    
    print("🔍 Debugging Policy RAG Process...")
    print("=" * 60)
    
    # Step 1: Check environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY not set!")
        return
    
    print("✅ Environment variables OK")
    
    # Step 2: Load and parse PDF
    print("\n📄 Step 1: Loading PDF...")
    file_link = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        document = PDFDocument(file_link)
        data = document.parse_pdf()
        print(f"✅ PDF parsed successfully. Text length: {len(data)} characters")
        print(f"📝 First 200 chars: {data[:200]}...")
    except Exception as e:
        print(f"❌ PDF parsing failed: {e}")
        return
    
    # Step 3: Chunking
    print("\n✂️ Step 2: Chunking...")
    try:
        chunker = Chunker(data)
        print(f"✅ Chunking completed. Created {len(chunker.chunks)} chunks")
        
        # Check chunk types
        if chunker.chunks:
            first_chunk = chunker.chunks[0]
            print(f"📋 First chunk type: {type(first_chunk)}")
            if hasattr(first_chunk, 'page_content'):
                print(f"📋 First chunk content (first 100 chars): {first_chunk.page_content[:100]}...")
            else:
                print(f"📋 First chunk content (first 100 chars): {str(first_chunk)[:100]}...")
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        return
    
    # Step 4: Embeddings
    print("\n🧠 Step 3: Embeddings...")
    try:
        embed = Embeddings(data)
        embeddings_data = embed.embed_text(data)
        print(f"✅ Embeddings created. Size: {len(embeddings_data) if embeddings_data else 0}")
    except Exception as e:
        print(f"❌ Embeddings failed: {e}")
        return
    
    # Step 5: Vector Store
    print("\n🗄️ Step 4: Vector Store...")
    try:
        sig_part = file_link.split("sig=")[1]
        sig = sig_part.split("&")[0] if "&" in sig_part else sig_part
        
        vector_store = VectorStore(chunker.chunks, embed.embeddings, sig)
        print(f"✅ Vector store initialized")
        print(f"📊 Index name: {vector_store.index_name}")
        print(f"📊 Namespace: {vector_store.sig}")
        print(f"📊 Documents count: {len(vector_store.documents)}")
    except Exception as e:
        print(f"❌ Vector store initialization failed: {e}")
        return
    
    # Step 6: Add chunks
    print("\n➕ Step 5: Adding chunks...")
    try:
        vector_store_result = vector_store.add_chunks()
        if vector_store_result:
            print("✅ Chunks added successfully")
        else:
            print("⚠️ Chunks not added (namespace might already exist)")
    except Exception as e:
        print(f"❌ Adding chunks failed: {e}")
        return
    
    # Step 7: Query
    print("\n🔍 Step 6: Querying...")
    try:
        query = "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
        query_res = vector_store.query_chunks(query)
        print(f"✅ Query completed. Found {len(query_res)} results")
        
        if query_res:
            for i, result in enumerate(query_res[:3]):  # Show first 3 results
                content = result.page_content if hasattr(result, 'page_content') else str(result)
                print(f"📄 Result {i+1}: {content[:150]}...")
        else:
            print("⚠️ No results found")
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return
    
    print("\n✅ Debug completed!")

if __name__ == "__main__":
    debug_rag_process() 