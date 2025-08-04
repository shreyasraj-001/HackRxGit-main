#!/usr/bin/env python3
"""
Simple test to check chunking functionality without Pinecone dependencies.
"""

from main import PDFDocument, Chunker, Embeddings

def test_chunking():
    """Test the chunking process"""
    print("🔍 Testing Chunking Process...")
    print("=" * 50)
    
    # Step 1: Load and parse PDF
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
    
    # Step 2: Test chunking
    print("\n✂️ Step 2: Testing Chunking...")
    try:
        chunker = Chunker(data)
        print(f"✅ Chunking completed. Created {len(chunker.chunks)} chunks")
        
        # Analyze chunks
        if chunker.chunks:
            print(f"📋 First chunk type: {type(chunker.chunks[0])}")
            
            # Check if chunks are Document objects or strings
            if hasattr(chunker.chunks[0], 'page_content'):
                print("📋 Chunks are Document objects")
                print(f"📋 First chunk content: {chunker.chunks[0].page_content[:200]}...")
            else:
                print("📋 Chunks are strings")
                print(f"📋 First chunk content: {str(chunker.chunks[0])[:200]}...")
            
            # Show chunk sizes
            chunk_sizes = [len(str(chunk)) for chunk in chunker.chunks[:5]]
            print(f"📋 First 5 chunk sizes: {chunk_sizes}")
            
        else:
            print("❌ No chunks created!")
            
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test embeddings (if possible)
    print("\n🧠 Step 3: Testing Embeddings...")
    try:
        embed = Embeddings(data)
        print("✅ Embeddings object created")
        
        # Try to create embeddings (this might fail without proper API key)
        try:
            embeddings_data = embed.embed_text(data[:100])  # Use small sample
            print(f"✅ Embeddings created. Size: {len(embeddings_data) if embeddings_data else 0}")
        except Exception as e:
            print(f"⚠️ Embedding creation failed (expected without API key): {e}")
            
    except Exception as e:
        print(f"❌ Embeddings initialization failed: {e}")
        return
    
    print("\n✅ Chunking test completed!")

if __name__ == "__main__":
    test_chunking() 