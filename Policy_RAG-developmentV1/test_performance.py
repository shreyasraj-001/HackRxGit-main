"""
Test script to compare chunking performance
Run this to see the speed improvements
"""

import time
import requests
import io
import pdfplumber as pp
from utils.splitter import semantic_split
from utils.fast_splitter import lightning_fast_split, smart_fast_split, adaptive_split

class PDFDocument:
    def __init__(self, link: str):
        self.link = link
        self.response = requests.get(self.link)
        self.parsed_data = ""    
    
    def load_pdf(self):
        """Load PDF from URL and return pages"""
        with pp.open(io.BytesIO(self.response.content)) as pdf:
            return pdf.pages

    def parse_pdf(self):
        """Parse PDF and extract text content"""
        pages = self.load_pdf()
        for page in pages:
            self.parsed_data += page.extract_text(layout=True)
        return self.parsed_data

def test_chunking_performance():
    """Test different chunking strategies and compare performance"""
    
    # Use the test policy document
    file_link = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    print("📄 Loading PDF document...")
    document = PDFDocument(file_link)
    pdf_text = document.parse_pdf()
    print(f"✅ PDF loaded: {len(pdf_text)} characters")
    
    # Test different chunking strategies
    strategies = [
        ("Lightning Fast Split", lightning_fast_split),
        ("Smart Fast Split", smart_fast_split),
        ("Adaptive Split", adaptive_split),
        ("Original Semantic Split", semantic_split),
    ]
    
    results = {}
    
    for strategy_name, strategy_func in strategies:
        print(f"\n🧪 Testing {strategy_name}...")
        
        try:
            start_time = time.time()
            
            if strategy_name == "Adaptive Split":
                chunks = strategy_func(pdf_text, chunk_size=1000, max_time=3.0)
            else:
                chunks = strategy_func(pdf_text, chunk_size=1000)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate chunk quality metrics
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
            min_chunk_size = min(len(chunk) for chunk in chunks) if chunks else 0
            max_chunk_size = max(len(chunk) for chunk in chunks) if chunks else 0
            
            results[strategy_name] = {
                "time": elapsed,
                "chunks": len(chunks),
                "avg_size": avg_chunk_size,
                "min_size": min_chunk_size,
                "max_size": max_chunk_size,
                "success": True
            }
            
            print(f"✅ {strategy_name}: {len(chunks)} chunks in {elapsed:.3f}s")
            print(f"   📊 Avg size: {avg_chunk_size:.0f}, Min: {min_chunk_size}, Max: {max_chunk_size}")
            
        except Exception as e:
            print(f"❌ {strategy_name} failed: {e}")
            results[strategy_name] = {
                "time": float('inf'),
                "chunks": 0,
                "success": False,
                "error": str(e)
            }
    
    # Print performance comparison
    print("\n" + "="*60)
    print("📈 PERFORMANCE COMPARISON")
    print("="*60)
    
    # Sort by time (successful strategies only)
    successful_results = {k: v for k, v in results.items() if v["success"]}
    sorted_results = sorted(successful_results.items(), key=lambda x: x[1]["time"])
    
    print(f"{'Strategy':<25} {'Time (s)':<10} {'Chunks':<8} {'Avg Size':<10}")
    print("-" * 60)
    
    for strategy_name, result in sorted_results:
        print(f"{strategy_name:<25} {result['time']:<10.3f} {result['chunks']:<8} {result['avg_size']:<10.0f}")
    
    # Calculate speed improvements
    if len(sorted_results) > 1:
        fastest = sorted_results[0]
        slowest = sorted_results[-1]
        speedup = slowest[1]["time"] / fastest[1]["time"]
        
        print(f"\n🚀 Speed Improvement: {fastest[0]} is {speedup:.1f}x faster than {slowest[0]}")
    
    # Show failed strategies
    failed_results = {k: v for k, v in results.items() if not v["success"]}
    if failed_results:
        print(f"\n❌ Failed Strategies:")
        for strategy_name, result in failed_results.items():
            print(f"   {strategy_name}: {result['error']}")
    
    return results

def test_chunk_quality(chunks, sample_size=3):
    """Test the quality of chunks by displaying samples"""
    print(f"\n📝 CHUNK QUALITY SAMPLE (showing {sample_size} chunks)")
    print("="*60)
    
    for i in range(min(sample_size, len(chunks))):
        chunk = chunks[i]
        print(f"\n--- Chunk {i+1} (Length: {len(chunk)}) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

if __name__ == "__main__":
    print("🧪 CHUNKING PERFORMANCE TEST")
    print("="*60)
    
    # Run performance test
    results = test_chunking_performance()
    
    # Test chunk quality for the fastest successful strategy
    successful_results = {k: v for k, v in results.items() if v["success"]}
    if successful_results:
        fastest_strategy = min(successful_results.items(), key=lambda x: x[1]["time"])
        strategy_name = fastest_strategy[0]
        
        print(f"\n🏆 Testing chunk quality for fastest strategy: {strategy_name}")
        
        # Re-run the fastest strategy to get chunks for quality test
        file_link = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        document = PDFDocument(file_link)
        pdf_text = document.parse_pdf()
        
        if strategy_name == "Lightning Fast Split":
            chunks = lightning_fast_split(pdf_text, chunk_size=1000)
        elif strategy_name == "Smart Fast Split":
            chunks = smart_fast_split(pdf_text, chunk_size=1000)
        elif strategy_name == "Adaptive Split":
            chunks = adaptive_split(pdf_text, chunk_size=1000, max_time=3.0)
        else:
            chunks = semantic_split(pdf_text, chunk_size=1000)
        
        test_chunk_quality(chunks)
