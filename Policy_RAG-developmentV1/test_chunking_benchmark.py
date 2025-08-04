#!/usr/bin/env python3
"""
Comprehensive chunking benchmark script to compare all available chunking methods
"""

import time
import requests
import fitz  # PyMuPDF
from typing import List, Dict, Any
import statistics
import json

# Import our chunking utilities
from utils.splitter import semantic_split
from utils.fast_splitter import lightning_fast_split, smart_fast_split, adaptive_split
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkingBenchmark:
    """Comprehensive benchmarking for all chunking methods"""
    
    def __init__(self, pdf_url: str):
        self.pdf_url = pdf_url
        self.text_content = ""
        self.results = {}
        
    def load_pdf_content(self):
        """Load and extract text from PDF using PyMuPDF"""
        print("📄 Loading PDF content...")
        start_time = time.time()
        
        try:
            # Download PDF
            response = requests.get(
                self.pdf_url, 
                timeout=10,
                stream=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            response.raise_for_status()
            
            # Parse with PyMuPDF
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            text_parts = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            
            self.text_content = '\n'.join(text_parts)
            pdf_document.close()
            
            load_time = time.time() - start_time
            print(f"✅ PDF loaded: {len(self.text_content)} characters in {load_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            raise
    
    def benchmark_semantic_chunking(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> Dict[str, Any]:
        """Benchmark semantic chunking"""
        print(f"\n🧠 Testing Semantic Chunking (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        results = []
        for run in range(3):  # 3 runs for averaging
            start_time = time.time()
            try:
                chunks = semantic_split(self.text_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                duration = time.time() - start_time
                
                # Analyze chunk quality
                chunk_sizes = [len(chunk) for chunk in chunks if chunk.strip()]
                avg_chunk_size = statistics.mean(chunk_sizes) if chunk_sizes else 0
                
                results.append({
                    'duration': duration,
                    'num_chunks': len(chunks),
                    'avg_chunk_size': avg_chunk_size,
                    'success': True
                })
                
                print(f"   Run {run+1}: {duration:.3f}s - {len(chunks)} chunks, avg size: {avg_chunk_size:.0f}")
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"   Run {run+1}: FAILED in {duration:.3f}s - {e}")
                results.append({
                    'duration': duration,
                    'num_chunks': 0,
                    'avg_chunk_size': 0,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate averages
        successful_runs = [r for r in results if r['success']]
        if successful_runs:
            return {
                'method': 'Semantic Chunking',
                'avg_duration': statistics.mean([r['duration'] for r in successful_runs]),
                'min_duration': min([r['duration'] for r in successful_runs]),
                'max_duration': max([r['duration'] for r in successful_runs]),
                'avg_chunks': statistics.mean([r['num_chunks'] for r in successful_runs]),
                'avg_chunk_size': statistics.mean([r['avg_chunk_size'] for r in successful_runs]),
                'success_rate': len(successful_runs) / len(results),
                'runs': results
            }
        else:
            return {
                'method': 'Semantic Chunking',
                'avg_duration': float('inf'),
                'success_rate': 0,
                'error': 'All runs failed',
                'runs': results
            }
    
    def benchmark_lightning_fast_split(self, chunk_size: int = 1000) -> Dict[str, Any]:
        """Benchmark lightning fast split"""
        print(f"\n⚡ Testing Lightning Fast Split (chunk_size={chunk_size})")
        
        results = []
        for run in range(3):
            start_time = time.time()
            try:
                chunks = lightning_fast_split(self.text_content, chunk_size=chunk_size)
                duration = time.time() - start_time
                
                chunk_sizes = [len(chunk) for chunk in chunks if chunk.strip()]
                avg_chunk_size = statistics.mean(chunk_sizes) if chunk_sizes else 0
                
                results.append({
                    'duration': duration,
                    'num_chunks': len(chunks),
                    'avg_chunk_size': avg_chunk_size,
                    'success': True
                })
                
                print(f"   Run {run+1}: {duration:.3f}s - {len(chunks)} chunks, avg size: {avg_chunk_size:.0f}")
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"   Run {run+1}: FAILED in {duration:.3f}s - {e}")
                results.append({
                    'duration': duration,
                    'num_chunks': 0,
                    'avg_chunk_size': 0,
                    'success': False,
                    'error': str(e)
                })
        
        successful_runs = [r for r in results if r['success']]
        if successful_runs:
            return {
                'method': 'Lightning Fast Split',
                'avg_duration': statistics.mean([r['duration'] for r in successful_runs]),
                'min_duration': min([r['duration'] for r in successful_runs]),
                'max_duration': max([r['duration'] for r in successful_runs]),
                'avg_chunks': statistics.mean([r['num_chunks'] for r in successful_runs]),
                'avg_chunk_size': statistics.mean([r['avg_chunk_size'] for r in successful_runs]),
                'success_rate': len(successful_runs) / len(results),
                'runs': results
            }
        else:
            return {
                'method': 'Lightning Fast Split',
                'avg_duration': float('inf'),
                'success_rate': 0,
                'error': 'All runs failed',
                'runs': results
            }
    
    def benchmark_smart_fast_split(self, chunk_size: int = 1000) -> Dict[str, Any]:
        """Benchmark smart fast split"""
        print(f"\n🎯 Testing Smart Fast Split (chunk_size={chunk_size})")
        
        results = []
        for run in range(3):
            start_time = time.time()
            try:
                chunks = smart_fast_split(self.text_content, chunk_size=chunk_size)
                duration = time.time() - start_time
                
                chunk_sizes = [len(chunk) for chunk in chunks if chunk.strip()]
                avg_chunk_size = statistics.mean(chunk_sizes) if chunk_sizes else 0
                
                results.append({
                    'duration': duration,
                    'num_chunks': len(chunks),
                    'avg_chunk_size': avg_chunk_size,
                    'success': True
                })
                
                print(f"   Run {run+1}: {duration:.3f}s - {len(chunks)} chunks, avg size: {avg_chunk_size:.0f}")
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"   Run {run+1}: FAILED in {duration:.3f}s - {e}")
                results.append({
                    'duration': duration,
                    'num_chunks': 0,
                    'avg_chunk_size': 0,
                    'success': False,
                    'error': str(e)
                })
        
        successful_runs = [r for r in results if r['success']]
        if successful_runs:
            return {
                'method': 'Smart Fast Split',
                'avg_duration': statistics.mean([r['duration'] for r in successful_runs]),
                'min_duration': min([r['duration'] for r in successful_runs]),
                'max_duration': max([r['duration'] for r in successful_runs]),
                'avg_chunks': statistics.mean([r['num_chunks'] for r in successful_runs]),
                'avg_chunk_size': statistics.mean([r['avg_chunk_size'] for r in successful_runs]),
                'success_rate': len(successful_runs) / len(results),
                'runs': results
            }
        else:
            return {
                'method': 'Smart Fast Split',
                'avg_duration': float('inf'),
                'success_rate': 0,
                'error': 'All runs failed',
                'runs': results
            }
    
    def benchmark_adaptive_split(self, chunk_size: int = 1000) -> Dict[str, Any]:
        """Benchmark adaptive split"""
        print(f"\n🔄 Testing Adaptive Split (chunk_size={chunk_size})")
        
        results = []
        for run in range(3):
            start_time = time.time()
            try:
                chunks = adaptive_split(self.text_content, chunk_size=chunk_size)
                duration = time.time() - start_time
                
                chunk_sizes = [len(chunk) for chunk in chunks if chunk.strip()]
                avg_chunk_size = statistics.mean(chunk_sizes) if chunk_sizes else 0
                
                results.append({
                    'duration': duration,
                    'num_chunks': len(chunks),
                    'avg_chunk_size': avg_chunk_size,
                    'success': True
                })
                
                print(f"   Run {run+1}: {duration:.3f}s - {len(chunks)} chunks, avg size: {avg_chunk_size:.0f}")
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"   Run {run+1}: FAILED in {duration:.3f}s - {e}")
                results.append({
                    'duration': duration,
                    'num_chunks': 0,
                    'avg_chunk_size': 0,
                    'success': False,
                    'error': str(e)
                })
        
        successful_runs = [r for r in results if r['success']]
        if successful_runs:
            return {
                'method': 'Adaptive Split',
                'avg_duration': statistics.mean([r['duration'] for r in successful_runs]),
                'min_duration': min([r['duration'] for r in successful_runs]),
                'max_duration': max([r['duration'] for r in successful_runs]),
                'avg_chunks': statistics.mean([r['num_chunks'] for r in successful_runs]),
                'avg_chunk_size': statistics.mean([r['avg_chunk_size'] for r in successful_runs]),
                'success_rate': len(successful_runs) / len(results),
                'runs': results
            }
        else:
            return {
                'method': 'Adaptive Split',
                'avg_duration': float('inf'),
                'success_rate': 0,
                'error': 'All runs failed',
                'runs': results
            }
    
    def benchmark_recursive_character_splitter(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> Dict[str, Any]:
        """Benchmark recursive character text splitter (baseline)"""
        print(f"\n📝 Testing Recursive Character Splitter (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        results = []
        for run in range(3):
            start_time = time.time()
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = text_splitter.split_text(self.text_content)
                duration = time.time() - start_time
                
                chunk_sizes = [len(chunk) for chunk in chunks if chunk.strip()]
                avg_chunk_size = statistics.mean(chunk_sizes) if chunk_sizes else 0
                
                results.append({
                    'duration': duration,
                    'num_chunks': len(chunks),
                    'avg_chunk_size': avg_chunk_size,
                    'success': True
                })
                
                print(f"   Run {run+1}: {duration:.3f}s - {len(chunks)} chunks, avg size: {avg_chunk_size:.0f}")
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"   Run {run+1}: FAILED in {duration:.3f}s - {e}")
                results.append({
                    'duration': duration,
                    'num_chunks': 0,
                    'avg_chunk_size': 0,
                    'success': False,
                    'error': str(e)
                })
        
        successful_runs = [r for r in results if r['success']]
        if successful_runs:
            return {
                'method': 'Recursive Character Splitter',
                'avg_duration': statistics.mean([r['duration'] for r in successful_runs]),
                'min_duration': min([r['duration'] for r in successful_runs]),
                'max_duration': max([r['duration'] for r in successful_runs]),
                'avg_chunks': statistics.mean([r['num_chunks'] for r in successful_runs]),
                'avg_chunk_size': statistics.mean([r['avg_chunk_size'] for r in successful_runs]),
                'success_rate': len(successful_runs) / len(results),
                'runs': results
            }
        else:
            return {
                'method': 'Recursive Character Splitter',
                'avg_duration': float('inf'),
                'success_rate': 0,
                'error': 'All runs failed',
                'runs': results
            }
    
    def run_all_benchmarks(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Run all chunking benchmarks"""
        print("🧪 CHUNKING METHODS BENCHMARK")
        print("=" * 60)
        print(f"📊 Document size: {len(self.text_content):,} characters")
        print(f"🔧 Chunk size: {chunk_size}")
        print(f"🔧 Chunk overlap: {chunk_overlap}")
        print("=" * 60)
        
        # Run all benchmarks
        benchmarks = [
            self.benchmark_semantic_chunking(chunk_size, chunk_overlap),
            self.benchmark_lightning_fast_split(chunk_size),
            self.benchmark_smart_fast_split(chunk_size),
            self.benchmark_adaptive_split(chunk_size),
            self.benchmark_recursive_character_splitter(chunk_size, chunk_overlap)
        ]
        
        self.results = {result['method']: result for result in benchmarks}
        
        return benchmarks
    
    def display_comparison(self):
        """Display comprehensive comparison of all methods"""
        print("\n📊 PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # Sort by average duration (fastest first)
        sorted_results = sorted(
            [(name, data) for name, data in self.results.items() if data.get('success_rate', 0) > 0],
            key=lambda x: x[1]['avg_duration']
        )
        
        if not sorted_results:
            print("❌ No successful chunking methods to compare")
            return
        
        # Performance ranking
        print("🏆 SPEED RANKING:")
        for i, (method, data) in enumerate(sorted_results, 1):
            duration = data['avg_duration']
            chunks = data['avg_chunks']
            chunk_size = data['avg_chunk_size']
            success_rate = data['success_rate'] * 100
            
            print(f"   {i}. {method}")
            print(f"      ⏱️ Time: {duration:.3f}s | 📦 Chunks: {chunks:.0f} | 📏 Avg Size: {chunk_size:.0f} | ✅ Success: {success_rate:.0f}%")
        
        # Speedup comparison (vs slowest)
        print(f"\n🚀 SPEEDUP COMPARISON (vs {sorted_results[-1][0]}):")
        slowest_time = sorted_results[-1][1]['avg_duration']
        
        for method, data in sorted_results:
            speedup = slowest_time / data['avg_duration']
            print(f"   {method}: {speedup:.2f}x faster")
        
        # Quality comparison
        print(f"\n📊 QUALITY METRICS:")
        print(f"{'Method':<25} {'Chunks':<8} {'Avg Size':<10} {'Success Rate':<12}")
        print("-" * 55)
        
        for method, data in sorted_results:
            chunks = f"{data['avg_chunks']:.0f}"
            avg_size = f"{data['avg_chunk_size']:.0f}"
            success = f"{data['success_rate']*100:.0f}%"
            print(f"{method:<25} {chunks:<8} {avg_size:<10} {success:<12}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        fastest = sorted_results[0]
        print(f"   🥇 Fastest: {fastest[0]} ({fastest[1]['avg_duration']:.3f}s)")
        
        # Find method with best chunk count
        best_chunks = max(sorted_results, key=lambda x: x[1]['avg_chunks'])
        print(f"   📦 Most Chunks: {best_chunks[0]} ({best_chunks[1]['avg_chunks']:.0f} chunks)")
        
        # Find semantic chunking position
        semantic_result = next((data for name, data in sorted_results if 'Semantic' in name), None)
        if semantic_result:
            semantic_pos = next(i for i, (name, data) in enumerate(sorted_results, 1) if 'Semantic' in name)
            print(f"   🧠 Semantic Chunking: Rank #{semantic_pos} ({semantic_result['avg_duration']:.3f}s)")
    
    def save_results(self, filename: str = "chunking_benchmark_results.json"):
        """Save benchmark results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main benchmark execution"""
    # Policy document URL
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        # Initialize benchmark
        benchmark = ChunkingBenchmark(pdf_url)
        
        # Load PDF content
        benchmark.load_pdf_content()
        
        # Run all benchmarks
        results = benchmark.run_all_benchmarks(chunk_size=1000, chunk_overlap=100)
        
        # Display comparison
        benchmark.display_comparison()
        
        # Save results
        benchmark.save_results()
        
        print("\n🎉 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
