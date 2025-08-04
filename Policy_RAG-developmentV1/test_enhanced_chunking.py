"""
Comprehensive Chunking Performance Test
Tests various chunking strategies and provides detailed performance analysis
"""

import time
import requests
import io
import statistics
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import PDF processing
try:
    import pdfplumber as pp
except ImportError:
    logger.warning("pdfplumber not available, using fitz instead")
    import fitz
    pp = None

# Import chunking strategies
from utils.splitter import (
    semantic_split, 
    fast_rule_based_split, 
    smart_policy_split,
    adaptive_split,
    lightning_fast_split,
    smart_fast_split,
    contextual_split
)

try:
    from utils.enhanced_chunking import (
        EnhancedChunkingFramework,
        ChunkingConfig,
        ChunkingStrategy
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    logger.warning("Enhanced chunking framework not available")

@dataclass
class ChunkingResult:
    """Results from a chunking strategy test"""
    strategy_name: str
    chunks: List[str]
    execution_time: float
    chunk_count: int
    avg_chunk_length: float
    std_chunk_length: float
    min_chunk_length: int
    max_chunk_length: int
    error: str = None
    quality_score: float = 0.0
    memory_usage: float = 0.0

class PDFDocument:
    """PDF document loader with multiple backend support"""
    
    def __init__(self, link: str):
        self.link = link
        self.response = requests.get(self.link, timeout=30)
        self.parsed_data = ""
    
    def parse_pdf_pdfplumber(self) -> str:
        """Parse PDF using pdfplumber"""
        with pp.open(io.BytesIO(self.response.content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text(layout=True) + "\n"
            return text
    
    def parse_pdf_fitz(self) -> str:
        """Parse PDF using PyMuPDF (fitz)"""
        doc = fitz.open(stream=self.response.content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    
    def parse_pdf(self) -> str:
        """Parse PDF using the best available method"""
        try:
            if pp is not None:
                self.parsed_data = self.parse_pdf_pdfplumber()
            else:
                self.parsed_data = self.parse_pdf_fitz()
            return self.parsed_data
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise

class ChunkingBenchmark:
    """Comprehensive chunking benchmark suite"""
    
    def __init__(self, pdf_url: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.pdf_url = pdf_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_text = ""
        self.results: Dict[str, ChunkingResult] = {}
    
    def load_document(self) -> str:
        """Load and parse the test document"""
        logger.info("Loading PDF document...")
        document = PDFDocument(self.pdf_url)
        self.document_text = document.parse_pdf()
        logger.info(f"Document loaded: {len(self.document_text)} characters")
        return self.document_text
    
    def test_chunking_strategy(self, strategy_name: str, chunking_func, *args, **kwargs) -> ChunkingResult:
        """Test a single chunking strategy"""
        logger.info(f"Testing {strategy_name}...")
        
        start_time = time.time()
        try:
            # Execute chunking strategy
            chunks = chunking_func(self.document_text, self.chunk_size, self.chunk_overlap, *args, **kwargs)
            execution_time = time.time() - start_time
            
            if not chunks:
                raise ValueError("No chunks generated")
            
            # Calculate statistics
            chunk_lengths = [len(chunk) for chunk in chunks]
            avg_length = statistics.mean(chunk_lengths)
            std_length = statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(chunks)
            
            result = ChunkingResult(
                strategy_name=strategy_name,
                chunks=chunks,
                execution_time=execution_time,
                chunk_count=len(chunks),
                avg_chunk_length=avg_length,
                std_chunk_length=std_length,
                min_chunk_length=min(chunk_lengths),
                max_chunk_length=max(chunk_lengths),
                quality_score=quality_score
            )
            
            logger.info(f"{strategy_name} completed: {len(chunks)} chunks in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{strategy_name} failed: {e}")
            
            return ChunkingResult(
                strategy_name=strategy_name,
                chunks=[],
                execution_time=execution_time,
                chunk_count=0,
                avg_chunk_length=0.0,
                std_chunk_length=0.0,
                min_chunk_length=0,
                max_chunk_length=0,
                error=str(e)
            )
    
    def _calculate_quality_score(self, chunks: List[str]) -> float:
        """Calculate a quality score for the chunks"""
        if not chunks:
            return 0.0
        
        score = 0.0
        
        # Factor 1: Consistency of chunk sizes (higher is better)
        chunk_lengths = [len(chunk) for chunk in chunks]
        avg_length = statistics.mean(chunk_lengths)
        std_length = statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
        consistency_score = max(0, 1 - (std_length / avg_length)) if avg_length > 0 else 0
        score += consistency_score * 0.3
        
        # Factor 2: Target size adherence (higher is better)
        target_adherence = sum(1 for length in chunk_lengths if 0.7 * self.chunk_size <= length <= 1.3 * self.chunk_size) / len(chunk_lengths)
        score += target_adherence * 0.3
        
        # Factor 3: Content preservation (check for important keywords)
        important_keywords = ['policy', 'coverage', 'exclusion', 'claim', 'premium', 'deductible', 'benefit']
        original_keywords = set()
        for keyword in important_keywords:
            if keyword.lower() in self.document_text.lower():
                original_keywords.add(keyword)
        
        preserved_keywords = set()
        for chunk in chunks:
            for keyword in important_keywords:
                if keyword.lower() in chunk.lower():
                    preserved_keywords.add(keyword)
        
        keyword_preservation = len(preserved_keywords) / len(original_keywords) if original_keywords else 1.0
        score += keyword_preservation * 0.4
        
        return min(score, 1.0)
    
    def run_all_tests(self) -> Dict[str, ChunkingResult]:
        """Run all available chunking strategies"""
        logger.info("Starting comprehensive chunking benchmark...")
        
        # Basic strategies
        strategies = [
            ("Fast Rule-Based", fast_rule_based_split),
            ("Smart Policy", smart_policy_split),
            ("Lightning Fast", lightning_fast_split),
            ("Smart Fast", smart_fast_split),
            ("Adaptive", adaptive_split),
            ("Contextual", contextual_split),
            ("Semantic", semantic_split),
        ]
        
        # Test each strategy
        for strategy_name, strategy_func in strategies:
            self.results[strategy_name] = self.test_chunking_strategy(strategy_name, strategy_func)
        
        # Test enhanced framework strategies if available
        if ENHANCED_AVAILABLE:
            logger.info("Testing enhanced chunking strategies...")
            
            enhanced_strategies = [
                ("Enhanced Adaptive", ChunkingStrategy.ADAPTIVE),
                ("Enhanced Hierarchical", ChunkingStrategy.HIERARCHICAL),
                ("Enhanced Semantic", ChunkingStrategy.SEMANTIC),
                ("Enhanced Token-Based", ChunkingStrategy.TOKEN_BASED),
                ("Enhanced Sentence-Based", ChunkingStrategy.SENTENCE_BASED),
                ("Enhanced Sliding Window", ChunkingStrategy.SLIDING_WINDOW),
                ("Enhanced Policy-Aware", ChunkingStrategy.POLICY_AWARE),
            ]
            
            for strategy_name, strategy_enum in enhanced_strategies:
                self.results[strategy_name] = self.test_enhanced_strategy(strategy_name, strategy_enum)
        
        return self.results
    
    def test_enhanced_strategy(self, strategy_name: str, strategy: ChunkingStrategy) -> ChunkingResult:
        """Test an enhanced chunking strategy"""
        logger.info(f"Testing {strategy_name}...")
        
        start_time = time.time()
        try:
            config = ChunkingConfig(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                strategy=strategy,
                timeout_seconds=15
            )
            
            framework = EnhancedChunkingFramework(config)
            result = framework.chunk_document(self.document_text, strategy)
            
            execution_time = time.time() - start_time
            chunks = result['chunks']
            
            if not chunks:
                raise ValueError("No chunks generated")
            
            # Calculate statistics
            chunk_lengths = [len(chunk) for chunk in chunks]
            avg_length = statistics.mean(chunk_lengths)
            std_length = statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
            
            # Use quality metrics from enhanced framework if available
            quality_score = result.get('quality_metrics', {}).get('content_preservation_score', 0.0)
            if quality_score == 0.0:
                quality_score = self._calculate_quality_score(chunks)
            
            return ChunkingResult(
                strategy_name=strategy_name,
                chunks=chunks,
                execution_time=execution_time,
                chunk_count=len(chunks),
                avg_chunk_length=avg_length,
                std_chunk_length=std_length,
                min_chunk_length=min(chunk_lengths),
                max_chunk_length=max(chunk_lengths),
                quality_score=quality_score
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{strategy_name} failed: {e}")
            
            return ChunkingResult(
                strategy_name=strategy_name,
                chunks=[],
                execution_time=execution_time,
                chunk_count=0,
                avg_chunk_length=0.0,
                std_chunk_length=0.0,
                min_chunk_length=0,
                max_chunk_length=0,
                error=str(e)
            )
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.results:
            return "No test results available"
        
        report = []
        report.append("🧪 COMPREHENSIVE CHUNKING PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"📄 Document: {len(self.document_text)} characters")
        report.append(f"🎯 Target chunk size: {self.chunk_size}")
        report.append(f"🔗 Chunk overlap: {self.chunk_overlap}")
        report.append("")
        
        # Sort results by execution time (fastest first)
        successful_results = {k: v for k, v in self.results.items() if not v.error}
        failed_results = {k: v for k, v in self.results.items() if v.error}
        
        if successful_results:
            sorted_results = sorted(successful_results.items(), key=lambda x: x[1].execution_time)
            
            report.append("📊 PERFORMANCE RANKING (by speed)")
            report.append("-" * 40)
            
            for i, (name, result) in enumerate(sorted_results):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                report.append(f"{rank} {name}: {result.execution_time:.3f}s ({result.chunk_count} chunks)")
            
            report.append("")
            report.append("📈 DETAILED RESULTS")
            report.append("-" * 40)
            
            for name, result in sorted_results:
                report.append(f"\n🔍 {name}")
                report.append(f"   ⏱️  Execution time: {result.execution_time:.3f}s")
                report.append(f"   📊 Chunk count: {result.chunk_count}")
                report.append(f"   📏 Avg length: {result.avg_chunk_length:.1f} chars")
                report.append(f"   📐 Std deviation: {result.std_chunk_length:.1f}")
                report.append(f"   📌 Range: {result.min_chunk_length} - {result.max_chunk_length} chars")
                report.append(f"   ⭐ Quality score: {result.quality_score:.3f}")
                
                # Calculate speed metrics
                chunks_per_second = result.chunk_count / result.execution_time if result.execution_time > 0 else 0
                chars_per_second = len(self.document_text) / result.execution_time if result.execution_time > 0 else 0
                
                report.append(f"   🚀 Speed: {chunks_per_second:.1f} chunks/s, {chars_per_second:.0f} chars/s")
        
        if failed_results:
            report.append("\n❌ FAILED STRATEGIES")
            report.append("-" * 40)
            for name, result in failed_results.items():
                report.append(f"   {name}: {result.error}")
        
        # Generate recommendations
        if successful_results:
            report.append("\n🎯 RECOMMENDATIONS")
            report.append("-" * 40)
            
            # Find best performers in different categories
            fastest = min(successful_results.items(), key=lambda x: x[1].execution_time)
            highest_quality = max(successful_results.items(), key=lambda x: x[1].quality_score)
            most_consistent = min(successful_results.items(), 
                                key=lambda x: x[1].std_chunk_length / x[1].avg_chunk_length if x[1].avg_chunk_length > 0 else float('inf'))
            
            report.append(f"🏃 Fastest: {fastest[0]} ({fastest[1].execution_time:.3f}s)")
            report.append(f"⭐ Highest quality: {highest_quality[0]} (score: {highest_quality[1].quality_score:.3f})")
            report.append(f"📏 Most consistent: {most_consistent[0]} (std: {most_consistent[1].std_chunk_length:.1f})")
            
            # Overall recommendation
            report.append("\n💡 Overall recommendation:")
            balanced_scores = {}
            for name, result in successful_results.items():
                # Balanced score: combine speed (inverse time) and quality
                speed_score = 1.0 / (result.execution_time + 0.001)  # Avoid division by zero
                balanced_score = (speed_score * 0.4) + (result.quality_score * 0.6)
                balanced_scores[name] = balanced_score
            
            best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
            report.append(f"   🏆 {best_balanced[0]} offers the best balance of speed and quality")
        
        return "\n".join(report)
    
    def save_detailed_results(self, filename: str = "chunking_benchmark_results.txt"):
        """Save detailed results to a file"""
        report = self.generate_report()
        
        # Add detailed chunk analysis
        report += "\n\n📝 DETAILED CHUNK ANALYSIS\n"
        report += "=" * 60 + "\n"
        
        for name, result in self.results.items():
            if result.error:
                continue
                
            report += f"\n🔍 {name} - Sample Chunks:\n"
            report += "-" * 30 + "\n"
            
            # Show first 3 chunks as samples
            for i, chunk in enumerate(result.chunks[:3]):
                report += f"Chunk {i+1} ({len(chunk)} chars):\n"
                report += chunk[:200] + ("..." if len(chunk) > 200 else "") + "\n"
                report += "-" * 20 + "\n"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Detailed results saved to {filename}")

def run_performance_test(pdf_url: str = None) -> Dict[str, ChunkingResult]:
    """Run the complete performance test suite"""
    if pdf_url is None:
        pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    benchmark = ChunkingBenchmark(pdf_url)
    
    try:
        # Load document
        benchmark.load_document()
        
        # Run all tests
        results = benchmark.run_all_tests()
        
        # Generate and display report
        report = benchmark.generate_report()
        print(report)
        
        # Save detailed results
        benchmark.save_detailed_results()
        
        return results
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        print(f"❌ Performance test failed: {e}")
        return {}

if __name__ == "__main__":
    print("🧪 STARTING COMPREHENSIVE CHUNKING PERFORMANCE TEST")
    print("=" * 60)
    
    results = run_performance_test()
    
    if results:
        print("\n🎉 Performance test completed successfully!")
        print(f"📊 Tested {len(results)} chunking strategies")
        
        successful_count = sum(1 for r in results.values() if not r.error)
        failed_count = len(results) - successful_count
        
        print(f"✅ Successful: {successful_count}")
        print(f"❌ Failed: {failed_count}")
    else:
        print("\n❌ Performance test failed to complete")
