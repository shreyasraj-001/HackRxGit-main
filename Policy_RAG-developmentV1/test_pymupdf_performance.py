#!/usr/bin/env python3
"""
Performance test script to compare PyMuPDF vs pdfplumber for PDF parsing speed
"""

import time
import requests
import fitz  # PyMuPDF
import pdfplumber
import io

def test_pymupdf_parsing():
    """Test PDF parsing speed with PyMuPDF (new implementation)"""
    print("🚀 Testing PyMuPDF parsing speed...")
    
    # Policy document URL
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    start_time = time.time()
    
    # Download PDF
    download_start = time.time()
    response = requests.get(
        pdf_url, 
        timeout=10,
        stream=True,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    )
    response.raise_for_status()
    download_time = time.time() - download_start
    
    # Parse with PyMuPDF
    parse_start = time.time()
    pdf_document = fitz.open(stream=response.content, filetype="pdf")
    text_parts = []
    
    total_pages = len(pdf_document)
    print(f"📄 Processing {total_pages} pages with PyMuPDF...")
    
    for page_num in range(total_pages):
        page = pdf_document[page_num]
        page_text = page.get_text()
        
        if page_text and page_text.strip():
            text_parts.append(page_text)
    
    parsed_data = '\n'.join(text_parts)
    pdf_document.close()
    parse_time = time.time() - parse_start
    
    total_time = time.time() - start_time
    
    print(f"✅ PyMuPDF Results:")
    print(f"   📥 Download time: {download_time:.3f}s")
    print(f"   📄 Parse time: {parse_time:.3f}s")
    print(f"   🔄 Total time: {total_time:.3f}s")
    print(f"   📊 Text extracted: {len(parsed_data)} characters")
    print(f"   📃 Pages processed: {len(text_parts)}/{total_pages}")
    
    return {
        "method": "PyMuPDF",
        "download_time": download_time,
        "parse_time": parse_time,
        "total_time": total_time,
        "text_length": len(parsed_data),
        "pages_processed": len(text_parts),
        "total_pages": total_pages
    }

def test_pdfplumber_parsing():
    """Test PDF parsing speed with pdfplumber (old implementation)"""
    print("\n🐌 Testing pdfplumber parsing speed...")
    
    # Policy document URL
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    start_time = time.time()
    
    # Download PDF
    download_start = time.time()
    response = requests.get(
        pdf_url, 
        timeout=10,
        stream=True,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    )
    response.raise_for_status()
    download_time = time.time() - download_start
    
    # Parse with pdfplumber
    parse_start = time.time()
    text_parts = []
    
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        total_pages = len(pdf.pages)
        print(f"📄 Processing {total_pages} pages with pdfplumber...")
        
        for page_num, page in enumerate(pdf.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                print(f"⚠️ Warning: Failed to extract text from page {page_num + 1}: {e}")
                continue
    
    parsed_data = '\n'.join(text_parts)
    parse_time = time.time() - parse_start
    
    total_time = time.time() - start_time
    
    print(f"✅ pdfplumber Results:")
    print(f"   📥 Download time: {download_time:.3f}s")
    print(f"   📄 Parse time: {parse_time:.3f}s")
    print(f"   🔄 Total time: {total_time:.3f}s")
    print(f"   📊 Text extracted: {len(parsed_data)} characters")
    print(f"   📃 Pages processed: {len(text_parts)}/{total_pages}")
    
    return {
        "method": "pdfplumber",
        "download_time": download_time,
        "parse_time": parse_time,
        "total_time": total_time,
        "text_length": len(parsed_data),
        "pages_processed": len(text_parts),
        "total_pages": total_pages
    }

def compare_results(pymupdf_result, pdfplumber_result):
    """Compare the performance results"""
    print("\n📊 PERFORMANCE COMPARISON:")
    print("=" * 50)
    
    # Parse time comparison
    parse_speedup = pdfplumber_result["parse_time"] / pymupdf_result["parse_time"]
    print(f"📄 Parse Time:")
    print(f"   PyMuPDF:    {pymupdf_result['parse_time']:.3f}s")
    print(f"   pdfplumber: {pdfplumber_result['parse_time']:.3f}s")
    print(f"   🚀 Speedup: {parse_speedup:.2f}x faster with PyMuPDF")
    
    # Total time comparison
    total_speedup = pdfplumber_result["total_time"] / pymupdf_result["total_time"]
    print(f"\n🔄 Total Time:")
    print(f"   PyMuPDF:    {pymupdf_result['total_time']:.3f}s")
    print(f"   pdfplumber: {pdfplumber_result['total_time']:.3f}s")
    print(f"   🚀 Speedup: {total_speedup:.2f}x faster with PyMuPDF")
    
    # Text extraction comparison
    text_diff = abs(pymupdf_result["text_length"] - pdfplumber_result["text_length"])
    text_diff_percent = (text_diff / max(pymupdf_result["text_length"], pdfplumber_result["text_length"])) * 100
    print(f"\n📊 Text Extraction:")
    print(f"   PyMuPDF:    {pymupdf_result['text_length']:,} characters")
    print(f"   pdfplumber: {pdfplumber_result['text_length']:,} characters")
    print(f"   📏 Difference: {text_diff:,} characters ({text_diff_percent:.1f}%)")
    
    # Pages processed
    print(f"\n📃 Pages Processed:")
    print(f"   PyMuPDF:    {pymupdf_result['pages_processed']}/{pymupdf_result['total_pages']}")
    print(f"   pdfplumber: {pdfplumber_result['pages_processed']}/{pdfplumber_result['total_pages']}")
    
    # Time savings calculation
    time_saved = pdfplumber_result["total_time"] - pymupdf_result["total_time"]
    print(f"\n⏱️ Time Savings:")
    print(f"   Absolute: {time_saved:.3f}s saved per document")
    print(f"   Relative: {((pdfplumber_result['total_time'] - pymupdf_result['total_time']) / pdfplumber_result['total_time'] * 100):.1f}% faster")

if __name__ == "__main__":
    print("🧪 PDF Parsing Performance Benchmark")
    print("=" * 50)
    
    try:
        # Test PyMuPDF (new implementation)
        pymupdf_result = test_pymupdf_parsing()
        
        # Test pdfplumber (old implementation)  
        pdfplumber_result = test_pdfplumber_parsing()
        
        # Compare results
        compare_results(pymupdf_result, pdfplumber_result)
        
        print("\n🎉 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
