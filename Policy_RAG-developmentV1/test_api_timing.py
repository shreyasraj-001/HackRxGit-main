"""
Test script to benchmark the optimized parallel API
"""

import requests
import time
import json

def test_parallel_api():
    """Test the optimized parallel API"""
    
    # Test data
    test_url = "http://127.0.0.1:8000/hackrx/run"
    test_payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the maximum sum insured available under this policy?",
            "What are the exclusions for mental illness under this policy?",
            "Is there a co-payment clause in this policy?",
            "What is the room rent limit under this policy?",
            "What is the claim settlement process for this policy?"
        ]
    }
    
    print("🚀 TESTING OPTIMIZED PARALLEL API")
    print("="*60)
    print(f"📋 Testing with {len(test_payload['questions'])} questions")
    print(f"🎯 Expected improvement: Much faster LLM processing")
    print(f"📄 Document: {test_payload['documents'][:50]}...")
    
    # Make API request
    print("\n⚡ Making parallel API request...")
    api_start = time.time()
    
    try:
        response = requests.post(
            test_url, 
            json=test_payload, 
            headers={"Content-Type": "application/json"},
            timeout=120  # Increased timeout for parallel processing
        )
        
        api_total = time.time() - api_start
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            
            print(f"✅ Parallel API request completed in {api_total:.3f}s")
            print(f"📊 Response: {len(answers)} answers received")
            print(f"⚡ Questions per second: {len(test_payload['questions'])/api_total:.2f}")
            print(f"🎯 Average time per question: {api_total/len(test_payload['questions']):.3f}s")
            
            # Show sample answers
            print(f"\n📝 SAMPLE ANSWERS:")
            print("-" * 60)
            for i, (question, answer) in enumerate(zip(test_payload['questions'][:3], answers[:3])):
                print(f"\n❓ Q{i+1}: {question}")
                print(f"✅ A{i+1}: {answer[:150]}..." if len(answer) > 150 else f"✅ A{i+1}: {answer}")
            
            if len(answers) > 3:
                print(f"\n... and {len(answers) - 3} more answers")
            
            # Performance analysis
            print(f"\n📈 PERFORMANCE ANALYSIS:")
            print("-" * 40)
            if api_total < 20:
                print("🎉 EXCELLENT: Sub-20 second processing!")
            elif api_total < 30:
                print("✅ GOOD: Significant improvement achieved!")
            else:
                print("⚠️ NEEDS OPTIMIZATION: Still room for improvement")
                
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.Timeout:
        api_total = time.time() - api_start
        print(f"⏰ API request timed out after {api_total:.3f}s")
    except Exception as e:
        api_total = time.time() - api_start
        print(f"❌ API request failed after {api_total:.3f}s: {e}")

def test_smaller_parallel_batch():
    """Test with smaller batch to verify parallel processing works"""
    
    test_url = "http://127.0.0.1:8000/hackrx/run"
    test_payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?"
        ]
    }
    
    print("\n\n🧪 TESTING PARALLEL PROCESSING (3 questions)")
    print("="*60)
    
    api_start = time.time()
    
    try:
        response = requests.post(
            test_url, 
            json=test_payload, 
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        api_total = time.time() - api_start
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            
            print(f"✅ Small parallel batch completed in {api_total:.3f}s")
            print(f"📊 Response: {len(answers)} answers received")
            print(f"⚡ Questions per second: {len(test_payload['questions'])/api_total:.2f}")
            print(f"🎯 Average time per question: {api_total/len(test_payload['questions']):.3f}s")
            
            # Detailed answers for small batch
            print(f"\n📝 ALL ANSWERS:")
            print("-" * 40)
            for i, (question, answer) in enumerate(zip(test_payload['questions'], answers)):
                print(f"\n❓ Q{i+1}: {question}")
                print(f"✅ A{i+1}: {answer}")
            
        else:
            print(f"❌ Small batch failed with status {response.status_code}")
            
    except Exception as e:
        api_total = time.time() - api_start
        print(f"❌ Small batch failed after {api_total:.3f}s: {e}")

def compare_with_previous():
    """Show expected improvement"""
    print("\n📊 EXPECTED IMPROVEMENT COMPARISON")
    print("="*60)
    print("Previous Performance (Sequential):")
    print("   📥 PDF Download/Parse: ~10.3s (38%)")
    print("   ✂️ Chunking: ~0.0s (0%)")
    print("   📝 Context Prep: ~0.0s (0%)")
    print("   🤖 LLM Processing: ~16.6s (61%)")
    print("   ✅ Response Validation: ~0.0s (0%)")
    print("   🔄 TOTAL: ~27s")
    print()
    print("Expected Performance (Parallel):")
    print("   📥 PDF Download/Parse: ~3-5s (optimized)")
    print("   ✂️ Semantic Chunking: ~0.0s")
    print("   📝 Context Prep: ~0.0s")
    print("   🚀 Parallel LLM Processing: ~3-5s (10x faster!)")
    print("   ✅ Response Validation: ~0.0s")
    print("   🔄 EXPECTED TOTAL: ~6-10s")
    print()
    print("🎯 Target: 60-80% time reduction!")

if __name__ == "__main__":
    print("� PARALLEL API PERFORMANCE BENCHMARK")
    print("="*60)
    print("⚠️  Make sure the server is running on http://127.0.0.1:8000")
    print("   Run: python main.py or uvicorn main:app --reload")
    print("="*60)
    
    # Show expected improvement
    compare_with_previous()
    
    # Test with small batch first
    test_smaller_parallel_batch()
    
    # Test with full batch
    test_parallel_api()
    
    print("\n🏁 PARALLEL BENCHMARK COMPLETE")
    print("Check the server logs for detailed timing breakdown!")
    print("🎯 Look for 'Parallel processing complete' and 'Questions per second' metrics!")
