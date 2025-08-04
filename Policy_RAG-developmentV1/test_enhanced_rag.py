"""
Test script for enhanced RAG retrieval system
Tests various query types and advanced features
"""

import requests
import json
import time
from typing import Dict, List

class EnhancedRAGTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_initialization(self) -> bool:
        """Test if the RAG system initializes properly"""
        print("🔧 Testing system initialization...")
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ System initialized: {result.get('system_ready', False)}")
                print(f"📊 Chunks created: {result.get('chunks_count', 0)}")
                print(f"🧠 Vector store: {result.get('vector_store_type', 'Unknown')}")
                print(f"🎯 Advanced features: {result.get('cosine_similarity', False)}")
                return result.get('system_ready', False)
            else:
                print(f"❌ Initialization failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return False
    
    def test_query_types(self) -> Dict[str, bool]:
        """Test different types of queries to verify intent recognition"""
        test_queries = {
            "definition": "What is a grace period?",
            "procedure": "How to file a claim?",
            "eligibility": "Who is eligible for this policy?", 
            "amount": "How much is the premium?",
            "time": "When does the grace period start?",
            "coverage": "What is covered under this policy?",
            "exclusion": "What is not covered by this insurance?",
            "general": "Tell me about this policy"
        }
        
        results = {}
        print(f"\n🧪 Testing {len(test_queries)} different query types...")
        
        for query_type, question in test_queries.items():
            print(f"\n📝 Testing {query_type}: {question}")
            
            try:
                # Test standard query
                start_time = time.time()
                response = requests.post(f"{self.base_url}/query", json={
                    "question": question,
                    "k": 5,
                    "use_advanced": True
                })
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    success = result.get('success', False)
                    
                    print(f"   ✅ Success: {success}")
                    print(f"   🕐 Time: {processing_time:.3f}s")
                    print(f"   📄 Chunks: {result.get('num_chunks_retrieved', 0)}")
                    print(f"   📊 Mode: {result.get('retrieval_mode', 'Unknown')}")
                    
                    if 'intent' in result:
                        print(f"   🎯 Intent: {result['intent']}")
                    if 'confidence' in result:
                        print(f"   🎖️ Confidence: {result['confidence']:.2f}")
                    
                    print(f"   💬 Answer: {result.get('answer', 'No answer')[:100]}...")
                    
                    results[query_type] = success
                else:
                    print(f"   ❌ Failed: {response.status_code}")
                    results[query_type] = False
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[query_type] = False
        
        return results
    
    def test_advanced_features(self) -> Dict[str, bool]:
        """Test advanced retrieval features"""
        print(f"\n🚀 Testing advanced features...")
        
        advanced_tests = {}
        
        # Test 1: Advanced query endpoint
        try:
            print(f"\n🔬 Testing advanced query endpoint...")
            response = requests.post(f"{self.base_url}/query/advanced", json={
                "question": "What is the grace period for premium payment?",
                "k": 7,
                "similarity_threshold": 0.3,
                "explain_retrieval": True
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Advanced query successful")
                print(f"   🎯 Intent: {result.get('intent', 'Unknown')}")
                print(f"   🎖️ Confidence: {result.get('confidence', 0):.2f}")
                
                if 'retrieval_explanation' in result:
                    print(f"   📝 Explanation available: Yes")
                if 'chunk_scores' in result:
                    print(f"   📊 Chunk scores: {len(result['chunk_scores'])} chunks")
                
                advanced_tests['advanced_endpoint'] = True
            else:
                print(f"   ❌ Advanced query failed: {response.status_code}")
                advanced_tests['advanced_endpoint'] = False
                
        except Exception as e:
            print(f"   ❌ Advanced query error: {e}")
            advanced_tests['advanced_endpoint'] = False
        
        # Test 2: Context-aware retrieval
        try:
            print(f"\n🧠 Testing context-aware retrieval...")
            response = requests.post(f"{self.base_url}/query/advanced", json={
                "question": "How much do I need to pay?",
                "conversation_history": [
                    "What is this policy about?",
                    "What is the premium payment process?"
                ],
                "similarity_threshold": 0.2
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Context-aware query successful")
                print(f"   📄 Chunks retrieved: {result.get('num_chunks_retrieved', 0)}")
                print(f"   🎯 Intent detected: {result.get('intent', 'Unknown')}")
                advanced_tests['context_aware'] = True
            else:
                print(f"   ❌ Context-aware query failed: {response.status_code}")
                advanced_tests['context_aware'] = False
                
        except Exception as e:
            print(f"   ❌ Context-aware error: {e}")
            advanced_tests['context_aware'] = False
        
        # Test 3: Similarity threshold filtering
        try:
            print(f"\n🎚️ Testing similarity threshold filtering...")
            
            # Test with high threshold
            response_high = requests.post(f"{self.base_url}/query", json={
                "question": "What is covered?",
                "similarity_threshold": 0.8,
                "use_advanced": True
            })
            
            # Test with low threshold  
            response_low = requests.post(f"{self.base_url}/query", json={
                "question": "What is covered?",
                "similarity_threshold": 0.1,
                "use_advanced": True
            })
            
            if response_high.status_code == 200 and response_low.status_code == 200:
                high_chunks = response_high.json().get('num_chunks_retrieved', 0)
                low_chunks = response_low.json().get('num_chunks_retrieved', 0)
                
                print(f"   📊 High threshold (0.8): {high_chunks} chunks")
                print(f"   📊 Low threshold (0.1): {low_chunks} chunks")
                
                # Low threshold should generally return more chunks
                advanced_tests['threshold_filtering'] = low_chunks >= high_chunks
                print(f"   ✅ Threshold filtering works: {advanced_tests['threshold_filtering']}")
            else:
                print(f"   ❌ Threshold testing failed")
                advanced_tests['threshold_filtering'] = False
                
        except Exception as e:
            print(f"   ❌ Threshold testing error: {e}")
            advanced_tests['threshold_filtering'] = False
        
        return advanced_tests
    
    def test_performance(self) -> Dict[str, float]:
        """Test system performance with various query complexities"""
        print(f"\n⚡ Testing performance...")
        
        performance_tests = [
            ("simple", "What is premium?"),
            ("medium", "How to file a claim for medical expenses under this policy?"),
            ("complex", "What are the eligibility criteria, coverage details, and exclusions for the National Parivar Mediclaim Plus Policy grace period?")
        ]
        
        performance_results = {}
        
        for complexity, question in performance_tests:
            print(f"\n📊 Testing {complexity} query...")
            times = []
            
            # Run multiple times for average
            for i in range(3):
                try:
                    start_time = time.time()
                    response = requests.post(f"{self.base_url}/query", json={
                        "question": question,
                        "use_advanced": True
                    })
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        times.append(end_time - start_time)
                    
                except Exception as e:
                    print(f"   ❌ Performance test error: {e}")
            
            if times:
                avg_time = sum(times) / len(times)
                performance_results[complexity] = avg_time
                print(f"   ⏱️ Average time: {avg_time:.3f}s")
            else:
                performance_results[complexity] = float('inf')
                print(f"   ❌ Performance test failed")
        
        return performance_results
    
    def run_comprehensive_test(self):
        """Run all tests and provide a comprehensive report"""
        print("🧪 Starting Enhanced RAG System Comprehensive Test")
        print("=" * 60)
        
        # Test 1: System initialization
        init_success = self.test_initialization()
        
        if not init_success:
            print("❌ System initialization failed. Cannot proceed with tests.")
            return
        
        # Test 2: Query types
        query_results = self.test_query_types()
        
        # Test 3: Advanced features
        advanced_results = self.test_advanced_features()
        
        # Test 4: Performance
        performance_results = self.test_performance()
        
        # Generate report
        print("\n" + "=" * 60)
        print("📋 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\n🎯 QUERY TYPE TESTS:")
        for query_type, success in query_results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {query_type.capitalize()}: {success}")
        
        query_success_rate = sum(query_results.values()) / len(query_results) * 100
        print(f"   📊 Overall success rate: {query_success_rate:.1f}%")
        
        print(f"\n🚀 ADVANCED FEATURES:")
        for feature, success in advanced_results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {feature.replace('_', ' ').title()}: {success}")
        
        print(f"\n⚡ PERFORMANCE:")
        for complexity, avg_time in performance_results.items():
            if avg_time != float('inf'):
                print(f"   📊 {complexity.capitalize()} queries: {avg_time:.3f}s average")
            else:
                print(f"   ❌ {complexity.capitalize()} queries: Failed")
        
        # Overall assessment
        overall_success = (
            init_success and 
            query_success_rate >= 80 and 
            any(advanced_results.values()) and
            any(t != float('inf') for t in performance_results.values())
        )
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        status = "✅ PASSED" if overall_success else "❌ NEEDS IMPROVEMENT"
        print(f"   {status}")
        
        if overall_success:
            print("🎉 Enhanced RAG system is working well!")
            print("💡 System is ready for production use.")
        else:
            print("⚠️ Some issues detected. Please review the test results.")
            print("🔧 Consider checking dependencies and configuration.")
        
        return overall_success


if __name__ == "__main__":
    # Run the comprehensive test
    tester = EnhancedRAGTester()
    success = tester.run_comprehensive_test()
    
    print(f"\n🏁 Test completed. Success: {success}")
    exit(0 if success else 1)
