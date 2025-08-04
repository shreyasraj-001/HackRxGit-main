"""
Test script for the new batch RAG endpoint
"""

import requests
import json
import time

# API endpoint
BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/hackrx/run"

# Sample data
test_data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?"
    ]
}

def test_batch_endpoint():
    """Test the batch RAG endpoint"""
    print("🧪 Testing batch RAG endpoint...")
    print(f"📍 Endpoint: {ENDPOINT}")
    print(f"📄 Document: {test_data['documents'][:50]}...")
    print(f"❓ Questions: {len(test_data['questions'])}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Make the request
        response = requests.post(
            ENDPOINT,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minute timeout
        )
        
        # Record end time
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            
            print(f"✅ Success! Received {len(answers)} answers")
            print("📋 Results:")
            print("=" * 80)
            
            for i, (question, answer) in enumerate(zip(test_data["questions"], answers), 1):
                print(f"\n🔸 Question {i}:")
                print(f"   {question}")
                print(f"🔹 Answer {i}:")
                print(f"   {answer}")
                print("-" * 60)
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - make sure the server is running")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_server_health():
    """Test if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            print("✅ Server is running and RAG system is initialized")
            return True
        else:
            print(f"⚠️ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error checking server health: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting batch RAG endpoint test")
    print("=" * 50)
    
    # Check server health first
    if test_server_health():
        print("\n" + "=" * 50)
        test_batch_endpoint()
    else:
        print("\n💡 To start the server, run: python main.py")
        print("   Or: uvicorn main:app --port 8000 --reload")
