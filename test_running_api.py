#!/usr/bin/env python3
"""
Test the running Data Analyst Agent API
"""

import requests
import json
import time

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to API: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to root endpoint: {e}")
        return False

def test_simple_question():
    """Test with a simple question"""
    try:
        # Test the test endpoint with a simple question
        simple_question = "What is 2 + 2? Please respond with just the number."
        
        response = requests.post(
            "http://127.0.0.1:8000/api/test",
            params={"question": simple_question},
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Simple question test passed")
            print(f"   Question: {simple_question}")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Simple question test failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Simple question test error: {e}")
        return False

def main():
    """Run API tests"""
    print("🧪 Testing Data Analyst Agent API")
    print("🔗 Assuming server is running at http://127.0.0.1:8000")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
        ("Simple Question", test_simple_question)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your API is working correctly.")
        print("\n📝 You can now:")
        print("1. Visit http://127.0.0.1:8000/docs for interactive API documentation")
        print("2. Visit http://127.0.0.1:8000/health for health check")
        print("3. Use the API for data analysis tasks")
    else:
        print(f"\n⚠️ {len(tests) - passed} tests failed.")
        print("Make sure the server is running: python start_server.py")

if __name__ == "__main__":
    main()
