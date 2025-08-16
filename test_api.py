#!/usr/bin/env python3
"""
Quick test script for the Data Analyst Agent
"""

import asyncio
import requests
import json
import sys
import os

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_sample_question():
    """Test with a sample question"""
    sample_question = "Analyze sample data: [1, 2, 3, 4, 5]. Calculate mean and create a simple visualization."
    
    try:
        response = requests.post(
            "http://localhost:8000/api/test",
            params={"question": sample_question},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Sample question test passed")
            print(f"Response: {json.dumps(result, indent=2)[:200]}...")
            return True
        else:
            print(f"❌ Sample question test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Sample question test error: {e}")
        return False

def test_file_upload():
    """Test file upload functionality"""
    # Create a temporary test file
    test_content = "What is 2 + 2? Provide the answer as a JSON array."
    
    try:
        with open("temp_question.txt", "w") as f:
            f.write(test_content)
        
        with open("temp_question.txt", "rb") as f:
            files = {"file": ("question.txt", f, "text/plain")}
            response = requests.post(
                "http://localhost:8000/api/",
                files=files,
                timeout=60
            )
        
        # Clean up
        os.remove("temp_question.txt")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ File upload test passed")
            print(f"Response: {json.dumps(result, indent=2)[:200]}...")
            return True
        else:
            print(f"❌ File upload test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ File upload test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Data Analyst Agent API...")
    print("=" * 50)
    
    # Check if API is running
    if not test_api_health():
        print("\n💡 Make sure the API is running:")
        print("   uvicorn main:app --reload")
        print("   or")
        print("   docker-compose up")
        sys.exit(1)
    
    print()
    
    # Run tests
    tests = [
        ("Sample Question Test", test_sample_question),
        ("File Upload Test", test_file_upload),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
