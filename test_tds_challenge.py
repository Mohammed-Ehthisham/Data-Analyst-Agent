#!/usr/bin/env python3
"""
Test the Data Analyst Agent with actual TDS challenge questions
"""

import requests
import json
import time
import sys

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy and running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure the server is running: python start_server.py")
        return False

def test_wikipedia_question():
    """Test the Wikipedia film analysis question"""
    print("\n🎬 Testing Wikipedia Film Analysis...")
    print("📋 Question: Analyze highest grossing films from Wikipedia")
    
    try:
        with open("test_wikipedia_question.txt", "rb") as f:
            files = {"file": ("question.txt", f, "text/plain")}
            
            response = requests.post(
                "http://127.0.0.1:8000/api/",
                files=files,
                timeout=180  # 3 minutes as specified
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Wikipedia question processed successfully!")
            
            # Validate response format
            if isinstance(result, list) and len(result) == 4:
                print("✅ Response format correct (JSON array with 4 elements)")
                
                # Check individual answers
                print(f"   1. $2bn movies before 2000: {result[0]}")
                print(f"   2. Earliest $1.5bn film: {result[1]}")
                print(f"   3. Rank-Peak correlation: {result[2]}")
                
                if isinstance(result[3], str) and result[3].startswith("data:image/"):
                    print("✅ Base64 image generated successfully")
                    print(f"   4. Plot size: {len(result[3])} characters")
                    
                    if len(result[3]) <= 100000:
                        print("✅ Image size within 100KB limit")
                    else:
                        print("⚠️ Image size exceeds 100KB limit")
                else:
                    print("❌ Invalid base64 image format")
                
                return True
            else:
                print(f"❌ Invalid response format. Expected array with 4 elements, got: {type(result)}")
                print(f"Response preview: {str(result)[:200]}...")
                return False
                
        else:
            print(f"❌ Wikipedia test failed with status {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return False
            
    except FileNotFoundError:
        print("❌ Test file 'test_wikipedia_question.txt' not found")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>3 minutes)")
        return False
    except Exception as e:
        print(f"❌ Wikipedia test error: {e}")
        return False

def test_database_question():
    """Test the database analysis question"""
    print("\n⚖️ Testing Database Analysis...")
    print("📋 Question: Analyze Indian High Court judgments dataset")
    
    try:
        with open("test_database_question.txt", "rb") as f:
            files = {"file": ("question.txt", f, "text/plain")}
            
            response = requests.post(
                "http://127.0.0.1:8000/api/",
                files=files,
                timeout=180  # 3 minutes
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Database question processed successfully!")
            
            # Validate response format
            if isinstance(result, dict):
                print("✅ Response format correct (JSON object)")
                
                expected_keys = [
                    "Which high court disposed the most cases from 2019 - 2022?",
                    "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?",
                    "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters"
                ]
                
                for key in expected_keys:
                    if key in result:
                        print(f"✅ Found answer for: {key[:50]}...")
                        if "base64" in key.lower() and isinstance(result[key], str):
                            if result[key].startswith("data:image/"):
                                print(f"   📊 Plot generated ({len(result[key])} chars)")
                            else:
                                print(f"   📊 Answer: {result[key]}")
                        else:
                            print(f"   📊 Answer: {result[key]}")
                    else:
                        print(f"❌ Missing answer for: {key[:50]}...")
                
                return True
            else:
                print(f"❌ Invalid response format. Expected JSON object, got: {type(result)}")
                return False
                
        else:
            print(f"❌ Database test failed with status {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return False
            
    except FileNotFoundError:
        print("❌ Test file 'test_database_question.txt' not found")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>3 minutes)")
        return False
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def test_simple_question():
    """Test with a simple question to verify basic functionality"""
    print("\n🧮 Testing Simple Question...")
    
    simple_question = "Calculate the mean of these numbers: [1, 2, 3, 4, 5]. Respond with just the number."
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/test",
            params={"question": simple_question},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Simple question processed successfully!")
            print(f"   📊 Response: {result}")
            return True
        else:
            print(f"❌ Simple question failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Simple question error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Data Analyst Agent - TDS Challenge Questions")
    print("=" * 70)
    print("🎯 This will test the actual sample questions from the TDS challenge")
    print("⏱️ Each test may take up to 3 minutes (as per requirements)")
    print("=" * 70)
    
    # Check if API is running
    if not test_api_health():
        return 1
    
    tests = [
        ("Simple Question Test", test_simple_question),
        ("Wikipedia Film Analysis", test_wikipedia_question),
        ("Database Analysis", test_database_question)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        start_time = time.time()
        
        if test_func():
            passed += 1
            elapsed = time.time() - start_time
            print(f"✅ {test_name} completed in {elapsed:.1f} seconds")
        else:
            elapsed = time.time() - start_time
            print(f"❌ {test_name} failed after {elapsed:.1f} seconds")
    
    print("\n" + "=" * 70)
    print(f"📊 Final Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Data Analyst Agent is ready for the TDS challenge!")
        print("\n📝 Next steps:")
        print("1. Deploy to a public URL (Railway, Render, Google Cloud Run)")
        print("2. Test the deployed API with these same questions")
        print("3. Submit your GitHub repo and API endpoint URLs")
    elif passed > 0:
        print(f"\n⚠️ {total - passed} tests failed, but {passed} passed.")
        print("Check the errors above and fix any issues.")
    else:
        print("\n❌ All tests failed. Check your setup and try again.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
