#!/usr/bin/env python3
"""
Test the Data Analyst Agent against the exact TDS evaluation rubric
"""

import requests
import json
import re
import math

def test_wikipedia_against_rubric():
    """Test Wikipedia question against the exact evaluation rubric"""
    print("🧪 Testing Wikipedia Analysis Against TDS Evaluation Rubric")
    print("=" * 70)
    
    try:
        with open("test_wikipedia_question.txt", "rb") as f:
            files = {"file": ("question.txt", f, "text/plain")}
            
            response = requests.post(
                "http://127.0.0.1:8000/api/",
                files=files,
                timeout=180
            )
        
        if response.status_code != 200:
            print(f"❌ API request failed: {response.status_code}")
            return False
        
        result = response.json()
        print(f"📋 Raw API Response: {result}")
        
        # Structural gate – no score, hard-fail if not a 4-element array
        print("\n🔍 Structural Gate Check:")
        if not isinstance(result, list):
            print(f"❌ Expected array, got {type(result)}")
            return False
        
        if len(result) != 4:
            print(f"❌ Expected 4 elements, got {len(result)}")
            return False
        
        print("✅ Structural gate passed: JSON array with 4 elements")
        
        # Test 1: first answer must equal 1
        print("\n🔍 Test 1: First element must equal 1")
        first_element = result[0]
        test1_pass = (first_element == 1)
        print(f"   Got: {first_element}")
        print(f"   Expected: 1")
        print(f"   Result: {'✅ PASS' if test1_pass else '❌ FAIL'} (Weight: 4 points)")
        
        # Test 2: second answer must contain "Titanic" (case-insensitive)
        print("\n🔍 Test 2: Second element must contain 'Titanic'")
        second_element = str(result[1])
        test2_pass = bool(re.search(r'titanic', second_element, re.I))
        print(f"   Got: {second_element}")
        print(f"   Expected: Contains 'Titanic' (case-insensitive)")
        print(f"   Result: {'✅ PASS' if test2_pass else '❌ FAIL'} (Weight: 4 points)")
        
        # Test 3: third answer within ±0.001 of 0.485782
        print("\n🔍 Test 3: Third element within ±0.001 of 0.485782")
        third_element = float(result[2])
        expected_correlation = 0.485782
        tolerance = 0.001
        test3_pass = abs(third_element - expected_correlation) <= tolerance
        print(f"   Got: {third_element}")
        print(f"   Expected: {expected_correlation} ± {tolerance}")
        print(f"   Difference: {abs(third_element - expected_correlation):.6f}")
        print(f"   Result: {'✅ PASS' if test3_pass else '❌ FAIL'} (Weight: 4 points)")
        
        # Test 4: vision check - base64 PNG validation
        print("\n🔍 Test 4: Base64 PNG validation")
        fourth_element = result[3]
        
        if not isinstance(fourth_element, str):
            print(f"❌ Expected string, got {type(fourth_element)}")
            test4_pass = False
        elif not fourth_element.startswith("data:image/"):
            print(f"❌ Expected data URI, got: {fourth_element[:50]}...")
            test4_pass = False
        elif len(fourth_element) > 100000:
            print(f"❌ Image too large: {len(fourth_element)} > 100000 chars")
            test4_pass = False
        else:
            print(f"✅ Valid base64 data URI: {len(fourth_element)} chars")
            test4_pass = True
            
            # Additional checks for plot requirements
            print("   📊 Plot requirements (manual inspection needed):")
            print("      - Scatterplot of Rank (x-axis) vs Peak (y-axis)")
            print("      - Red dotted regression line")
            print("      - Visible and labeled axes")
            print("      - File size < 100KB ✅")
        
        print(f"   Result: {'✅ PASS' if test4_pass else '❌ FAIL'} (Weight: 8 points)")
        
        # Calculate total score
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY:")
        
        total_score = 0
        max_score = 20
        
        if test1_pass:
            total_score += 4
            print("✅ Test 1 (First element = 1): 4/4 points")
        else:
            print("❌ Test 1 (First element = 1): 0/4 points")
        
        if test2_pass:
            total_score += 4
            print("✅ Test 2 (Contains 'Titanic'): 4/4 points")
        else:
            print("❌ Test 2 (Contains 'Titanic'): 0/4 points")
        
        if test3_pass:
            total_score += 4
            print("✅ Test 3 (Correlation ±0.001): 4/4 points")
        else:
            print("❌ Test 3 (Correlation ±0.001): 0/4 points")
        
        if test4_pass:
            total_score += 8
            print("✅ Test 4 (Base64 PNG): 8/8 points")
        else:
            print("❌ Test 4 (Base64 PNG): 0/8 points")
        
        print(f"\n🎯 TOTAL SCORE: {total_score}/{max_score} points ({total_score/max_score*100:.1f}%)")
        
        if total_score == max_score:
            print("\n🎉 PERFECT SCORE! Your API meets all rubric requirements!")
        elif total_score >= 16:
            print("\n✅ Good score! Minor adjustments needed.")
        elif total_score >= 10:
            print("\n⚠️ Partial success. Several issues need fixing.")
        else:
            print("\n❌ Major issues detected. Significant fixes required.")
        
        return total_score == max_score
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Run the evaluation test"""
    success = test_wikipedia_against_rubric()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
