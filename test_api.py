#!/usr/bin/env python3
"""Test the HouseGPT API endpoints."""

import requests
import json
import time

def test_api():
    """Test the HouseGPT API."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing HouseGPT API...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Root Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return
    
    # Test chat endpoint
    print("\n3. Testing chat endpoint...")
    test_messages = [
        "Hello House, how are you?",
        "I think someone is lying to me",
        "What do you think about people in general?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Chat Test {i} ---")
        print(f"User: {message}")
        
        try:
            chat_data = {
                "message": message,
                "user_id": "test_user"
            }
            
            response = requests.post(
                f"{base_url}/chat",
                json=chat_data,
                headers={"Content-Type": "application/json"},
                timeout=60  # House model can take time
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"🏥 House: {result['response']}")
                print(f"📊 Confidence: {result['confidence']:.2f}")
                print(f"😏 Sarcasm Level: {result['sarcasm_level']}")
                print(f"⏱️ Response Time: {result['response_time']:.2f}s")
            else:
                print(f"❌ Chat failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Chat request failed: {e}")
    
    print("\n✅ API testing completed!")

if __name__ == "__main__":
    test_api()
