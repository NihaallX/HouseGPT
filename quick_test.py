#!/usr/bin/env python3
"""Quick API test."""

import requests
import json

def quick_test():
    """Quick test of the API."""
    try:
        print("🧪 Quick API Test...")
        
        # Test health
        health = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health: {health.json()}")
        
        # Test chat
        chat_data = {"message": "Hello House", "user_id": "test"}
        chat = requests.post("http://localhost:8000/chat", json=chat_data, timeout=60)
        
        if chat.status_code == 200:
            result = chat.json()
            print(f"🏥 House says: {result['response']}")
            print(f"📊 Confidence: {result['confidence']:.2f}")
            print(f"⏱️ Time: {result['response_time']:.2f}s")
        else:
            print(f"❌ Chat failed: {chat.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test()
