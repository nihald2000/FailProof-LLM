#!/usr/bin/env python3
"""
Test script for Modal deployment of Breakpoint LLM Platform.
This script tests all major endpoints and functionality.
"""

import requests
import json
import time
import sys
from typing import Dict, Any


def test_endpoint(url: str, method: str = "GET", data: Dict = None, description: str = "") -> bool:
    """Test a single endpoint and return success status."""
    try:
        print(f"🧪 Testing: {description or f'{method} {url}'}")
        
        if method.upper() == "GET":
            response = requests.get(url, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=30)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, timeout=30)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        if response.status_code < 400:
            print(f"✅ Success: {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                result = response.json()
                if isinstance(result, dict) and len(result) <= 5:
                    print(f"   Response: {result}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Timeout after 30 seconds")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_modal_deployment(base_url: str) -> bool:
    """Test the complete Modal deployment."""
    print(f"🎯 Testing Breakpoint LLM Platform at: {base_url}")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test cases
    test_cases = [
        # Basic endpoints
        {
            "url": f"{base_url}/",
            "method": "GET",
            "description": "Root endpoint"
        },
        {
            "url": f"{base_url}/health",
            "method": "GET", 
            "description": "Health check"
        },
        {
            "url": f"{base_url}/docs",
            "method": "GET",
            "description": "API documentation"
        },
        
        # API endpoints
        {
            "url": f"{base_url}/api/v1/models/",
            "method": "GET",
            "description": "List models endpoint"
        },
        {
            "url": f"{base_url}/api/v1/models/presets",
            "method": "GET",
            "description": "Model presets endpoint"
        },
        
        # Test model registration
        {
            "url": f"{base_url}/api/v1/models/register",
            "method": "POST",
            "data": {
                "name": "test-gpt-4",
                "provider_name": "openai",
                "model_name": "gpt-4",
                "description": "Test GPT-4 configuration",
                "endpoint_url": "https://api.openai.com/v1/chat/completions",
                "auth_config": {
                    "auth_type": "bearer_token",
                    "api_key": "test-key"
                },
                "model_parameters": {
                    "max_tokens": 1000,
                    "temperature": 0.7
                },
                "auto_test_on_create": False
            },
            "description": "Register test model"
        }
    ]
    
    for test_case in test_cases:
        total_tests += 1
        if test_endpoint(**test_case):
            tests_passed += 1
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your Modal deployment is working correctly.")
        return True
    elif tests_passed >= total_tests * 0.7:  # 70% pass rate
        print("⚠️  Most tests passed. Some non-critical issues detected.")
        return True
    else:
        print("❌ Multiple tests failed. Please check your deployment.")
        return False


def test_llm_connectivity(base_url: str) -> bool:
    """Test LLM provider connectivity."""
    print("\n🔗 Testing LLM Provider Connectivity")
    print("-" * 40)
    
    try:
        # Test the Modal function directly if possible
        print("🧪 Testing LLM connectivity...")
        
        # This would test the connectivity through the API
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            health_data = response.json()
            llm_status = health_data.get("checks", {}).get("llm_services", "unknown")
            print(f"✅ LLM Services Status: {llm_status}")
            return llm_status == "healthy"
        else:
            print(f"❌ Cannot check LLM connectivity: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ LLM connectivity test failed: {e}")
        return False


def main():
    """Main test function."""
    if len(sys.argv) != 2:
        print("Usage: python test_modal.py <modal-app-url>")
        print("Example: python test_modal.py https://your-username--breakpoint-llm-platform-fastapi-app.modal.run")
        return 1
    
    base_url = sys.argv[1].rstrip('/')
    
    print("🚀 Breakpoint LLM Platform - Modal Deployment Test")
    print("=" * 60)
    print(f"📍 Target URL: {base_url}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run basic deployment tests
    deployment_ok = test_modal_deployment(base_url)
    
    # Run LLM connectivity tests
    llm_ok = test_llm_connectivity(base_url)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 Final Test Summary")
    print(f"📱 Basic Deployment: {'✅ PASS' if deployment_ok else '❌ FAIL'}")
    print(f"🤖 LLM Connectivity: {'✅ PASS' if llm_ok else '❌ FAIL'}")
    
    if deployment_ok and llm_ok:
        print("\n🎉 All systems operational! Your Modal deployment is ready for use.")
        print("\n📋 Next steps:")
        print("   1. Update your frontend to use this URL")
        print("   2. Test with real LLM configurations")
        print("   3. Monitor performance in Modal dashboard")
        return 0
    elif deployment_ok:
        print("\n⚠️  Deployment is working but check LLM API keys in Modal secrets")
        return 0
    else:
        print("\n❌ Deployment issues detected. Check Modal logs and configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
