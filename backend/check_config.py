#!/usr/bin/env python3
"""
Pre-deployment configuration checker for Breakpoint LLM Platform.
Verifies all requirements are met before Modal deployment.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Tuple


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a required file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False


def check_python_imports() -> bool:
    """Check if all required Python packages can be imported."""
    required_packages = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "Data validation"),
        ("aiohttp", "Async HTTP client"),
        ("openai", "OpenAI API client"),
        ("anthropic", "Anthropic API client"),
        ("modal", "Modal deployment platform")
    ]
    
    print("\n🐍 Checking Python packages...")
    all_good = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {description}: {package}")
        except ImportError:
            print(f"❌ {description}: {package} (NOT INSTALLED)")
            all_good = False
    
    return all_good


def check_environment_variables() -> bool:
    """Check environment variables setup."""
    print("\n🔐 Checking environment variables...")
    
    # Optional but recommended
    env_vars = [
        ("OPENAI_API_KEY", "OpenAI API key", False),
        ("ANTHROPIC_API_KEY", "Anthropic API key", False),
        ("HUGGINGFACE_API_KEY", "HuggingFace API key", False),
    ]
    
    for var_name, description, required in env_vars:
        value = os.getenv(var_name)
        if value:
            # Don't print the actual key, just confirm it exists
            masked_value = f"{value[:6]}...{value[-4:]}" if len(value) > 10 else "***"
            print(f"✅ {description}: {masked_value}")
        elif required:
            print(f"❌ {description}: Not set (REQUIRED)")
        else:
            print(f"⚠️  {description}: Not set (will need to be configured in Modal)")
    
    return True  # Environment variables are optional for Modal deployment


def check_app_structure() -> bool:
    """Check the application structure."""
    print("\n📁 Checking application structure...")
    
    required_files = [
        ("app/main.py", "FastAPI main application"),
        ("app/models/llm_config.py", "LLM configuration models"),
        ("app/services/llm_service.py", "LLM service"),
        ("app/api/v1/models.py", "Models API endpoints"),
        ("app/core/deps.py", "Core dependencies"),
        ("modal_deploy.py", "Modal deployment configuration"),
        ("requirements.txt", "Python dependencies")
    ]
    
    all_good = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    return all_good


def check_modal_cli() -> bool:
    """Check Modal CLI installation and authentication."""
    print("\n🔧 Checking Modal CLI...")
    
    try:
        # Check if modal is installed
        result = subprocess.run(
            ["modal", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ Modal CLI installed: {result.stdout.strip()}")
        
        # Check if authenticated
        try:
            result = subprocess.run(
                ["modal", "token", "current"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print("✅ Modal CLI authenticated")
            return True
        except subprocess.CalledProcessError:
            print("❌ Modal CLI not authenticated (run: modal token new)")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Modal CLI not installed (run: pip install modal)")
        return False


def check_app_syntax() -> bool:
    """Check if the app has syntax errors."""
    print("\n🔍 Checking application syntax...")
    
    try:
        # Try to import the main app
        sys.path.insert(0, str(Path.cwd()))
        from app.main import app
        print("✅ FastAPI app imports successfully")
        
        # Basic validation
        if hasattr(app, 'routes') and len(app.routes) > 0:
            print(f"✅ FastAPI app has {len(app.routes)} routes")
            return True
        else:
            print("⚠️  FastAPI app has no routes")
            return False
            
    except Exception as e:
        print(f"❌ FastAPI app import failed: {e}")
        return False


def generate_deployment_checklist() -> None:
    """Generate a pre-deployment checklist."""
    print("\n📋 Pre-Deployment Checklist")
    print("-" * 40)
    print("Before deploying to Modal, ensure:")
    print("1. ✅ All required files are present")
    print("2. ✅ Python packages are installable")
    print("3. ✅ FastAPI app imports without errors")
    print("4. ✅ Modal CLI is installed and authenticated")
    print("5. 🔐 API keys are ready for Modal secrets")
    print("6. 🧪 Local testing completed")
    print("\nTo deploy:")
    print("  python deploy.py")
    print("\nOr manually:")
    print("  modal deploy modal_deploy.py")


def main():
    """Main configuration check."""
    print("🎯 Breakpoint LLM Platform - Pre-Deployment Check")
    print("=" * 55)
    
    checks = [
        ("Application Structure", check_app_structure),
        ("Python Packages", check_python_imports), 
        ("Environment Variables", check_environment_variables),
        ("Application Syntax", check_app_syntax),
        ("Modal CLI", check_modal_cli),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}")
        print("-" * (len(check_name) + 4))
        result = check_func()
        results.append((check_name, result))
    
    # Summary
    print("\n" + "=" * 55)
    print("📊 Configuration Check Summary")
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {check_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready for Modal deployment.")
        generate_deployment_checklist()
        return 0
    elif passed >= total * 0.8:  # 80% pass rate
        print("\n⚠️  Most checks passed. Review failed items before deploying.")
        generate_deployment_checklist()
        return 0
    else:
        print("\n❌ Multiple issues detected. Please fix before deploying.")
        print("\n🔧 Common fixes:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Install Modal CLI: pip install modal")
        print("   - Authenticate Modal: modal token new")
        print("   - Check file paths and imports")
        return 1


if __name__ == "__main__":
    sys.exit(main())
