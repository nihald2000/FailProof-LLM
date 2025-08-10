#!/usr/bin/env python3
"""
Quick deployment script for Breakpoint LLM Platform to Modal.
Run this script to deploy your FastAPI backend to Modal with all configurations.
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔄 {description or cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False


def check_modal_cli():
    """Check if Modal CLI is installed and authenticated."""
    print("🔍 Checking Modal CLI...")
    
    # Check if modal is installed
    if not run_command("modal --version", "Checking Modal CLI installation"):
        print("📦 Installing Modal CLI...")
        if not run_command("pip install modal", "Installing Modal"):
            print("❌ Failed to install Modal CLI")
            return False
    
    # Check if authenticated
    if not run_command("modal token current", "Checking Modal authentication"):
        print("🔐 Modal authentication required.")
        print("Please run: modal token new")
        return False
    
    return True


def create_secrets():
    """Guide user through creating required secrets."""
    print("\n🔐 Setting up Modal secrets...")
    
    secrets_config = [
        {
            "name": "openai-api-key",
            "env_var": "OPENAI_API_KEY", 
            "description": "OpenAI API key (starts with sk-)",
            "required": True
        },
        {
            "name": "anthropic-api-key", 
            "env_var": "ANTHROPIC_API_KEY",
            "description": "Anthropic API key (starts with sk-ant-)",
            "required": False
        },
        {
            "name": "huggingface-api-key",
            "env_var": "HUGGINGFACE_API_KEY", 
            "description": "HuggingFace API key (starts with hf_)",
            "required": False
        }
    ]
    
    for secret in secrets_config:
        # Check if secret already exists
        check_cmd = f"modal secret list | grep {secret['name']}"
        if run_command(check_cmd, f"Checking if {secret['name']} exists"):
            print(f"✅ Secret {secret['name']} already exists")
            continue
        
        if secret['required']:
            print(f"\n🔑 Required: {secret['description']}")
            api_key = input(f"Enter your {secret['env_var']}: ").strip()
            
            if not api_key:
                print(f"❌ {secret['env_var']} is required!")
                return False
            
            create_cmd = f'modal secret create {secret["name"]} {secret["env_var"]}="{api_key}"'
            if not run_command(create_cmd, f"Creating {secret['name']} secret"):
                return False
        else:
            print(f"\n🔑 Optional: {secret['description']}")
            api_key = input(f"Enter your {secret['env_var']} (press Enter to skip): ").strip()
            
            if api_key:
                create_cmd = f'modal secret create {secret["name"]} {secret["env_var"]}="{api_key}"'
                run_command(create_cmd, f"Creating {secret['name']} secret")
            else:
                print(f"⏭️  Skipping {secret['name']}")
    
    return True


def deploy_to_modal():
    """Deploy the application to Modal."""
    print("\n🚀 Deploying to Modal...")
    
    # Check if modal_deploy.py exists
    if not Path("modal_deploy.py").exists():
        print("❌ modal_deploy.py not found in current directory")
        return False
    
    # Deploy the application
    if not run_command("modal deploy modal_deploy.py", "Deploying FastAPI app to Modal"):
        return False
    
    print("\n✅ Deployment completed successfully!")
    return True


def test_deployment():
    """Test the deployed application."""
    print("\n🧪 Testing deployment...")
    
    # Get app info
    if run_command("modal app show breakpoint-llm-platform", "Getting app information"):
        print("\n🔗 Your app is deployed!")
        print("📖 API Documentation: https://your-app-url.modal.run/docs")
        print("💚 Health Check: https://your-app-url.modal.run/health")
        
        print("\n📋 To get your exact URL, run:")
        print("   modal app show breakpoint-llm-platform")
        
        return True
    
    return False


def main():
    """Main deployment workflow."""
    print("🎯 Breakpoint LLM Platform - Modal Deployment")
    print("=" * 50)
    
    # Change to backend directory if not already there
    if Path("app").exists() and Path("modal_deploy.py").exists():
        print("✅ Already in backend directory")
    elif Path("backend").exists():
        print("📁 Changing to backend directory...")
        os.chdir("backend")
    else:
        print("❌ Cannot find backend directory with app/ folder")
        return 1
    
    # Step 1: Check Modal CLI
    if not check_modal_cli():
        return 1
    
    # Step 2: Create secrets
    if not create_secrets():
        return 1
    
    # Step 3: Deploy to Modal
    if not deploy_to_modal():
        return 1
    
    # Step 4: Test deployment
    if not test_deployment():
        print("⚠️  Deployment completed but testing failed")
        print("   Check the Modal dashboard for details")
    
    print("\n🎉 Deployment workflow completed!")
    print("\n📚 Next steps:")
    print("   1. Test your API endpoints")
    print("   2. Update your frontend to use the Modal URL")
    print("   3. Monitor logs with: modal logs breakpoint-llm-platform")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
