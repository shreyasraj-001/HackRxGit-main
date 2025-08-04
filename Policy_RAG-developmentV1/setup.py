#!/usr/bin/env python3
"""
Installation script for Policy RAG System
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_environment():
    """Check if .env file exists"""
    if not os.path.exists(".env"):
        print("⚠️ No .env file found!")
        print("📝 Please create a .env file with your OpenRouter API key:")
        print("   OPENROUTER_API_KEY=your_api_key_here")
        print("📖 You can copy from .env.template as a starting point")
        return False
    
    print("✅ .env file found")
    return True

def main():
    """Main installation process"""
    print("🚀 Policy RAG System Setup")
    print("=" * 40)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check environment
    env_ok = check_environment()
    
    print("\n" + "=" * 40)
    print("🎉 Setup complete!")
    
    if env_ok:
        print("\n📋 Next steps:")
        print("   1. Start the server: python main.py")
        print("   2. Initialize the system: curl http://localhost:8000/")
        print("   3. Test the batch endpoint: python test_batch_endpoint.py")
    else:
        print("\n📋 Next steps:")
        print("   1. Create .env file with your OpenRouter API key")
        print("   2. Start the server: python main.py")
        print("   3. Initialize the system: curl http://localhost:8000/")
        print("   4. Test the batch endpoint: python test_batch_endpoint.py")

if __name__ == "__main__":
    main()
