#!/usr/bin/env python3
"""
Helper script to set up environment variables for the Policy RAG application.
"""

import os
from dotenv import load_dotenv

def check_environment():
    """Check if required environment variables are set."""
    load_dotenv()
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY is not set!")
        print("\nTo set it up:")
        print("1. Get your API key from https://app.pinecone.io/")
        print("2. Set it in one of these ways:")
        print("   - Create a .env file with: PINECONE_API_KEY=your_key_here")
        print("   - Set it in terminal: export PINECONE_API_KEY=your_key_here")
        print("   - Set it temporarily: PINECONE_API_KEY=your_key_here python main.py")
        return False
    else:
        print("✅ PINECONE_API_KEY is set")
        return True

if __name__ == "__main__":
    check_environment() 