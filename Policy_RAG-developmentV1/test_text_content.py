#!/usr/bin/env python3
"""
Test to check if PDF text content is being parsed correctly.
"""

from main import PDFDocument

def test_text_content():
    """Test if PDF text content is being parsed correctly"""
    print("🔍 Testing PDF Text Content...")
    print("=" * 50)
    
    file_link = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        document = PDFDocument(file_link)
        data = document.parse_pdf()
        
        print(f"✅ PDF parsed successfully")
        print(f"📏 Text length: {len(data)} characters")
        print(f"📝 First 500 characters:")
        print("-" * 50)
        print(data[:500])
        print("-" * 50)
        
        # Check for common issues
        if not data.strip():
            print("❌ Text is empty or only whitespace!")
        elif len(data.strip()) < 100:
            print("⚠️ Text seems too short - might be parsing issue")
        else:
            print("✅ Text content looks good")
            
        # Check for specific content
        if "grace period" in data.lower():
            print("✅ Found 'grace period' in text")
        else:
            print("⚠️ 'grace period' not found in text")
            
        if "premium" in data.lower():
            print("✅ Found 'premium' in text")
        else:
            print("⚠️ 'premium' not found in text")
            
    except Exception as e:
        print(f"❌ PDF parsing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_text_content() 