"""
Test FRED API Key Configuration
This script tests if your FRED API key is properly configured and working.
"""

import os
from dotenv import load_dotenv
import sys

def test_fred_api_key():
    """Test FRED API key configuration."""
    print("🔐 Testing FRED API Key Configuration")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is loaded
    api_key = os.getenv('FRED_API_KEY')
    
    if not api_key:
        print("❌ FRED_API_KEY not found in environment variables")
        print("💡 Make sure you have:")
        print("   1. Created a .env file")
        print("   2. Added FRED_API_KEY=your_actual_key")
        print("   3. The .env file is in the same directory as this script")
        return False
    
    if api_key == 'your_actual_api_key_here':
        print("⚠️  FRED_API_KEY is still set to placeholder value")
        print("💡 Please replace 'your_actual_api_key_here' with your real API key")
        return False
    
    print("✅ FRED_API_KEY loaded successfully")
    print(f"🔑 Key preview: {api_key[:8]}...{api_key[-4:]} (length: {len(api_key)})")
    
    # Test API connection
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        
        # Try to fetch a simple series
        print("\n🌐 Testing API connection...")
        gdp_data = fred.get_series('GDP', limit=1)
        
        if gdp_data is not None and len(gdp_data) > 0:
            print("✅ API connection successful!")
            print(f"📊 Sample data: GDP = {gdp_data.iloc[0]:.2f} (Date: {gdp_data.index[0]})")
            return True
        else:
            print("❌ API connection failed - no data returned")
            return False
            
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("💡 Please check your API key is valid")
        return False

if __name__ == "__main__":
    success = test_fred_api_key()
    
    if success:
        print("\n🎉 Your FRED API is properly configured!")
        print("🚀 You can now use real economic data in your models")
    else:
        print("\n❌ FRED API configuration needs attention")
        print("📖 Follow the steps above to fix the configuration")
    
    print("\n" + "="*50)
