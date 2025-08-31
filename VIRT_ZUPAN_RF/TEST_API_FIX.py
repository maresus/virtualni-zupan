"""
TEST_API_FIX.py - Debug and fix OpenAI API connection
"""

import os
import sys
import ssl
import socket
import requests
from openai import OpenAI
from dotenv import load_dotenv
import urllib3

# Load environment
load_dotenv()
load_dotenv(".env")
load_dotenv("../.env")

print("=" * 60)
print("🔧 API CONNECTION DEBUGGER")
print("=" * 60)

# 1. Check API key
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("❌ No API key in environment!")
    sys.exit(1)

print(f"✅ API Key found: {api_key[:20]}...")
print(f"   Full length: {len(api_key)} characters")

if not api_key.startswith("sk-"):
    print("⚠️ Warning: API key doesn't start with 'sk-'")

# 2. Test basic internet
print("\n📡 Testing Internet Connection:")
try:
    response = requests.get("https://www.google.com", timeout=5)
    print("  ✅ Google.com reachable")
except Exception as e:
    print(f"  ❌ No internet? {e}")

# 3. Test DNS resolution
print("\n🔍 Testing DNS:")
try:
    ip = socket.gethostbyname('api.openai.com')
    print(f"  ✅ api.openai.com resolves to: {ip}")
except Exception as e:
    print(f"  ❌ DNS error: {e}")

# 4. Test raw HTTPS connection
print("\n🔐 Testing HTTPS to OpenAI:")
try:
    response = requests.get("https://api.openai.com", timeout=5)
    print(f"  ✅ HTTPS connection works (status: {response.status_code})")
except requests.exceptions.SSLError as e:
    print(f"  ❌ SSL Error: {e}")
    print("  Try: pip install --upgrade certifi")
except Exception as e:
    print(f"  ❌ Connection error: {e}")

# 5. Test with different methods
print("\n🧪 Testing OpenAI API with different methods:")

# Method 1: Direct requests
print("\n1️⃣ Direct requests to API:")
try:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(
        "https://api.openai.com/v1/models",
        headers=headers,
        timeout=10
    )
    if response.status_code == 200:
        print("  ✅ Direct API call works!")
        models = response.json()
        print(f"  Found {len(models.get('data', []))} models")
    else:
        print(f"  ❌ API returned status: {response.status_code}")
        print(f"  Response: {response.text[:200]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Method 2: OpenAI client with custom settings
print("\n2️⃣ OpenAI client with timeout:")
try:
    import httpx
    
    # Try with custom httpx client
    http_client = httpx.Client(
        timeout=30.0,
        verify=True  # or False if SSL issues
    )
    
    client = OpenAI(
        api_key=api_key,
        http_client=http_client
    )
    
    # Simple test
    response = client.models.list()
    print(f"  ✅ Client works! Found {len(list(response))} models")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Method 3: Try with disabled SSL (NOT for production!)
print("\n3️⃣ Testing with SSL verification disabled (debug only):")
try:
    import httpx
    
    http_client = httpx.Client(
        timeout=30.0,
        verify=False  # Disable SSL verification
    )
    
    client = OpenAI(
        api_key=api_key,
        http_client=http_client
    )
    
    response = client.models.list()
    print(f"  ⚠️ Works without SSL verification!")
    print("  This suggests an SSL/certificate issue")
except Exception as e:
    print(f"  ❌ Still doesn't work: {e}")

# 6. Check for proxy
print("\n🌐 Checking for proxy settings:")
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
proxy_found = False
for var in proxy_vars:
    value = os.environ.get(var)
    if value:
        print(f"  ⚠️ {var} = {value}")
        proxy_found = True

if not proxy_found:
    print("  ✅ No proxy configured")

# 7. Try alternative base URL
print("\n🔄 Testing alternative configurations:")
try:
    # Some regions might need different base URL
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1"  # Explicit base URL
    )
    
    response = client.models.list()
    print("  ✅ Works with explicit base URL")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Final recommendations
print("\n" + "=" * 60)
print("📋 RECOMMENDATIONS")
print("=" * 60)

print("""
Based on the tests above, try:

1. If SSL error:
   pip install --upgrade certifi
   pip install --upgrade urllib3

2. If timeout:
   - Check firewall settings
   - Try using a VPN
   - Check with your ISP

3. If API key issue:
   - Verify key at: https://platform.openai.com/api-keys
   - Check if key has correct permissions
   - Check if billing is active

4. Quick workaround - use proxy:
   export HTTPS_PROXY=http://your-proxy:port

5. For immediate testing, create new client like this:
   
   import httpx
   from openai import OpenAI
   
   http_client = httpx.Client(timeout=30.0, verify=False)
   client = OpenAI(api_key=api_key, http_client=http_client)
""")