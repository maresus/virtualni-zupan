"""
TEST_EMBEDDINGS.py - Test all possible embedding configurations
"""

import os
import sys
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv()
load_dotenv(".env")
load_dotenv("../.env")

print("=" * 60)
print("🔍 EMBEDDING MODEL DETECTOR")
print("=" * 60)

# Setup paths
GITHUBE_DIR = Path("/Users/markosatler/Documents/ZUPAN JULIJ 2025/GITHUBE")
CHROMA_PATH = str(GITHUBE_DIR / "chroma_db")

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ No API key!")
    sys.exit(1)

print(f"✅ API Key: {api_key[:20]}...")
print(f"📁 ChromaDB: {CHROMA_PATH}\n")

# Test OpenAI connection first
print("Testing OpenAI API connection...")
try:
    client = OpenAI(api_key=api_key)
    # Test chat
    test_chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("✅ OpenAI Chat API works\n")
except Exception as e:
    print(f"❌ OpenAI Chat API error: {e}\n")

# Connect to ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = chroma_client.list_collections()
    print(f"Found {len(collections)} collections:")
    for col in collections:
        print(f"  - {col.name}: {col.count()} docs")
    print()
except Exception as e:
    print(f"❌ ChromaDB error: {e}")
    sys.exit(1)

# Test each collection with different embedding models
test_models = [
    ("text-embedding-3-small", 1536),
    ("text-embedding-ada-002", 1536),
    ("text-embedding-3-large", 3072),
]

for col in collections:
    print(f"\n{'='*50}")
    print(f"Testing collection: {col.name}")
    print(f"Documents: {col.count()}")
    print("="*50)
    
    # First, get collection without embedding function to check metadata
    try:
        simple_col = chroma_client.get_collection(col.name)
        sample = simple_col.get(limit=1, include=["metadatas", "documents"])
        if sample['metadatas']:
            print(f"Sample metadata keys: {list(sample['metadatas'][0].keys())[:5]}")
        if sample['documents']:
            print(f"Sample doc preview: {sample['documents'][0][:100]}...")
    except Exception as e:
        print(f"Error getting sample: {e}")
    
    print("\nTesting embedding models:")
    
    for model_name, expected_dim in test_models:
        try:
            # Create embedding function
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model_name
            )
            
            # Try to get collection with this embedding
            test_col = chroma_client.get_collection(
                name=col.name,
                embedding_function=ef
            )
            
            # Try a simple query
            results = test_col.query(
                query_texts=["test občina"],
                n_results=1
            )
            
            if results['documents'] and results['documents'][0]:
                print(f"  ✅ {model_name}: WORKS! Found {len(results['documents'][0])} results")
                print(f"     This is the correct model for {col.name}")
                
                # Do a real test query
                real_results = test_col.query(
                    query_texts=["kdaj je odvoz smeti"],
                    n_results=2
                )
                if real_results['documents'][0]:
                    print(f"     Real query test: ✅ {len(real_results['documents'][0])} results")
            else:
                print(f"  ⚠️ {model_name}: Connected but no results")
                
        except Exception as e:
            error_str = str(e)
            if "dimension" in error_str.lower():
                # Extract dimension info if available
                import re
                dim_match = re.search(r'(\d+)', error_str)
                if dim_match:
                    actual_dim = dim_match.group(1)
                    print(f"  ❌ {model_name}: Wrong dimension (expected {expected_dim}, got {actual_dim})")
                else:
                    print(f"  ❌ {model_name}: Dimension mismatch")
            elif "connection" in error_str.lower():
                print(f"  ❌ {model_name}: Connection error - API issue?")
            else:
                print(f"  ❌ {model_name}: {error_str[:50]}...")

print("\n" + "="*60)
print("💡 RECOMMENDATIONS")
print("="*60)

print("""
If all models show 'Connection error':
1. Check your internet connection
2. Verify API key has embedding permissions
3. Try: curl https://api.openai.com/v1/embeddings -H "Authorization: Bearer YOUR_KEY"
4. Check if you're behind a firewall/proxy

If dimension errors:
- The database was created with a different model
- You may need to recreate the database with the current model
""")

# Final test - try direct embedding API
print("\n" + "="*60)
print("🧪 DIRECT EMBEDDING API TEST")
print("="*60)

client = OpenAI(api_key=api_key)
for model_name, _ in test_models:
    try:
        response = client.embeddings.create(
            model=model_name,
            input="test text"
        )
        dim = len(response.data[0].embedding)
        print(f"✅ {model_name}: API works (dimension: {dim})")
    except Exception as e:
        print(f"❌ {model_name}: {str(e)[:100]}")