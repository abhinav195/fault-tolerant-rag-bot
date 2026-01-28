import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# ✅ THE WINNING MODEL
MODEL_NAME = "models/gemini-embedding-001"

print("=" * 60)
print(f"🔑 TESTING API KEYS WITH: {MODEL_NAME}")
print("=" * 60)

valid_keys = []
invalid_keys = []

# Test all 3 keys
for i in range(1, 4):
    api_key = os.getenv(f"GOOGLE_API_KEY_{i}")
    
    if not api_key:
        print(f"\n❌ Key {i}: GOOGLE_API_KEY_{i} not found in .env")
        invalid_keys.append(i)
        continue
    
    print(f"\n🔑 Testing Key {i}: {api_key[:20]}...")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=api_key
        )
        
        # Test embedding
        result = embeddings.embed_query("test")
        
        print(f"   ✅ SUCCESS! Key {i} is VALID. (Dim: {len(result)})")
        valid_keys.append(i)
        
    except Exception as e:
        error_msg = str(e)
        invalid_keys.append(i)
        
        if "429" in error_msg:
            print(f"   ⚠️  Key {i}: Quota exhausted (429)")
        elif "403" in error_msg or "PERMISSION_DENIED" in error_msg:
            print(f"   ❌ Key {i}: Invalid or Expired Key (403)")
        elif "400" in error_msg:
             print(f"   ❌ Key {i}: Bad Request (400) - Check API Key permissions")
        else:
             print(f"   ❌ Key {i}: Error - {error_msg[:100]}...")

print("\n" + "=" * 60)
print("📊 KEY STATUS SUMMARY")
print("=" * 60)
print(f"✅ Valid Keys: {len(valid_keys)}/3   -> {valid_keys}")
print(f"❌ Invalid Keys: {len(invalid_keys)}/3 -> {invalid_keys}")
print("=" * 60)
