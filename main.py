import os
import re
import itertools
import time
import logging
from datetime import datetime
from typing import TypedDict, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# ============= LOGGING CONFIGURATION =============

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging (similar to SLF4J in Java)
log_filename = f"logs/rag_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_filename}")

# ============= MULTI-KEY CONFIGURATION =============

def load_api_keys():
    """Load multiple Google API keys from .env"""
    keys = []
    
    # Method 1: Load numbered keys (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.)
    i = 1
    while True:
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    
    # Method 2: Comma-separated list (fallback)
    if not keys:
        keys_str = os.getenv("GOOGLE_API_KEYS")
        if keys_str:
            keys = [k.strip() for k in keys_str.split(",")]
    
    # Method 3: Single key (fallback)
    if not keys:
        single_key = os.getenv("GOOGLE_API_KEY")
        if single_key:
            keys = [single_key]
    
    if not keys:
        raise ValueError("❌ No Google API keys found in .env!")
    
    return keys

# Load all API keys
API_KEYS = load_api_keys()

# Global key tracking - remembers which key is currently working
class KeyTracker:
    def __init__(self):
        self.embedding_key_index = 0
        self.llm_key_index = 0
        self.search_key_index = 0
    
    def get_embedding_key(self):
        return API_KEYS[self.embedding_key_index % len(API_KEYS)], self.embedding_key_index % len(API_KEYS) + 1
    
    def rotate_embedding_key(self):
        self.embedding_key_index += 1
        logger.info(f"🔄 Rotated embedding key to Key {self.embedding_key_index % len(API_KEYS) + 1}")
    
    def get_llm_key(self):
        return API_KEYS[self.llm_key_index % len(API_KEYS)], self.llm_key_index % len(API_KEYS) + 1
    
    def rotate_llm_key(self):
        self.llm_key_index += 1
        logger.info(f"🔄 Rotated LLM key to Key {self.llm_key_index % len(API_KEYS) + 1}")
    
    def get_search_key(self):
        return API_KEYS[self.search_key_index % len(API_KEYS)], self.search_key_index % len(API_KEYS) + 1
    
    def rotate_search_key(self):
        self.search_key_index += 1
        logger.info(f"🔄 Rotated search key to Key {self.search_key_index % len(API_KEYS) + 1}")

key_tracker = KeyTracker()

# ============= FAULT-TOLERANT LLM WRAPPER =============

def invoke_llm_with_fallback(prompt: str, max_retries=None):
    """
    Invoke LLM with automatic API key fallback on quota exhaustion
    
    Args:
        prompt: The prompt to send to the LLM
        max_retries: Maximum number of keys to try (defaults to all keys)
    
    Returns:
        LLM response content
    """
    if max_retries is None:
        max_retries = len(API_KEYS)
    
    for attempt in range(max_retries):
        key, key_num = key_tracker.get_llm_key()
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=key,
                temperature=0
            )
            
            response = llm.invoke(prompt)
            
            if attempt > 0:
                logger.info(f"   ✅ LLM call SUCCESS with Key {key_num} (after {attempt} retries)")
            
            return response.content
            
        except Exception as e:
            error_msg = str(e)
            
            is_quota_exhausted = any(x in error_msg for x in [
                "RESOURCE_EXHAUSTED", 
                "429", 
                "quota", 
                "rate limit"
            ])
            
            if is_quota_exhausted:
                if attempt < max_retries - 1:
                    logger.warning(f"   ⚠️  LLM Key {key_num} exhausted, trying next key...")
                    key_tracker.rotate_llm_key()
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"   ❌ All {max_retries} API keys exhausted for LLM!")
                    raise Exception(f"All API keys exhausted. Please wait or add more keys.")
            else:
                # Non-quota error, raise immediately
                raise e
    
    raise Exception("Failed to invoke LLM after all retries")

# ============= INITIALIZE FASTAPI =============

app = FastAPI(
    title="RAG Q&A Support Bot",
    description="Production-ready RAG system with Full Fault Tolerance, Multi-Key Support, Persistent Key Tracking",
    version="2.3.2",
    contact={
        "name": "RAG Bot Support",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT"
    },
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============= FAULT-TOLERANT MULTI-KEY CHROMA WRAPPER =============

class MultiKeyChroma:
    """ChromaDB wrapper with fault-tolerant automatic API key rotation"""
    
    def __init__(self, collection_name="rag_collection", persist_directory="./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize with first key
        key, _ = key_tracker.get_embedding_key()
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",  # ✅ UPDATED MODEL
            google_api_key=key
        )
        
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self, documents):
        """Add documents with fault tolerance and automatic recovery"""
        batch_size = 10
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        # Track batch status
        batch_status = {}
        failed_batches = []
        max_retries_per_batch = len(API_KEYS) * 2
        
        logger.info("="*60)
        logger.info("🚀 FAULT-TOLERANT EMBEDDING STARTED")
        logger.info("="*60)
        logger.info(f"📊 Total Chunks: {len(documents)}")
        logger.info(f"📦 Total Batches: {total_batches} (batch size: {batch_size})")
        logger.info(f"🔑 API Keys Available: {len(API_KEYS)}")
        logger.info(f"🛡️  Fault Tolerance: ENABLED")
        logger.info("="*60)
        
        def process_batch_with_retry(batch_idx, batch, retry_count=0):
            """Process a single batch with automatic retry on failure"""
            
            # Track failed keys and detect non-transient errors
            failed_keys = set()
            last_error_type = None
            
            if retry_count >= max_retries_per_batch:
                logger.error(f"❌ Batch {batch_idx+1}/{total_batches} FAILED after {retry_count} retries")
                return False
            
            # Get current API key (remembers last working key)
            key, key_num = key_tracker.get_embedding_key()
            
            # Check if all keys have been tried
            if key_num in failed_keys and len(failed_keys) >= len(API_KEYS):
                logger.error(f"❌ All {len(API_KEYS)} API keys have failed for Batch {batch_idx+1}/{total_batches}")
                logger.error(f"   └─ All keys failed with same error type: {last_error_type}")
                return False
            
            try:
                # Create embeddings with current key
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",  # ✅ UPDATED MODEL
                    google_api_key=key
                )
                
                # Create temporary vectorstore
                temp_vs = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=embeddings,
                    persist_directory=self.persist_directory
                )
                
                # Add batch
                temp_vs.add_documents(batch)
                
                # Success
                if retry_count > 0:
                    logger.info(f"✅ Batch {batch_idx+1}/{total_batches} SUCCESS (Key {key_num}, recovered after {retry_count} retries)")
                else:
                    logger.info(f"✅ Batch {batch_idx+1}/{total_batches} (Key {key_num})")
                
                batch_status[batch_idx] = "success"
                return True
                
            except Exception as e:
                error_msg = str(e)
                
                # Track this key as failed
                failed_keys.add(key_num)
                
                # Detect error type
                is_not_found = any(x in error_msg for x in ["404", "NOT_FOUND", "not found"])
                is_invalid_model = any(x in error_msg for x in ["Invalid", "invalid", "model not found"])
                is_rate_limit = any(x in error_msg.lower() for x in ["429", "quota", "rate limit", "too many requests", "resource_exhausted"])
                is_api_key_error = any(x in error_msg.lower() for x in ["invalid_argument", "expired", "api key", "invalid key", "unauthorized", "403"])
                
                # Determine error category
                if is_not_found or is_invalid_model:
                    last_error_type = "NON_TRANSIENT_MODEL_ERROR"
                elif is_rate_limit:
                    last_error_type = "RATE_LIMIT"
                elif is_api_key_error:
                    last_error_type = "API_KEY_ERROR"
                else:
                    last_error_type = "UNKNOWN"
                
                # Log the error
                if retry_count == 0:
                    logger.warning(f"⚠️  Batch {batch_idx+1}/{total_batches} FAILED with Key {key_num}")
                
                # For non-transient errors (404, invalid model), fail fast
                if is_not_found or is_invalid_model:
                    logger.error(f"   └─ NON-TRANSIENT ERROR: {error_msg[:100]}...")
                    logger.error(f"   └─ This error will affect ALL keys. Stopping retries.")
                    
                    # Check if we've tried at least one other key
                    if len(failed_keys) >= 2:
                        logger.error(f"   └─ Confirmed: {len(failed_keys)} keys failed with same error")
                        return False
                    elif len(failed_keys) >= len(API_KEYS):
                        logger.error(f"   └─ All {len(API_KEYS)} keys failed")
                        return False
                    else:
                        logger.warning(f"   └─ Trying one more key to confirm...")
                        key_tracker.rotate_embedding_key()
                        time.sleep(1)
                        return process_batch_with_retry(batch_idx, batch, retry_count + 1)
                
                # For transient errors, continue rotation
                if is_rate_limit:
                    logger.warning(f"   └─ Rate limit detected (Key {key_num}), switching to next key...")
                    key_tracker.rotate_embedding_key()
                    time.sleep(2)
                elif is_api_key_error:
                    logger.warning(f"   └─ API key issue (Key {key_num}), switching to next key...")
                    key_tracker.rotate_embedding_key()
                    time.sleep(1)
                else:
                    logger.warning(f"   └─ Error: {error_msg[:80]}...")
                    key_tracker.rotate_embedding_key()
                    time.sleep(1)
                
                # Retry with next key
                return process_batch_with_retry(batch_idx, batch, retry_count + 1)
        
        # Process all batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_idx = i // batch_size
            
            success = process_batch_with_retry(batch_idx, batch)
            
            if not success:
                failed_batches.append(batch_idx)
        
        # Summary
        successful = sum(1 for status in batch_status.values() if status == "success")
        
        logger.info("="*60)
        logger.info("📊 EMBEDDING SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Successful: {successful}/{total_batches} batches")
        logger.info(f"❌ Failed: {len(failed_batches)} batches")
        
        if failed_batches:
            logger.error(f"⚠️  Failed Batch Numbers: {[b+1 for b in failed_batches]}")
            logger.info("="*60)
            raise Exception(f"Failed to index {len(failed_batches)} batches after all retry attempts")
        else:
            logger.info("🎉 ALL CHUNKS INDEXED SUCCESSFULLY!")
            logger.info("="*60)
    
    def similarity_search(self, query, k=5):
        """Search with current embeddings"""
        return self._vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query, k=5):
        """Search with scores using fault-tolerant key rotation with persistence"""
        # Start from last working key (not always Key 1!)
        for attempt in range(len(API_KEYS)):
            key, key_num = key_tracker.get_search_key()
            
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",  # ✅ UPDATED MODEL
                    google_api_key=key
                )
                self._vectorstore._embedding_function = embeddings
                return self._vectorstore.similarity_search_with_score(query, k=k)
            except Exception as e:
                if attempt == len(API_KEYS) - 1:
                    # Last key also failed
                    raise e
                # Try next key
                logger.warning(f"   ⚠️  Search failed with Key {key_num}, switching to next key...")
                key_tracker.rotate_search_key()
                time.sleep(1)
                continue
    
    @property
    def _collection(self):
        """Access underlying collection"""
        return self._vectorstore._collection

# ============= INITIALIZE COMPONENTS =============

# Initialize multi-key vectorstore
vectorstore = MultiKeyChroma()

# Text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# ============= HELPER FUNCTIONS =============

def clean_html(soup: BeautifulSoup) -> str:
    """Aggressively strip unwanted HTML elements"""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "meta", "noscript", "iframe"]):
        tag.decompose()
    
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def detect_spa_or_blocked(html_content: str, status_code: int) -> bool:
    """Detect if page is SPA/JS-heavy or blocked"""
    if status_code == 403:
        return True
    
    text_length = len(html_content.strip())
    if text_length < 500:
        return True
    
    spa_indicators = [
        'id="root"',
        'id="app"',
        'data-reactroot',
        'ng-app',
        'vue-app',
        '<div id="__next"'
    ]
    
    if any(indicator in html_content for indicator in spa_indicators):
        soup = BeautifulSoup(html_content, 'lxml')
        visible_text = soup.get_text(strip=True)
        if len(visible_text) < 500:
            return True
    
    return False

# ============= PYDANTIC MODELS =============

class CrawlRequest(BaseModel):
    url: HttpUrl
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
            }
        }

class AskRequest(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is artificial intelligence?"
            }
        }

# ============= API ENDPOINTS =============

@app.get("/", tags=["System"])
async def root():
    """API Information and Available Endpoints"""
    return {
        "message": "RAG Q&A Support Bot - Ready",
        "version": "2.3.2",
        "features": ["Full Fault Tolerance", "Persistent Key Tracking", "File Logging", "Auto-Recovery", "Smart Retry Logic"],
        "api_keys_loaded": len(API_KEYS),
        "rate_limit": f"{len(API_KEYS) * 15} requests/minute",
        "log_file": log_filename,
        "endpoints": {
            "GET /": "API info",
            "GET /health": "Health check",
            "GET /stats": "System statistics",
            "GET /docs": "Swagger UI",
            "GET /redoc": "ReDoc UI",
            "POST /crawl": "Crawl and index a URL",
            "POST /ask": "Ask a question",
            "DELETE /clear": "Clear vector database"
        },
        "langsmith": "https://smith.langchain.com",
        "documentation": "http://localhost:8000/docs"
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Health Check Endpoint"""
    try:
        collection = vectorstore._collection
        count = collection.count()
        
        # Test LLM with fallback
        test_response = invoke_llm_with_fallback("test", max_retries=1)
        
        return {
            "status": "healthy",
            "service": "RAG Q&A Bot",
            "vectordb": "connected",
            "llm": "connected",
            "indexed_chunks": count,
            "api_keys": len(API_KEYS),
            "fault_tolerance": "enabled (embeddings + LLM + persistent keys + smart retry)",
            "rate_limit": f"{len(API_KEYS) * 15} req/min",
            "log_file": log_filename,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/stats", tags=["System"])
async def get_stats():
    """Get System Statistics"""
    try:
        collection = vectorstore._collection
        count = collection.count()
        
        if count > 0:
            results = collection.get(include=["metadatas"])
            sources = list(set([meta.get("source", "unknown") for meta in results.get("metadatas", [])]))
        else:
            sources = []
        
        return {
            "total_chunks": count,
            "unique_sources": len(sources),
            "indexed_urls": sources,
            "embedding_model": "models/gemini-embedding-001",  # ✅ UPDATED MODEL
            "llm_model": "gemini-2.5-flash-lite",
            "vector_db": "ChromaDB (local persistent)",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "api_keys_loaded": len(API_KEYS),
            "fault_tolerance": "enabled (embeddings + LLM + persistent keys + smart retry)",
            "rate_limit": f"{len(API_KEYS) * 15} requests/minute",
            "log_file": log_filename,
            "current_keys": {
                "embedding": key_tracker.embedding_key_index % len(API_KEYS) + 1,
                "llm": key_tracker.llm_key_index % len(API_KEYS) + 1,
                "search": key_tracker.search_key_index % len(API_KEYS) + 1
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

@app.delete("/clear", tags=["System"])
async def clear_database():
    """Clear Vector Database"""
    try:
        collection = vectorstore._collection
        old_count = collection.count()
        
        if old_count > 0:
            ids = collection.get()["ids"]
            collection.delete(ids=ids)
        
        logger.info(f"🗑️  Cleared {old_count} chunks from database")
        
        return {
            "status": "success",
            "message": f"Cleared {old_count} chunks from database",
            "current_count": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.post("/crawl", tags=["RAG Operations"])
async def crawl_endpoint(request: CrawlRequest):
    """Crawl and Index a URL with Fault Tolerance"""
    url = str(request.url)
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if detect_spa_or_blocked(response.text, response.status_code):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot crawl: Site is either SPA/JS-heavy or returned 403. Status: {response.status_code}"
            )
        
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        cleaned_text = clean_html(soup)
        
        if len(cleaned_text) < 100:
            raise HTTPException(
                status_code=400,
                detail="Extracted text too short. Likely a dynamic site requiring JavaScript."
            )
        
        chunks = text_splitter.split_text(cleaned_text)
        
        documents = [
            Document(page_content=chunk, metadata={"source": url, "chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]
        
        logger.info(f"🌐 Crawling URL: {url}")
        
        # Add documents with fault tolerance
        vectorstore.add_documents(documents)
        
        return {
            "status": "success",
            "url": url,
            "chunks_created": len(chunks),
            "total_characters": len(cleaned_text),
            "api_keys_used": len(API_KEYS),
            "fault_tolerance": "enabled",
            "rate_limit": f"{len(API_KEYS) * 15} req/min"
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# ============= LANGGRAPH LOGIC =============

class GraphState(TypedDict):
    """State for the Q&A graph"""
    query: str
    documents: list[Document]
    answer: str
    search_attempts: int
    answer_attempts: int
    enough_results: bool
    similarity_scores: list[float]
    answer_quality_good: bool

def search_node(state: GraphState) -> dict:
    """Retrieve documents from ChromaDB with similarity scores"""
    query = state["query"]
    search_attempts = state.get("search_attempts", 0) + 1
    answer_attempts = state.get("answer_attempts", 0)
    
    k_values = {1: 5, 2: 7, 3: 10}
    k = k_values.get(search_attempts, 10)
    
    logger.info("="*60)
    logger.info(f"🔍 SEARCH ATTEMPT {search_attempts} (Answer Attempt {answer_attempts}) - Retrieving top {k} documents")
    logger.info("="*60)
    
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    
    docs = [doc for doc, score in docs_and_scores]
    scores = [float(score) for doc, score in docs_and_scores]
    
    quality_thresholds = {1: 0.7, 2: 0.5, 3: 0.3}
    min_score = quality_thresholds.get(search_attempts, 0.3)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    logger.info(f"📊 Retrieved {len(docs)} documents")
    logger.info(f"📈 Similarity Scores: {[f'{s:.3f}' for s in scores[:5]]}")
    logger.info(f"📉 Average Score: {avg_score:.3f}")
    logger.info(f"🎯 Quality Threshold (Attempt {search_attempts}): {min_score}")
    
    quality_check_passed = avg_score >= min_score and len(docs) >= 3
    
    if quality_check_passed:
        logger.info(f"✅ QUALITY CHECK PASSED! (avg={avg_score:.3f} >= {min_score})")
        enough_results = True
    else:
        logger.warning(f"❌ QUALITY CHECK FAILED! (avg={avg_score:.3f} < {min_score} or docs < 3)")
        enough_results = False
    
    return {
        "documents": docs,
        "search_attempts": search_attempts,
        "enough_results": enough_results,
        "similarity_scores": scores
    }

def summarize_node(state: GraphState) -> dict:
    """Generate answer using retrieved context"""
    query = state["query"]
    documents = state["documents"]
    search_attempts = state.get("search_attempts", 0)
    answer_attempts = state.get("answer_attempts", 0) + 1
    scores = state.get("similarity_scores", [])
    
    logger.info("="*60)
    logger.info(f"✨ SUMMARIZE NODE - Generating Answer (Attempt {answer_attempts})")
    logger.info("="*60)
    logger.info(f"📚 Using {len(documents)} documents")
    logger.info(f"🔍 After {search_attempts} search attempt(s)")
    logger.info(f"📊 Scores: {[f'{s:.3f}' for s in scores[:5]]}")
    
    context = "\n\n".join([doc.page_content for doc in documents])
    
    prompt = f"""You are a helpful assistant. Answer the question ONLY using the context provided below.
If the context does not contain enough information to answer, say "I don't have enough information to answer this question based on the crawled content."

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        answer = invoke_llm_with_fallback(prompt)
        logger.info(f"✅ Answer generated: {answer[:100]}...")
    except Exception as e:
        logger.error(f"❌ All API keys exhausted for LLM: {str(e)}")
        answer = "All API keys are exhausted. Please wait for quota reset or add more API keys."
    
    return {
        "answer": answer,
        "answer_attempts": answer_attempts
    }

def validate_answer_node(state: GraphState) -> dict:
    """Validate answer quality"""
    answer = state["answer"]
    answer_attempts = state.get("answer_attempts", 0)
    
    logger.info("="*60)
    logger.info("🔎 VALIDATE ANSWER NODE - Checking Answer Quality")
    logger.info("="*60)
    logger.info(f"📝 Answer Attempt: {answer_attempts}")
    
    refusal_phrases = [
        "I don't have enough information",
        "based on the crawled content",
        "not enough information",
        "cannot answer",
        "API keys are exhausted"
    ]
    
    is_refusal = any(phrase.lower() in answer.lower() for phrase in refusal_phrases)
    
    if is_refusal:
        logger.warning("⚠️  Answer is a REFUSAL - quality check FAILED")
        answer_quality_good = False
    else:
        if len(answer.strip()) < 50:
            logger.warning(f"⚠️  Answer too SHORT ({len(answer)} chars) - quality check FAILED")
            answer_quality_good = False
        else:
            logger.info(f"✅ Answer looks GOOD ({len(answer)} chars) - quality check PASSED")
            answer_quality_good = True
    
    return {
        "answer_quality_good": answer_quality_good
    }

def fallback_node(state: GraphState) -> dict:
    """Handle retry logic"""
    search_attempts = state.get("search_attempts", 0)
    answer_attempts = state.get("answer_attempts", 0)
    scores = state.get("similarity_scores", [])
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    logger.info("="*60)
    logger.info("⚠️  FALLBACK NODE - Retry Preparation")
    logger.info("="*60)
    logger.info(f"🔄 Current search attempt: {search_attempts}")
    logger.info(f"🔄 Current answer attempt: {answer_attempts}")
    logger.info(f"📉 Quality score was: {avg_score:.3f}")
    logger.info("🔄 Will retry with broader search...")
    
    return {
        "enough_results": False
    }

def final_fallback_node(state: GraphState) -> dict:
    """Final fallback when max attempts reached"""
    search_attempts = state.get("search_attempts", 0)
    answer_attempts = state.get("answer_attempts", 0)
    scores = state.get("similarity_scores", [])
    
    logger.info("="*60)
    logger.info("🛑 FINAL FALLBACK NODE - Max Retries Reached")
    logger.info("="*60)
    logger.error(f"❌ Failed after {search_attempts} search attempts and {answer_attempts} answer attempts")
    logger.info(f"📉 Best scores: {[f'{s:.3f}' for s in scores[:3]] if scores else 'None'}")
    
    return {
        "answer": f"Could not find high-quality relevant results after {search_attempts} search attempts and {answer_attempts} answer attempts. Please try:\n1. Crawling more relevant URLs\n2. Rephrasing your question\n3. Asking about topics covered in the indexed content"
    }

def route_after_search(state: GraphState) -> Literal["summarize", "fallback"]:
    """Route based on search quality"""
    enough_results = state.get("enough_results", False)
    search_attempts = state.get("search_attempts", 0)
    scores = state.get("similarity_scores", [])
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    logger.info("🔀 ROUTING DECISION after search:")
    logger.info(f"   Search Attempts: {search_attempts}, Enough Results: {enough_results}, Avg Score: {avg_score:.3f}")
    
    if enough_results:
        logger.info("   ➡️  Route: SUMMARIZE (quality passed!)")
        return "summarize"
    else:
        logger.info("   ➡️  Route: FALLBACK (quality too low, will retry)")
        return "fallback"

def route_after_validate(state: GraphState) -> Literal["end", "fallback"]:
    """Route based on answer quality"""
    answer_quality_good = state.get("answer_quality_good", False)
    answer_attempts = state.get("answer_attempts", 0)
    max_answer_attempts = 3
    
    logger.info("🔀 ROUTING DECISION after validate:")
    logger.info(f"   Answer Attempts: {answer_attempts}/{max_answer_attempts}, Quality Good: {answer_quality_good}")
    
    if answer_quality_good:
        logger.info("   ➡️  Route: END (answer is good!)")
        return "end"
    elif answer_attempts >= max_answer_attempts:
        logger.info("   ➡️  Route: END (max answer attempts reached, keeping best answer)")
        return "end"
    else:
        logger.info("   ➡️  Route: FALLBACK (answer quality low, will re-search)")
        return "fallback"

def route_after_fallback(state: GraphState) -> Literal["search", "final_fallback"]:
    """Decide whether to retry search or give up"""
    search_attempts = state.get("search_attempts", 0)
    max_search_attempts = 3
    
    logger.info("🔀 ROUTING DECISION after fallback:")
    logger.info(f"   Search Attempts: {search_attempts}/{max_search_attempts}")
    
    if search_attempts < max_search_attempts:
        logger.info(f"   ➡️  Route: SEARCH (retry with attempt {search_attempts + 1})")
        return "search"
    else:
        logger.info("   ➡️  Route: FINAL_FALLBACK (max search attempts reached)")
        return "final_fallback"

# Build the graph
workflow = StateGraph(GraphState)

workflow.add_node("search", search_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("validate_answer", validate_answer_node)
workflow.add_node("fallback", fallback_node)
workflow.add_node("final_fallback", final_fallback_node)

workflow.add_edge(START, "search")
workflow.add_conditional_edges("search", route_after_search, {"summarize": "summarize", "fallback": "fallback"})
workflow.add_edge("summarize", "validate_answer")
workflow.add_conditional_edges("validate_answer", route_after_validate, {"end": END, "fallback": "fallback"})
workflow.add_conditional_edges("fallback", route_after_fallback, {"search": "search", "final_fallback": "final_fallback"})
workflow.add_edge("final_fallback", END)

graph = workflow.compile()

@app.post("/ask", tags=["RAG Operations"])
async def ask_endpoint(request: AskRequest):
    """Ask a Question with Full Fault Tolerance"""
    try:
        logger.info(f"❓ Question received: {request.question}")
        
        result = graph.invoke({
            "query": request.question,
            "documents": [],
            "answer": "",
            "search_attempts": 0,
            "answer_attempts": 0,
            "enough_results": False,
            "similarity_scores": [],
            "answer_quality_good": False
        })
        
        return {
            "question": request.question,
            "answer": result["answer"],
            "documents_used": len(result.get("documents", [])),
            "search_attempts": result.get("search_attempts", 0),
            "answer_attempts": result.get("answer_attempts", 0),
            "max_attempts": 3,
            "similarity_scores": [round(s, 3) for s in result.get("similarity_scores", [])[:5]],
            "answer_quality_good": result.get("answer_quality_good", False),
            "api_keys_available": len(API_KEYS),
            "fault_tolerance": "enabled (persistent keys + smart retry)",
            "langsmith_trace": "https://smith.langchain.com"
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# ============= SERVER STARTUP =============

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 RAG Q&A Support Bot v2.3.2 - Starting...")
    logger.info("=" * 60)
    logger.info(f"📊 LangSmith Project: {os.getenv('LANGCHAIN_PROJECT')}")
    logger.info(f"🔍 Tracing Enabled: {os.getenv('LANGCHAIN_TRACING_V2')}")
    logger.info(f"🌐 LangSmith Dashboard: https://smith.langchain.com")
    logger.info(f"📚 Swagger UI: http://localhost:8000/docs")
    logger.info(f"📖 ReDoc UI: http://localhost:8000/redoc")
    logger.info(f"🔑 API Keys: {len(API_KEYS)} loaded")
    logger.info(f"⚡ Rate Limit: {len(API_KEYS) * 15} req/min")
    logger.info(f"🛡️  Fault Tolerance: ENABLED (Persistent Keys + Smart Retry)")
    logger.info(f"📝 Log File: {log_filename}")
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
