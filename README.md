# 🛡️ Fault-Tolerant RAG Q&A Support Bot

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-v0.1-orange.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-v0.1-blueviolet.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

A robust, production-grade **Retrieval-Augmented Generation (RAG)** system designed for resilience. Unlike standard RAG demos, this bot includes a self-correcting brain that detects failures, rotates API keys to bypass rate limits, and validates answer quality before responding.

Built with **FastAPI**, **LangGraph**, **ChromaDB**, and **Google Gemini Models**.

---

## 🚀 Key Features

### 🧠 **Smart Fault Tolerance**
- **Circular API Key Rotation:** Automatically rotates between multiple Google API keys when rate limits (`429`) or quota errors are detected.
- **Smart Retry Logic:** If a search yields low-quality results, the system automatically expands the search scope and retries up to 3 times.
- **Hallucination Prevention:** The LLM is instructed to refuse questions not found in the source text, preventing misinformation.

### ⚙️ **Architecture**
- **LangGraph Workflow:** State-machine based processing (`Search` → `Summarize` → `Validate` → `Fallback`).
- **Vector Database:** Local persistence using **ChromaDB** with `gemini-embedding-001`.
- **LLM Engine:** Powered by `gemini-1.5-flash` (or compatible) for high-speed inference.
- **Containerization:** Fully Dockerized for easy deployment.

---

## 📂 Project Structure

```text
RAG Q&A SUPPORT BOT/
├── .langgraph_api/      # LangGraph configuration
├── chroma_db/           # Persisted vector database storage
├── logs/                # Application logs
├── venv/                # Virtual environment
├── .env                 # Environment variables (API Keys)
├── .gitignore           # Git ignore rules
├── docker-compose.yml   # Docker services configuration
├── Dockerfile           # Docker image build instructions
├── langgraph.json       # LangGraph export/config
├── main.py              # Main FastAPI application & RAG logic
├── requirements.txt     # Python dependencies
├── test_api.py          # Script to verify API keys & Models
├── test_rag.bat         # Automated comprehensive test suite (Windows)
├── test_quick.bat       # Quick smoke test script
└── test_stress.bat      # Load testing script
```

---

## 🛠️ Setup & Installation

### Option 1: Docker (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/fault-tolerant-rag-bot.git
    cd fault-tolerant-rag-bot
    ```

2.  **Create a `.env` file:**
    ```ini
    # Add your Google Gemini API Keys
    GOOGLE_API_KEY_1=AIzaSy...
    GOOGLE_API_KEY_2=AIzaSy...
    GOOGLE_API_KEY_3=AIzaSy...
    
    # LangChain Tracing (Optional)
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=lsv2_...
    LANGCHAIN_PROJECT=RAG-QA-Bot
    ```

3.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    The API will be available at `http://localhost:8000`.

### Option 2: Local Development

1.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # Windows (Git Bash)
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server:**
    ```bash
    python main.py
    ```

---

## 📖 API Usage

### 1. Crawl & Index Content
Submit a URL to be scraped, chunked, and embedded into the vector database.

```bash
curl -X POST "http://localhost:8000/crawl" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"}'
```

### 2. Ask a Question
Query the indexed knowledge base. The system will auto-retry if the initial answer is poor.

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the ethical risks of AI?"}'
```

---

## 🧪 Testing Suite

The project includes robust Windows batch scripts for automated testing:

*   **`test_api.py`**: Validates that your API keys and Models (`gemini-embedding-001`) are reachable.
*   **`test_rag.bat`**: Runs a comprehensive scenario test:
    *   ✅ **Happy Path:** Simple definition questions.
    *   🔄 **Retry Logic:** Hard questions requiring multiple search attempts.
    *   🛑 **Refusal:** Questions unrelated to the text (e.g., "What is Quantum Computing?" when only AI docs are indexed).

To run tests (Windows):
```cmd
test_rag.bat
```

---

## 📊 Monitoring

- **Logs:** Detailed application logs are stored in the `/logs` directory.
- **LangSmith:** If configured in `.env`, full execution traces (latency, token usage, retry steps) are sent to LangSmith.

## 📄 License

MIT License. See [LICENSE](LICENSE) file for details.
