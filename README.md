# 🤖 AstraRAG – Agentic RAG Chatbot

**AstraRAG** is a full-stack, containerized Agentic Retrieval-Augmented Generation (RAG) chatbot. Unlike standard RAG pipelines that blindly retrieve documents for every query, AstraRAG uses **CrewAI** as an intelligent orchestrator to decide when to query a **LlamaIndex** + **ChromaDB** vector store, synthesizing accurate, source-backed answers via **Groq** LLMs.

---

## 🏗️ System Architecture

```text
[User] 
  │
  ▼
[Streamlit Frontend (Port 8501)]
  │  (HTTP POST /chat/answer)
  ▼
[FastAPI Backend (Port 8000)]
  │
  ▼
[CrewAI Agent ("Question Answer Agent")]
  ├──► Decides if external knowledge is needed
  └──► Invokes Custom Tool: [rag_query_tool]
         │
         ▼
       [LlamaIndex Query Engine] ──► [HuggingFace Embeddings (BAAI/bge-small-en)]
         │
         ▼
       [ChromaDB Persistent Vector Store]
```

### Key Features
* **Agentic Routing:** Powered by CrewAI, the agent evaluates user intent, utilizes tools only when necessary, and structures outputs using Pydantic (`AnswerStructure`) to provide transparent answers, citations, and reasoning.
* **Custom RAG Pipeline:** Built with LlamaIndex and ChromaDB for local, persistent vector storage and semantic retrieval.
* **Decoupled Full-Stack Architecture:** FastAPI handles asynchronous API requests while Streamlit provides a clean, interactive chat UI.
* **Production-Ready Containerization:** Multi-service Docker setup (`start.sh`) with automated document ingestion, layer caching via `uv`, and auto-restart resilience (`--restart=always`) for AWS EC2 deployments.
* **Quantitative Evaluation:** Benchmarked using the **RAGAS** framework to validate retrieval quality and minimize hallucinations.

---

## 🛠️ Tech Stack

* **Agent Orchestration:** CrewAI
* **RAG & Indexing:** LlamaIndex
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`BAAI/bge-small-en`)
* **LLM Inference:** Groq API (`openai/gpt-oss-120b`)
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit
* **Evaluation:** RAGAS, HuggingFace Datasets
* **DevOps & Cloud:** Docker, AWS EC2 (Ubuntu)

---

## 📊 RAGAS Evaluation Metrics

The retrieval and generation pipeline was evaluated offline using the **RAGAS** framework (`ragas_llm_eval.py`) with local HuggingFace embeddings and sequential execution to respect API rate limits:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Answer Relevancy** | **0.97** | Measures how directly and completely the generated answer addresses the user's prompt. |
| **Context Precision** | **0.67** | Evaluates the signal-to-noise ratio of the chunks retrieved from ChromaDB. |

---

## 📂 Project Structure

```text
agentic-ai-project/
├── docs_dir/                     # Source documents (PDFs, TXTs) for ingestion
├── doc_vector_store/             # Persisted ChromaDB vector database
├── src/
│   ├── agents_src/
│   │   ├── agents/               # CrewAI agent definitions
│   │   ├── config/               # Pydantic environment settings
│   │   ├── llm/                  # LLM initialization and model configs
│   │   ├── tasks/                # CrewAI task & Pydantic output schemas
│   │   ├── tools/                # Custom LlamaIndex RAG retrieval tool
│   │   └── crew.py               # Crew assembly
│   ├── backend_src/              # FastAPI application & endpoints
│   ├── frontend_src/             # Streamlit web interface
│   └── rag_doc_ingestion/        # Document chunking & embedding ingestion script
├── .dockerignore
├── .gitignore
├── Dockerfile                    # Container build instructions (using uv)
├── ragas_llm_eval.py             # Offline RAGAS evaluation benchmark script
├── requirements.txt              # Production dependencies
├── start.sh                      # Entrypoint script running ingestion, FastAPI & Streamlit
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* Docker Desktop (for containerized deployment)
* A free API key from [Groq Console](https://console.groq.com/)

### 1. Clone the Repository
```bash
git clone https://github.com/masudkazi94/agentic-ai-project.git
cd agentic-ai-project
```

### 2. Environment Configuration
Create a `.env` file in the root directory and configure the following variables:
```env
GROQ_API_KEY=your_groq_api_key_here
DOCUMENTS_DIR=docs_dir
VECTOR_STORE_DIR=doc_vector_store
COLLECTION_NAME=document_collection
MODEL_NAME=groq/openai/gpt-oss-120b
MODEL_TEMPERATURE=0.0
CHAT_ENDPOINT_URL=http://localhost:8000/chat/answer
```

---

## 💻 Running Locally (Without Docker)

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

3. **Ingest documents into ChromaDB:**
   Place your `.pdf` or `.txt` files inside `docs_dir/` and run:
   ```bash
   python -m src.rag_doc_ingestion.ingest_docs
   ```

4. **Start the Backend (Terminal 1):**
   ```bash
   uvicorn src.backend_src.main:app --host 0.0.0.0 --port 8000
   ```

5. **Start the Frontend (Terminal 2):**
   ```bash
   streamlit run src/frontend_src/app.py --server.port 8501 --server.address 0.0.0.0
   ```
   Access the UI at `http://localhost:8501`.

---

## 🐳 Running with Docker & AWS EC2

The project includes a `Dockerfile` optimized with `uv` for fast dependency resolution and a `start.sh` script that automatically runs document ingestion before launching both Uvicorn and Streamlit.

1. **Build the Docker Image:**
   ```bash
   docker build -t agentic-rag-chatbot:latest .
   ```

2. **Run the Container (with Auto-Restart for Cloud Resilience):**
   ```bash
   docker run -d \
     --restart=always \
     -p 8000:8000 \
     -p 8501:8501 \
     -e GROQ_API_KEY=your_groq_api_key_here \
     --name agentic-rag-container \
     agentic-rag-chatbot:latest
   ```

3. **Access the Application:**
   * **Local:** `http://localhost:8501`
   * **AWS EC2:** `http://<YOUR-EC2-PUBLIC-IP>:8501` *(Ensure inbound TCP ports `8501` and `8000` are open in your EC2 Security Group).*

---

## 🧪 Running RAGAS Evaluation

To reproduce the evaluation metrics locally without bloating the production Docker image:

```bash
pip install ragas llama-index-llms-groq langchain-community sentence-transformers datasets
python ragas_llm_eval.py
```

---

## ⚠️ Known Limitations & Engineering Notes

* **API Rate Limits:** When deployed using Groq's free tier (8,000 Tokens Per Minute limit), rapid consecutive queries may trigger a `429 RateLimitError`. To mitigate this, chunk sizes (`512`), `similarity_top_k` (`2`), `max_iter` (`2`), and `stream=False` are configured to minimize token overhead. Wait ~15–20 seconds between complex queries when testing on the free tier.