# rag-based-sales-assistant

A local RAG-powered AI assistant for vehicle dealership CRM operations. Uses **Llama 3.2** (via Ollama) + **ChromaDB** vector search + **pandas** structured queries to answer natural language questions about enquiries, appointments, and customer feedback.

---

## 📁 Project Structure

```
sales-crm-ai/
├── data/
│   ├── sales_enquiry_dataset.csv       # Customer enquiries & lead data
│   ├── sales_appointment_dataset.csv   # Scheduled appointments
│   └── sales_feedback_dataset.csv      # Customer feedback & ratings
│
├── model.py        # SalesLLM — main LLM interface (generate, analyze, chat)
├── prompts.py      # SalesPrompts — all prompt templates
├── rag.py          # SalesRAG — data loading, vector indexing, structured queries
├── vector_db.py    # VectorDB — ChromaDB wrapper with Ollama embeddings
├── test.py         # Interactive CLI chatbot entry point
├── config.py       # ⚙️ Central config (paths, model names, settings)
└── requirements.txt
```

---

## ⚙️ Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| [Ollama](https://ollama.com) | Latest |
| Llama 3.2 model | `ollama pull llama3.2` |
| nomic-embed-text | `ollama pull nomic-embed-text` |

---

## 🚀 Setup & Installation

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd sales-crm-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull required Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 5. Place your data files

Put the three CSV files inside a `data/` folder at the project root:

```
data/
├── sales_enquiry_dataset.csv
├── sales_appointment_dataset.csv
└── sales_feedback_dataset.csv
```

### 6. Load data into the vector database

Run this **once** (or whenever your CSV data changes):

```bash
python rag.py
```

This processes all three CSVs and stores embeddings in `./chroma_db/`.

### 7. Start the AI assistant

```bash
python test.py
```

---

## 💬 Usage Examples

Once the assistant is running, you can ask:

```
👤 You: How many appointments are scheduled for today?
👤 You: Show me the enquiries for Honda City
👤 You: How many total bookings do we have?
👤 You: What appointments are scheduled for 22/04/2026?
👤 You: Summarize the latest customer feedback
```

Type `exit` or `quit` to close the assistant.

---

## 🧠 How It Works

```
User Query
    │
    ▼
SalesLLM.chat_with_data()
    │
    ├──► Structured Path (pandas)
    │     • Date-based appointment lookups
    │     • Count queries (how many booked, total appointments, etc.)
    │     • Vehicle-specific summaries
    │
    └──► Semantic Path (ChromaDB + nomic-embed-text)
          • Fuzzy similarity search across all CRM records
          • Returns top-6 relevant document chunks
    │
    ▼
SalesPrompts.general_chat()  ← builds the final prompt with context
    │
    ▼
Llama 3.2 (Ollama)  ← streams the response token by token
```

