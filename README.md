# RAG Assistant: AI Papers Q&A with Chunking Experiment

A Retrieval-Augmented Generation (RAG) pipeline built on 5 foundational AI papers, with a full evaluation framework to identify the optimal chunking configuration.

---

## What it does

Ask any question about the five most important papers in modern AI, the system retrieves the most relevant passages and generates a grounded, source-cited answer using Claude (Haiku).

**Papers indexed:**
- Attention Is All You Need — Vaswani et al. (2017)
- BERT — Devlin et al. (2018)
- GPT-3 — Brown et al. (2020)
- RAG — Lewis et al. (2020)
- LoRA — Hu et al. (2021)

---

## Architecture

```
src/
├── indexer.py          # Loads PDFs, splits into chunks, generates embeddings, stores in ChromaDB
├── query.py            # Embeds user query, retrieves relevant chunks, calls Claude API
├── RAG_evaluation.py   # Orchestrates the chunking experiment across 9 configurations
├── LLM_scoring.py      # LLM-as-a-judge evaluation (0–3 score per answer)
├── semantic_scoring.py # Cosine similarity evaluation (0–1 score per answer)
└── evaluation_set.json # 10 evaluation questions with gold standard answers
```

**Stack:** Python · LangChain · ChromaDB · HuggingFace Embeddings (`all-MiniLM-L6-v2`) · Anthropic Claude API (Haiku) · scikit-learn · pandas

---

## Chunking Experiment

The core contribution of this project is a systematic evaluation of 9 chunking configurations (3 CHUNK_SIZE × 3 TOP_K) against a 10-question evaluation set covering all 5 papers.

### Configurations tested

| Config | CHUNK_SIZE | TOP_K |
|--------|-----------|-------|
| A | 200 | 3 |
| B | 200 | 5 |
| C | 200 | 7 |
| D | 500 | 3 |
| E | 500 | 5 |
| F | 500 | 7 |
| G | 1000 | 3 |
| H | 1000 | 5 |
| I | 1000 | 7 |

### Evaluation framework

Each configuration was evaluated using two complementary methods:

**LLM-as-a-judge**: Claude scores each answer 0–3 based on correctness, completeness, and source citation. Maximum score per configuration: 30.

**Semantic similarity**: Cosine similarity between the RAG answer and the gold standard answer, both embedded with `all-MiniLM-L6-v2`. Score range: 0–1.

### Results

**LLM Scoring (total /30):**

| Config | CHUNK_SIZE | TOP_K | Total /30 |
|--------|-----------|-------|-----------|
| A | 200 | 3 | 14 |
| **B** | **200** | **5** | **19 ✓ winner** |
| C | 200 | 7 | 17 |
| D | 500 | 3 | 14 |
| E | 500 | 5 | 16 |
| F | 500 | 7 | 18 |
| G | 1000 | 3 | 16 |
| H | 1000 | 5 | 18 |
| I | 1000 | 7 | 18 |

**Semantic Scoring (average similarity 0–1):**

| Config | CHUNK_SIZE | TOP_K | Avg Similarity |
|--------|-----------|-------|---------------|
| A | 200 | 3 | 0.634 |
| B | 200 | 5 | 0.649 |
| C | 200 | 7 | 0.655 |
| D | 500 | 3 | 0.630 |
| E | 500 | 5 | 0.651 |
| F | 500 | 7 | 0.638 |
| G | 1000 | 3 | 0.620 |
| **H** | **1000** | **5** | **0.664 ✓ winner** |
| I | 1000 | 7 | 0.651 |

### Key findings

- **CHUNK_SIZE=200, TOP_K=5** scored highest on LLM evaluation (19/30): small, focused chunks provide more precise context for factual questions.
- **CHUNK_SIZE=1000, TOP_K=5** scored highest on semantic similarity (0.664): larger chunks contain richer vocabulary that aligns better with gold standard phrasing.
- Both methods agree that **TOP_K=5** is the optimal retrieval count: too few chunks miss relevant context, too many introduce noise.
- The optimal configuration is **CHUNK_SIZE=200, TOP_K=5**, prioritising factual precision over semantic coverage.

---

## How to run

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/marta-btez/rag-assistant.git
cd rag-assistant
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file in the root:

```
ANTHROPIC_API_KEY=your_key_here
```

### 3. Add the papers

Place the 5 PDFs in a `data/` folder. Papers are not included in this repo due to copyright.

### 4. Index the papers

```bash
python src/indexer.py
```

### 5. Run the evaluation experiment

```bash
python src/RAG_evaluation.py
```

---

## Author

Marta Benítez Aguilar — Aerospace Engineer & AI/Cloud enthusiast  
[LinkedIn](https://www.linkedin.com/in/marta-benitez-aguilar) · [GitHub](https://github.com/marta-btez)
