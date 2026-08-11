# Qdrant Practice Project

## Purpose
Learning and experimenting with Qdrant vector database — building practical knowledge through hands-on exercises and performance experiments.

## Project Status
**Active Learning Phase**
- Day 0-2 completed: vector search fundamentals, HNSW algorithm, performance tuning
- Working through structured curriculum with daily notebooks
- Next topics: payload filtering, hybrid search, production optimization

## Stack / Dependencies
- **Language**: Python 3.12+
- **Vector DB**: Qdrant Cloud (GCP europe-west6)
- **Client**: `qdrant-client` v1.13.4+
- **Embeddings**: `sentence-transformers` v5.7.0+
- **Framework**: `llama-index` v0.14.23+ (optional high-level abstractions)
- **Package Manager**: `uv` (modern, fast alternative to pip)
- **Environment**: `.env` file with `QDRANT_URL` and `QDRANT_API_KEY`

## Key Decisions & Context

### Architecture
- Using Qdrant Cloud rather than self-hosted for simplicity during learning phase
- Credentials loaded via `python-dotenv` from local `.env` file
- `load_creds.py` provides reusable connection helper with health check

### Embedding Strategy
- `sentence-transformers` chosen for flexibility and local execution
- Can experiment with different models (all-MiniLM-L6-v2, etc.) as needed

### Learning Path
1. **Day 0**: Basic vector operations, collection setup, simple queries
2. **Day 1**: Vector search deep dive, understanding similarity metrics
3. **Day 2**: HNSW indexing, performance parameters (`m`, `ef_construct`)
4. **Next**: Payload filtering, hybrid search, multi-vector scenarios

### HNSW Performance Insights
- **`m`** (max connections per node): Higher = better recall, slower build, more memory
- **`ef_construct`** (candidates during build): Higher = better quality, slower indexing
- Defaults (m=16, ef_construct=100) are good starting points

## Collaboration Style
**Act as a tutor and guide:**
- Explain concepts when asked or when the user appears stuck
- Provide examples and analogies to clarify Qdrant/vector search concepts
- Suggest next experiments or areas to explore
- **Do NOT make code changes** unless explicitly requested
- Let the user write code and experiment independently — only intervene when asked

## Project Structure
```
qdrant_practice/
├── qdrant/                           # All experiments live here
│   ├── load_creds.py                 # Reusable Qdrant client setup
│   ├── day0_*.ipynb                  # Day 0 notebooks
│   ├── day1_*.ipynb                  # Day 1 notebooks
│   └── day2_*.ipynb                  # Day 2 notebooks
├── .env                              # Qdrant credentials (not committed)
├── pyproject.toml                    # Dependency manifest
├── uv.lock                           # Locked dependencies
└── README.md                         # User-facing documentation
```

## Notes
- This is a **personal learning project** — no production constraints
- Update this file as new concepts are explored or architectural decisions are made
- When adding new notebooks, update the experiments table in README.md
