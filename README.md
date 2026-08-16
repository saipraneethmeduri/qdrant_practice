# Qdrant Practice

A comprehensive, hands-on learning journey through [Qdrant](https://qdrant.tech/)—from basic vector operations to production-grade semantic search systems.

## 🎯 What This Is

This repository documents a structured 6-day curriculum covering the complete Qdrant vector database stack:

- **Day 0**: Vector search fundamentals
- **Day 1**: Semantic search with real embeddings
- **Day 2**: HNSW performance tuning at scale (100K vectors)
- **Day 3**: Sparse vectors and hybrid search
- **Day 4**: Large-scale ingestion (400M vectors) and quantization
- **Day 5**: Universal Query API and multi-stage retrieval

Each day includes tutorial notebooks, hands-on projects, and comprehensive documentation explaining concepts, tradeoffs, and production patterns.

---

## Setup

### Prerequisites

- Python (see [.python-version](.python-version) for the pinned version)
- A Qdrant instance (cloud or local)

### Install dependencies

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e .
```

### Environment variables

Copy the example below into a `.env` file at the project root and fill in your values:

```env
QDRANT_URL=<your-qdrant-cluster-url>
QDRANT_API_KEY=<your-api-key>
```

---

## 📁 Project Structure

```
qdrant_practice/
├── .env                      # Local credentials (not committed)
├── .python-version           # Python 3.12+
├── README.md                 # This file
├── CLAUDE.md                 # Project context and collaboration guide
├── pyproject.toml            # Dependencies (uv-managed)
├── uv.lock                   # Locked dependencies
└── qdrant/
    ├── load_creds.py         # Reusable Qdrant connection helper
    ├── day0/                 # Vector Search Fundamentals
    │   ├── README.md         # Day 0 learning guide (3.9K)
    │   ├── day0_basic_vector_search.ipynb
    │   └── day0_project.ipynb
    ├── day1/                 # Semantic Search Engine
    │   ├── README.md         # Day 1 learning guide (6.7K)
    │   ├── day1_vector_search_dunds.ipynb
    │   └── day1_project.ipynb
    ├── day2/                 # HNSW Performance Tuning
    │   ├── README.md         # Day 2 learning guide (8.3K)
    │   ├── day2_hnsw_performance_tuning.ipynb
    │   └── day2_project.ipynb
    ├── day3/                 # Sparse Vectors & Hybrid Search
    │   ├── README.md         # Day 3 learning guide (11K)
    │   ├── day3_1_Sparse_Vectors_in_Qdrant.ipynb
    │   ├── day3_2_Sparse_(Neural)_Text_Retrieval_in_Qdrant.ipynb
    │   ├── day3_3_Introduction_to_Qdrant_Hybrid_Search_in_practice.ipynb
    │   └── day3_project.ipynb
    ├── day4/                 # Large-Scale & Quantization
    │   ├── README.md         # Day 4 learning guide (18K)
    │   ├── day4_large_scale_ingestion(don't_run_locally).ipynb
    │   └── day4_project.ipynb
    └── day5/                 # Universal Query API
        ├── README.md         # Day 5 learning guide (19K)
        ├── day5_universal_query_demo.ipynb
        └── day5_project.ipynb
```

**Each day folder contains:**
- Detailed README with concepts, examples, and pro tips
- Tutorial notebooks with explanations
- Hands-on project notebooks for practice

---

## 📚 Learning Path

### Day 0: Vector Search Fundamentals
> First steps into vector databases

- **What you'll learn**: Creating collections, inserting vectors, similarity search, distance metrics
- **Project**: Simple product search with 4D vectors
- **Key takeaway**: Understanding what vectors are and how they represent data

👉 [Day 0 README](qdrant/day0/README.md) for detailed concepts and guidance

### Day 1: Building a Semantic Search Engine
> Real embeddings and production patterns

- **What you'll learn**: Sentence transformers, text chunking strategies, named vectors, filtering
- **Project**: Movie recommendation system with 15 films and 357 chunks
- **Key takeaway**: Three chunking strategies (fixed, sentence, semantic) and their tradeoffs

👉 [Day 1 README](qdrant/day1/README.md) for implementation details

### Day 2: HNSW Performance Tuning
> Scaling to production size

- **What you'll learn**: HNSW algorithm, bulk upload strategies, payload indexes, filtering optimization
- **Project**: 100K DBpedia vectors on Qdrant Cloud with performance profiling
- **Key takeaway**: Filtered search without indexes = +1.6s, with indexes = -77ms vs baseline

👉 [Day 2 README](qdrant/day2/README.md) for tuning strategies

### Day 3: Sparse Vectors & Hybrid Search
> Combining semantic and lexical retrieval

- **What you'll learn**: Sparse vectors, BM25, SPLADE, RRF fusion, hybrid search architecture
- **Project**: Hybrid search engine combining dense + sparse retrieval
- **Key takeaway**: Hybrid search outperforms either approach alone by 5-15%

👉 [Day 3 README](qdrant/day3/README.md) for hybrid patterns

### Day 4: Large-Scale Ingestion & Quantization
> Enterprise-grade optimization

- **What you'll learn**: LAION-400M benchmark (⚠️ don't run locally), quantization (scalar, binary, product), rescoring pipelines
- **Project**: Recipe search with 4-32x compression and accuracy measurement
- **Key takeaway**: Quantization achieves 4.9x speedup with 100% accuracy retention (with rescoring)

👉 [Day 4 README](qdrant/day4/README.md) for production optimization

### Day 5: Universal Query API
> Multi-stage retrieval in one request

- **What you'll learn**: Prefetch, fusion, ColBERT multivectors, filter propagation, atomic queries
- **Project**: Research paper discovery (arXiv) and recommendation system
- **Key takeaway**: Universal Query reduces latency from 300ms → 80ms (3.75x faster)

👉 [Day 5 README](qdrant/day5/README.md) for advanced patterns

---

## 🎓 Key Concepts Covered

### Core Operations
- [x] Creating and configuring collections
- [x] Inserting vectors (upsert, batch upload)
- [x] Similarity search (COSINE, DOT, EUCLIDEAN)
- [x] Payload filtering and indexes
- [x] Named vectors (multiple vectors per point)

### Advanced Retrieval
- [x] Dense vectors (semantic understanding)
- [x] Sparse vectors (keyword matching)
- [x] ColBERT multivectors (token-level reranking)
- [x] Hybrid search with RRF fusion
- [x] Universal Query API (multi-stage retrieval)

### Performance Optimization
- [x] HNSW algorithm (m, ef_construct, ef_search)
- [x] Payload indexes (keyword, integer, float, datetime, bool)
- [x] Quantization (scalar, binary, product)
- [x] On-disk storage and float16 precision
- [x] Bulk upload strategies for large datasets

### Production Patterns
- [x] Text chunking strategies (fixed, sentence, semantic)
- [x] Embedding generation (sentence-transformers, FastEmbed)
- [x] Filter propagation in multi-stage queries
- [x] Oversampling and rescoring pipelines
- [x] Global filters for multi-tenant systems
- [x] Performance profiling and benchmarking

---

## 🛠️ Tech Stack

**Vector Database:**
- Qdrant Cloud (GCP europe-west6)
- Qdrant Python Client v1.15.0+

**Embeddings:**
- `sentence-transformers` (dense vectors)
- `fastembed` (dense, sparse, ColBERT)
- Models: all-MiniLM-L6-v2, SPLADE, ColBERTv2

**Data Processing:**
- `llama-index` (text chunking)
- `pandas`, `numpy` (data manipulation)
- `datasets` (HuggingFace datasets)

**Development:**
- Python 3.12+
- `uv` for package management
- Jupyter notebooks for experimentation

---

## 💡 Real-World Applications

The techniques learned here power:
- **Semantic search engines** (Google, Bing)
- **Recommendation systems** (Netflix, Spotify, Amazon)
- **RAG applications** (ChatGPT retrieval, enterprise chatbots)
- **Content discovery** (YouTube, TikTok, Pinterest)
- **E-commerce search** (visual + text search)
- **Knowledge bases** (Notion, Confluence)

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd qdrant_practice
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Qdrant credentials
   ```

4. **Start with Day 0**
   - Read [Day 0 README](qdrant/day0/README.md)
   - Open `qdrant/day0/day0_basic_vector_search.ipynb`
   - Follow along and experiment!

---

## 📊 Progress Tracking

| Day | Status | Notebooks | Concepts Mastered |
|-----|--------|-----------|-------------------|
| Day 0 | ✅ Complete | 2/2 | Vector basics, collections, similarity search |
| Day 1 | ✅ Complete | 2/2 | Embeddings, chunking, semantic search |
| Day 2 | ✅ Complete | 2/2 | HNSW, payload indexes, 100K scale |
| Day 3 | ✅ Complete | 4/4 | Sparse vectors, hybrid search, RRF fusion |
| Day 4 | ✅ Complete | 2/2 | Quantization, 400M scale patterns |
| Day 5 | ✅ Complete | 2/2 | Universal Query, multi-stage retrieval |

**Total**: 14 notebooks, 67K+ lines of documentation

---

## 🤝 Contributing

This is a personal learning repository, but feel free to:
- Open issues for questions or suggestions
- Fork and adapt for your own learning
- Share your own Qdrant experiments
- Suggest improvements to the documentation

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🙏 Acknowledgments

- [Qdrant](https://qdrant.tech/) for the excellent vector database
- [FastEmbed](https://github.com/qdrant/fastembed) for efficient embeddings
- The vector search research community
- Open-source contributors in the embedding and retrieval space

---

## 📖 Additional Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Discord Community](https://discord.gg/qdrant)
- [Vector Search Blog](https://qdrant.tech/blog/)
- [FastEmbed Documentation](https://qdrant.github.io/fastembed/)

---

**Happy learning! 🎉**
