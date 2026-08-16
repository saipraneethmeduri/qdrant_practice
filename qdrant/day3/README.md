# Day 3: Sparse Vectors & Hybrid Search

## 🎯 Agenda

Expand beyond dense embeddings to learn about sparse vectors and how combining both creates the most powerful retrieval systems.

**Learning objectives:**
- Understand sparse vs dense vector representations
- Learn how BM25 and sparse neural models work
- Implement sparse vector search in Qdrant
- Build hybrid search combining dense + sparse retrieval
- Understand when and why to use hybrid approaches

## 📋 Requirements

### Dependencies
- `qdrant-client` v1.13.4+
- `fastembed` or `splade` (for sparse embeddings)
- `sentence-transformers` (for dense embeddings)
- Additional libraries for specific notebooks
- `python-dotenv`

### Knowledge Prerequisites
- Completed Day 0, 1, and 2
- Understanding of dense vector embeddings
- Basic information retrieval concepts (helpful but not required)

## 📓 Notebooks in This Folder

### 1. `day3_1_Sparse_Vectors_in_Qdrant.ipynb`
**Introduction to sparse vectors in Qdrant**

**What it covers:**
- What sparse vectors are (mostly zeros, high-dimensional)
- Creating collections with `sparse_vectors_config`
- Understanding `indices` and `values` representation
- Inserting sparse vectors with `SparseVector`
- Running sparse similarity search (always uses dot product)
- Configuring the inverted index parameters

**Key concepts introduced:**
- **Sparse vectors**: Vectors with mostly zero values (e.g., `[0, 0.1, 0, 0, 0.3, 0, ...]`)
- **Indices**: Positions of non-zero values (stored as uint32, up to 4 billion dimensions)
- **Values**: The actual non-zero values (stored as floats)
- **Inverted index**: Data structure for efficient sparse vector comparison
- **Dot product**: The only distance metric for sparse vectors
- **Named vectors**: Storing both dense and sparse vectors for the same point

**Example:**
```python
SparseVector(
    indices=[1, 2, 3],      # Non-zero positions
    values=[0.2, -0.2, 0.2] # Non-zero values
)
```

### 2. `day3_2_Sparse_(Neural)_Text_Retrieval_in_Qdrant.ipynb`
**Deep dive into sparse neural retrieval**

**What it covers:**
- Classical sparse methods (BM25, TF-IDF)
- Neural sparse models (SPLADE, SPLADEv2, SPLADEv3)
- How neural models learn better sparse representations
- Generating sparse embeddings from text
- Comparing sparse vs dense retrieval performance
- Understanding the sparse vector advantages

**Key concepts introduced:**
- **BM25**: Classical term frequency-based sparse retrieval
- **TF-IDF**: Term frequency × inverse document frequency weighting
- **SPLADE**: Neural model that learns semantic sparse representations
- **Learned sparsity**: Neural networks predicting which dimensions matter
- **Interpretability**: Sparse vectors can be inspected (unlike dense)
- **Zero-shot transfer**: Sparse neural models generalize across domains

**Why neural sparse > classical sparse:**
- Learns semantic relationships (synonyms, context)
- Better handles out-of-vocabulary terms
- Captures query-document relevance more accurately

### 3. `day3_3_Introduction_to_Qdrant_Hybrid_Search_in_practice.ipynb`
**Building production hybrid search systems**

**What it covers:**
- Why hybrid search beats either approach alone
- Creating collections with both dense and sparse vectors
- Generating dual embeddings for documents
- Running hybrid queries with reciprocal rank fusion (RRF)
- Tuning fusion parameters (alpha weights)
- Benchmarking hybrid vs dense-only vs sparse-only

**Key concepts introduced:**
- **Hybrid search**: Combining multiple retrieval strategies
- **Reciprocal Rank Fusion (RRF)**: Algorithm for merging result lists
- **Fusion weights**: Balancing dense vs sparse contributions
- **Query performance**: When hybrid helps most
- **Prefusion**: Qdrant's optimization for server-side fusion

**The hybrid advantage:**
- Dense vectors: Capture semantic meaning, handle paraphrases
- Sparse vectors: Excel at exact term matches, rare keywords
- Together: Cover both semantic and lexical relevance

### 4. `day3_project.ipynb`
**Hands-on project:** Build your own hybrid search engine.

## ✅ What You've Accomplished

By completing Day 3, you've:

1. **Mastered sparse vectors** - Created, inserted, and queried sparse representations
2. **Understood inverted indexes** - How Qdrant efficiently compares sparse vectors
3. **Learned neural sparse models** - SPLADE and modern learned sparse retrieval
4. **Built hybrid search** - Combined dense + sparse for best-of-both-worlds retrieval
5. **Implemented RRF fusion** - Merged ranked lists from different vector types
6. **Benchmarked approaches** - Measured when hybrid beats single-vector search

## 💡 Key Takeaways & Benefits

### What You Learned

**Sparse vectors are fundamentally different:**
- **Dense**: All dimensions have values (e.g., 384 floats)
- **Sparse**: Mostly zeros, store only non-zero indices + values
- **Dimensionality**: Sparse can have millions of dimensions efficiently
- **Comparison**: Dot product only (negative values allowed)
- **Search**: Always exact (no HNSW, uses inverted index)

**When to use sparse vectors:**
- ✅ Exact keyword matching is important
- ✅ Domain-specific terminology
- ✅ Rare or technical terms
- ✅ Interpretability matters
- ✅ Combining with dense vectors (hybrid)

**When dense is enough:**
- ✅ Semantic understanding is primary goal
- ✅ Handling paraphrases and synonyms
- ✅ Cross-lingual search
- ✅ Embedding-based recommendations

**Hybrid search dominates when:**
- ✅ Both semantic and lexical relevance matter
- ✅ Diverse query types (some semantic, some keyword-based)
- ✅ Maximum recall is critical
- ✅ Production search engines with varied user behavior

### Real-World Performance

**Typical hybrid search improvements:**
- 5-15% better recall than dense-only
- 10-20% better recall than sparse-only
- Best on queries with mixed intent

**Example use cases:**
- **E-commerce search**: "red leather jacket" (color + material keywords + style semantics)
- **Code search**: Exact function names + semantic "what does it do"
- **Legal/medical**: Precise terminology + conceptual understanding
- **Documentation**: API names + natural language questions

### Why This Matters

**Modern retrieval systems need both:**
- Users ask semantic questions: "How do I upload files?"
- Users search exact terms: "upload_file_to_s3 function"
- Hybrid handles both gracefully

**Industry adoption:**
- Used by major search engines (Google, Bing, Elastic)
- Standard in enterprise RAG systems
- Best practice for high-stakes retrieval (medical, legal)

## 🚀 What's Next

You've completed the core Qdrant curriculum! You now know:
- ✅ Basic vector operations (Day 0)
- ✅ Semantic search with embeddings (Day 1)
- ✅ HNSW performance tuning (Day 2)
- ✅ Sparse vectors and hybrid search (Day 3)

**Advanced topics to explore:**
- **Multi-tenancy**: Sharding and partitioning strategies
- **Quantization**: Reducing memory with scalar/product quantization
- **Reranking**: Adding cross-encoder models after initial retrieval
- **Streaming**: Real-time index updates
- **Distributed deployment**: Running Qdrant in production clusters
- **Monitoring**: Metrics, logging, and observability

**Build real applications:**
- RAG chatbot with hybrid search
- Semantic documentation search
- Content recommendation engine
- Multi-modal search (text + images)
- Real-time anomaly detection

## 🔍 Deep Dive: How Hybrid Search Works

**Step-by-step hybrid query:**

1. **Dense search** generates top-k results with scores
   ```
   ["doc_A": 0.89, "doc_C": 0.85, "doc_B": 0.82]
   ```

2. **Sparse search** generates top-k results with scores
   ```
   ["doc_B": 15.3, "doc_D": 12.1, "doc_A": 10.8]
   ```

3. **RRF fusion** combines using reciprocal ranks:
   ```python
   score = α/(rank_dense + k) + β/(rank_sparse + k)
   
   doc_A: 0.5/(1+60) + 0.5/(3+60) = 0.0082 + 0.0079 = 0.0161
   doc_B: 0.5/(3+60) + 0.5/(1+60) = 0.0079 + 0.0082 = 0.0161
   doc_C: 0.5/(2+60) + 0.5/(∞)    = 0.0081 + 0       = 0.0081
   doc_D: 0.5/(∞)    + 0.5/(2+60) = 0      + 0.0081  = 0.0081
   ```

4. **Re-rank** by fused scores and return top results

**RRF benefits:**
- Robust to score scale differences
- Doesn't require calibration
- Simple and effective

## 🎓 Pro Tips from Day 3

1. **Always test hybrid vs single-vector** - Not every use case benefits equally
2. **Tune fusion weights** - Domain-specific optimal α/β ratios
3. **Sparse vectors are interpretable** - Debug by inspecting top indices
4. **Inverted index is efficient** - Millions of dimensions are fine
5. **Create payload indexes** - Still applies to hybrid collections
6. **Monitor query latency** - Hybrid = 2 searches + fusion overhead
7. **Consider query type distribution** - Optimize for your actual queries

## 🧪 Experiment Ideas

1. **Compare sparse models**: Test BM25 vs SPLADE vs SPLADEv3
2. **Vary fusion weights**: Try α=0.3/β=0.7 vs α=0.7/β=0.3
3. **Measure query types**: Track which queries benefit from hybrid
4. **Test cross-lingual**: How do sparse vectors handle multilingual content?
5. **Inspect activations**: Which terms activate which sparse dimensions?

## 🏆 Mastery Achieved

You've now mastered:
- **Three vector types**: Dense, sparse, and hybrid
- **Multiple search modes**: Semantic, lexical, and combined
- **Performance optimization**: From upload to query to fusion
- **Production patterns**: Building real-world retrieval systems

**The complete Qdrant stack:**
```
┌─────────────────────────────────────┐
│  Hybrid Search (RRF Fusion)         │
├─────────────────┬───────────────────┤
│  Dense HNSW     │  Sparse Inverted  │
│  (semantic)     │  (lexical)        │
├─────────────────┴───────────────────┤
│  Payload Filters & Indexes          │
├─────────────────────────────────────┤
│  Qdrant Storage & Optimization      │
└─────────────────────────────────────┘
```

You're ready to build production-grade vector search applications that rival commercial search engines. The techniques you've learned are used by companies building the next generation of AI-powered search and discovery!

## 📚 Further Learning

- **Qdrant Documentation**: https://qdrant.tech/documentation/
- **SPLADE Paper**: "SPLADE: Sparse Lexical and Expansion Model"
- **Hybrid Search Theory**: Information Retrieval textbooks (Manning et al.)
- **Vector Search at Scale**: Qdrant blog and case studies
- **Community**: Join Qdrant Discord for discussions and help

Congratulations on completing the Qdrant practice curriculum! 🎉
