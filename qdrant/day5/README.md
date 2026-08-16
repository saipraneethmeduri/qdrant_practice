# Day 5: Universal Query API & Multi-Stage Retrieval

## 🎯 Agenda

Master Qdrant's most powerful feature: the Universal Query API. Build atomic, multi-stage retrieval pipelines that combine dense, sparse, and multivector search with fusion and reranking—all in a single request.

**Learning objectives:**
- Understand the Universal Query architecture and execution model
- Combine dense, sparse, and ColBERT vectors in one query
- Implement multi-stage retrieval pipelines (prefetch → fusion → rerank)
- Use global filters that propagate through all stages
- Build production-grade recommendation systems
- Optimize query performance with strategic prefetching
- Apply ColBERT multivectors for fine-grained reranking

## 📋 Requirements

### Dependencies
- `qdrant-client` v1.15.0+ (Universal Query support)
- `fastembed` v0.7.0+ (dense, sparse, ColBERT embeddings)
- `arxiv` (for research paper demo)
- `sentence-transformers` (alternative embedding library)
- `python-dotenv`

### Knowledge Prerequisites
- Completed Day 0-4
- Understanding of dense and sparse vectors (Day 3)
- Familiarity with hybrid search and RRF fusion (Day 3)
- Basic knowledge of recommendation systems (helpful)

### Qdrant Setup
- Qdrant Cloud or local instance v1.15.0+
- Universal Query API is available in recent versions

## 📓 Notebooks in This Folder

### 1. `day5_universal_query_demo.ipynb` ✅

**Real-world demo: Research paper discovery with arXiv**

**What it covers:**

**Part 1: Multi-Vector Collection Setup**
- Creating a collection with three vector types:
  - **Dense vectors** (384D): Semantic understanding via `all-MiniLM-L6-v2`
  - **Sparse vectors**: Keyword matching via SPLADE
  - **ColBERT multivectors** (128D): Token-level reranking
- Creating payload indexes for efficient filtering
- Understanding when to use each vector type

**Part 2: Real Data Ingestion**
- Fetching 50 research papers from arXiv API
- Query: "transformer AND multimodal"
- Generating three embeddings per paper
- Automated research area categorization
- Uploading with FastEmbed integration

**Part 3: Universal Query Architecture**

The magic of Universal Query—a single request that does:

```python
# Stage 1: Parallel prefetch (dense + sparse)
hybrid_query = [
    Prefetch(query=dense_vector, using="dense", limit=100),
    Prefetch(query=sparse_vector, using="sparse", limit=100),
]

# Stage 2: Fusion with RRF
fusion_query = Prefetch(
    prefetch=hybrid_query,
    query=FusionQuery(fusion=Fusion.RRF),
    limit=100,
)

# Stage 3: ColBERT reranking
response = client.query_points(
    prefetch=fusion_query,
    query=colbert_vector,
    using="colbert",
    query_filter=global_filter,  # Propagates everywhere!
    limit=10,
)
```

**Part 4: Global Filter Propagation**
- Define business rules once at the top level
- Filters automatically propagate to all prefetch stages
- No need to repeat filter logic
- Example filters: research area, open access, date range, citations, impact score

**Part 5: Results Analysis**
- Top 10 research papers ranked by relevance
- Multi-stage scoring: semantic + keyword + token-level
- Demonstrates how each stage contributes to final ranking

**Key concepts introduced:**
- **Universal Query API**: Single atomic query for multi-stage retrieval
- **Prefetch**: Parallel candidate retrieval stages
- **Fusion**: Combining results from multiple sources (RRF)
- **Multivector reranking**: Fine-grained token-level scoring
- **Global filter propagation**: DRY (Don't Repeat Yourself) filtering
- **Atomic execution**: All stages execute as one transaction
- **ColBERT**: Late interaction model for precise reranking
- **MAX_SIM comparator**: How multivectors compute similarity

**Why Universal Query matters:**
- ✅ **Single request** instead of multiple round-trips
- ✅ **Atomic execution** with consistent filtering
- ✅ **Reduced latency** through server-side fusion
- ✅ **Cleaner code** with no manual result merging
- ✅ **Optimized performance** with intelligent prefetching

### 2. `day5_project.ipynb` ✅

**Build a production recommendation system**

**What it covers:**

**Part 1: Recommendation Architecture**
- Designing a hybrid collection for content recommendations
- Three vector types for different aspects of relevance:
  - Dense: Semantic content understanding
  - Sparse: Exact keyword/title matching
  - ColBERT: Fine-grained description matching
- Configuring HNSW: Disable for ColBERT (reranking only)

**Part 2: Sample Data Creation**
- Movie/content catalog with rich metadata
- Business attributes: category, user_segment, popularity
- Quality metrics: rating, release_date
- Genre tags and descriptions

**Part 3: Embedding Generation**
- Using FastEmbed for three embedding types
- Batch processing for efficiency
- Understanding parallel embedding generation
- Alternative: Local inference syntax with Qdrant

**Part 4: Building Recommendation Queries**

**User profile-based filtering:**
```python
global_filter = Filter(
    must=[
        FieldCondition(key="category", match="movie"),
        FieldCondition(key="user_segment", match="premium"),
        FieldCondition(key="release_date", range=DatetimeRange(gte=...)),
        FieldCondition(key="popularity_score", range=Range(gte=0.7)),
    ]
)
```

**Multi-stage retrieval pipeline:**
1. Dense semantic search (user intent)
2. Sparse keyword matching (explicit preferences)
3. RRF fusion of candidates
4. ColBERT reranking (final scoring)
5. Global filters applied at every stage

**Part 5: Production Service Implementation**
- `build_recommendation_filter()`: Dynamic filter construction
- `get_recommendations()`: Complete recommendation function
- User profile and preference handling
- Error handling and edge cases

**Part 6: Testing & Validation**
- Sample user profiles with preferences
- Testing different recommendation scenarios
- Validating filter propagation
- Performance measurement

**Key concepts introduced:**
- **Recommendation system architecture**: Multi-signal ranking
- **User profiling**: Segments, preferences, history
- **Business rules as filters**: Content policy, licensing, availability
- **Dynamic query construction**: Building queries from user input
- **Service patterns**: Reusable recommendation functions
- **Multi-tenant filtering**: Segment-based access control

**Production patterns demonstrated:**
- Separating filter logic from query logic
- Building composable filter functions
- Handling missing or incomplete user profiles
- Graceful degradation (fewer signals when data missing)
- Performance-conscious prefetch limits

## ✅ What You've Accomplished

By completing Day 5, you've:

1. **Mastered Universal Query API** - The most powerful retrieval pattern in Qdrant
2. **Built multi-stage pipelines** - Prefetch → Fusion → Reranking in one request
3. **Implemented global filters** - DRY filtering that propagates automatically
4. **Used ColBERT multivectors** - Fine-grained token-level reranking
5. **Integrated real data** - Worked with arXiv research papers
6. **Built a recommendation system** - Production-ready multi-signal ranking
7. **Applied best practices** - Composable, maintainable query patterns

## 💡 Key Takeaways & Benefits

### What You Learned

**The Universal Query execution model:**

```
┌─────────────────────────────────────────────────┐
│  Final Stage: ColBERT Reranking                 │
│  (query=multivector, using="colbert")           │
├─────────────────────────────────────────────────┤
│  Fusion Stage: RRF                              │
│  (query=FusionQuery(fusion=RRF))                │
├──────────────────────┬──────────────────────────┤
│  Prefetch: Dense     │  Prefetch: Sparse        │
│  (semantic search)   │  (keyword search)        │
├──────────────────────┴──────────────────────────┤
│  Global Filter (propagates to all stages)       │
└─────────────────────────────────────────────────┘
```

**Why this is revolutionary:**

**Before Universal Query:**
```python
# Multiple requests, manual merging, inconsistent filtering
dense_results = client.search(query=dense_vec, filter=f1)
sparse_results = client.search(query=sparse_vec, filter=f2)  # Oops, different filter!
fused = rrf_merge(dense_results, sparse_results)  # Manual merging
reranked = colbert_rerank(fused, colbert_model)  # Another request
# 4+ round trips, error-prone, slow
```

**With Universal Query:**
```python
# One request, server-side fusion, automatic filter propagation
response = client.query_points(
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=100),
        Prefetch(query=sparse_vec, using="sparse", limit=100),
    ],
    query=colbert_vec,
    using="colbert",
    query_filter=global_filter,  # Applied everywhere!
    limit=10,
)
# 1 request, atomic, fast, correct
```

**Benefits:**
- **10-100x latency reduction** (no network round trips)
- **Guaranteed consistency** (same filter everywhere)
- **Simplified code** (no manual merging)
- **Server-side optimization** (intelligent prefetching)
- **Atomic execution** (all stages or none)

**ColBERT multivectors explained:**

Traditional dense vectors: one vector per document
```
Document → [0.1, 0.3, ..., 0.5]  (384 numbers)
```

ColBERT multivectors: one vector per token
```
Document = "The quick brown fox"
  "The"   → [0.1, 0.2, ..., 0.3]  (128 numbers)
  "quick" → [0.4, 0.1, ..., 0.7]  (128 numbers)
  "brown" → [0.2, 0.6, ..., 0.1]  (128 numbers)
  "fox"   → [0.8, 0.3, ..., 0.4]  (128 numbers)
```

**Scoring:** MAX_SIM computes maximum similarity between any query token and any document token
- More fine-grained than dense vectors
- Captures token-level relevance
- Great for reranking top candidates
- Too expensive for first-stage retrieval (use prefetch)

**Filter propagation rules:**

```python
# Filter defined at top level
global_filter = Filter(must=[...])

# Automatically propagates to:
response = client.query_points(
    prefetch=[
        Prefetch(...),  # ← Applied here
        Prefetch(...),  # ← And here
    ],
    query=...,
    query_filter=global_filter,  # ← Defined once
)
```

**Override for specific stages:**
```python
Prefetch(
    query=...,
    query_filter=specific_filter,  # Overrides global
    limit=100
)
```

### Why This Matters

**Modern retrieval is multi-stage:**
1. **Fast, approximate filtering** → 1M to 10K candidates
2. **Semantic + keyword fusion** → 10K to 100 candidates
3. **Precise reranking** → 100 to 10 final results

Each stage trades off different metrics:
- Stage 1: Speed over precision (HNSW + quantization)
- Stage 2: Coverage over speed (hybrid fusion)
- Stage 3: Precision over speed (expensive rerankers)

**Universal Query makes this practical:**
- No manual orchestration
- Optimal performance (server-side)
- Production-ready patterns

**Real-world applications:**
- **Search engines**: Google, Bing (multi-stage ranking)
- **Recommendation systems**: Netflix, Spotify (collaborative + content)
- **E-commerce**: Amazon (search + personalization)
- **Content discovery**: YouTube, TikTok (multi-signal ranking)
- **Enterprise search**: Confluence, Notion (semantic + keyword)
- **RAG systems**: Advanced document retrieval

### Performance Insights

**Typical latency breakdown:**

Single-stage search: ~50ms
```
50ms: HNSW search with filters
```

Manual multi-stage (old way): ~300ms
```
50ms: Dense search
50ms: Sparse search
100ms: Network overhead (2 round trips)
50ms: Client-side RRF fusion
50ms: Reranking request
```

Universal Query (new way): ~80ms
```
25ms: Parallel prefetch (dense + sparse)
10ms: Server-side RRF fusion
35ms: ColBERT reranking
10ms: Network overhead (1 round trip)
```

**Speedup: 3.75x faster! 🚀**

## 🚀 What's Next

You've now mastered the complete Qdrant stack from basics to advanced production patterns!

**Advanced techniques to explore:**
- **Multi-modal search**: Combining text + image + audio vectors
- **Learned fusion**: ML models instead of RRF
- **Cross-encoder reranking**: BERT-based rerankers
- **User embeddings**: Personalized search with user vectors
- **Context-aware search**: Session history in queries
- **A/B testing**: Comparing retrieval strategies
- **Monitoring**: Tracking relevance metrics in production

**System design patterns:**
- **Caching layers**: Redis for hot queries
- **Rate limiting**: Per-user query quotas
- **Query optimization**: Adaptive prefetch limits
- **Fallback strategies**: Graceful degradation
- **Multi-tenancy**: Per-tenant collections and filters

**Build advanced projects:**
- Multi-modal e-commerce search (text + images)
- Personalized news feed with real-time updates
- Conversational search with context tracking
- Cross-lingual semantic search
- Real-time content moderation pipeline
- Federated search across multiple sources

## 🔍 Deep Dive: Universal Query Execution

**How Qdrant executes a Universal Query:**

```python
response = client.query_points(
    collection_name="items",
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=100),
        Prefetch(query=sparse_vec, using="sparse", limit=100),
    ],
    query=colbert_vec,
    using="colbert",
    query_filter=global_filter,
    limit=10,
)
```

**Step-by-step execution:**

1. **Filter preparation** (1ms)
   - Parse global_filter
   - Prepare to apply at each stage

2. **Parallel prefetch** (25ms) - Runs concurrently!
   - Thread 1: HNSW search on "dense" with filter → 100 candidates
   - Thread 2: Inverted index search on "sparse" with filter → 100 candidates
   - Both use quantization for speed

3. **Fusion stage** (10ms)
   - Collect 200 candidates (100 from each)
   - Apply RRF formula: `score = 1/(k + rank)`
   - Deduplicate by ID
   - Sort by fused score
   - Take top 100 for reranking

4. **Reranking stage** (35ms)
   - Load full-precision ColBERT multivectors for top 100
   - Compute MAX_SIM with query multivector
   - Sort by ColBERT score
   - Apply limit=10

5. **Response assembly** (5ms)
   - Fetch payloads for top 10
   - Serialize to JSON
   - Return to client

**Total: ~80ms**

Compare to sequential execution: 50ms + 50ms + 10ms + 35ms = 145ms
**Parallelization saves 65ms!**

## 🎓 Pro Tips from Day 5

1. **Use prefetch liberally** - Qdrant parallelizes intelligently
2. **Set generous prefetch limits** - Over-fetch before reranking (100-200 candidates)
3. **Keep ColBERT for final stage** - Too expensive for first-stage retrieval
4. **Define filters once** - Let propagation handle the rest
5. **Override filters sparingly** - Only when stage-specific logic needed
6. **Monitor prefetch hit rates** - Are candidates making it to final results?
7. **Tune RRF weights** - Default is equal, but you can adjust
8. **Batch user queries** - Amortize model loading overhead
9. **Cache embeddings** - Store user profile embeddings
10. **Profile your queries** - Identify bottleneck stages

## 🧪 Experiment Ideas

1. **Compare fusion strategies**: RRF vs weighted sum vs learned fusion
2. **Vary prefetch limits**: 50 vs 100 vs 200 - impact on accuracy and speed
3. **Test without reranking**: Is ColBERT worth the overhead for your data?
4. **A/B test vector types**: Dense-only vs hybrid vs full 3-stage
5. **Measure filter impact**: Query with/without filters at each stage
6. **Profile token counts**: How do ColBERT multivector sizes affect latency?
7. **Test edge cases**: Empty results at prefetch stage, filter eliminating all candidates
8. **Benchmark parallelization**: Measure actual speedup from parallel prefetch

## 🏆 Mastery Achieved

You've now mastered the complete Qdrant journey:

**Day 0**: First vectors → Basic similarity search
**Day 1**: Semantic search → Real embeddings and chunking
**Day 2**: HNSW tuning → Production-scale performance
**Day 3**: Hybrid search → Dense + sparse fusion
**Day 4**: Quantization → Memory optimization at scale
**Day 5**: Universal Query → Multi-stage retrieval pipelines ✨

**You can now build:**
- World-class semantic search engines
- Production-grade recommendation systems
- Multi-modal discovery platforms
- Enterprise knowledge bases
- Real-time personalization systems
- Advanced RAG applications

**Your skills include:**
- Vector database architecture and operations
- Embedding strategy and optimization
- Multi-stage retrieval pipelines
- Performance tuning and scaling
- Production deployment patterns
- Modern search system design

## 📚 Further Learning

**Qdrant Resources:**
- [Universal Query Documentation](https://qdrant.tech/documentation/concepts/search/)
- [ColBERT Integration Guide](https://qdrant.tech/documentation/fastembed/fastembed-multivector/)
- [FastEmbed Documentation](https://qdrant.github.io/fastembed/)
- [RRF Fusion Explanation](https://qdrant.tech/documentation/concepts/explore/#fusion)

**Research Papers:**
- "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction" (Khattab & Zaharia, 2020)
- "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking" (Formal et al., 2021)
- "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (Cormack et al., 2009)

**Advanced Topics:**
- Cross-encoder reranking models
- Learned-to-rank systems
- Query understanding and expansion
- Personalization in retrieval
- Multi-modal embeddings

## 🎉 Congratulations!

You've completed the entire Qdrant practice curriculum! From basic vector operations to sophisticated multi-stage retrieval pipelines, you've gained production-ready skills that power modern AI applications.

**What you've built:**
- Semantic search engines with real embeddings
- High-performance systems with HNSW tuning
- Hybrid search with dense + sparse vectors
- Memory-optimized deployments with quantization
- Multi-stage retrieval with Universal Query API

**Where to go from here:**
1. **Build a portfolio project** showcasing these skills
2. **Contribute to open source** - Qdrant, FastEmbed, or ecosystem tools
3. **Share your learnings** - Blog posts, tutorials, talks
4. **Join the community** - Qdrant Discord, forums, meetups
5. **Keep experimenting** - Try new models, datasets, and patterns

The future of search is semantic, hybrid, and multi-stage. You're now equipped to build it! 🚀

**Happy building, and welcome to the world of modern vector search!** 🎊
