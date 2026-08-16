# Day 1: Building a Semantic Search Engine

## 🎯 Agenda

Level up from toy examples to real-world semantic search! You'll build a movie recommendation system that understands meaning, not just keywords.

**Learning objectives:**
- Generate real text embeddings using sentence transformers
- Handle documents longer than model token limits
- Compare different text chunking strategies
- Work with named vectors (multiple vectors per point)
- Implement filtering and result grouping

## 📋 Requirements

### Dependencies
- `qdrant-client` v1.13.4+
- `sentence-transformers` v5.7.0+
- `llama-index-core` v0.14.23+ (for chunking utilities)
- `llama-index-embeddings-huggingface`
- `transformers` (tokenizer tools)
- `python-dotenv`

### Hardware
- **Local**: Any modern CPU (embeddings run on CPU by default)
- **RAM**: 4GB+ recommended for sentence-transformers models

### Knowledge Prerequisites
- Completed Day 0 (basic vector operations)
- Understanding of text tokenization (helpful but not required)

## 📓 Notebooks in This Folder

### 1. `day1_vector_search_dunds.ipynb` (Main Tutorial)
**The big project:** Build a semantic movie recommendation system from scratch.

**What it covers:**
- Setting up the `all-MiniLM-L6-v2` embedding model (384 dimensions)
- Processing 15 science fiction movie descriptions
- Implementing three chunking strategies:
  - **Fixed-size chunks**: Raw token-based splitting
  - **Sentence chunks**: Respecting sentence boundaries with overlap
  - **Semantic chunks**: AI-powered meaning-aware splitting
- Creating a collection with named vectors (one vector field per strategy)
- Uploading 357 vectors across all movies and strategies
- Running semantic queries like "alien invasion" or "time travel"
- Comparing chunk strategies for retrieval quality
- Applying payload filters (e.g., year >= 2000)
- Grouping results by movie title to avoid duplicate chunks

**Key concepts introduced:**
- **Embeddings**: Dense vector representations of text capturing semantic meaning
- **Token limits**: Why models have input size constraints (256 tokens for MiniLM)
- **Chunking**: Breaking long text into smaller pieces
- **Chunk overlap**: Preserving context between chunks
- **Named vectors**: Multiple vector representations per point
- **Semantic chunking**: Using embeddings to find natural breakpoints
- **Result grouping**: De-duplicating search results by a field

### 2. `day1_project.ipynb`
**Mini-project:** Apply Day 1 concepts to build your own semantic search system.

## ✅ What You've Accomplished

By completing Day 1, you've:

1. **Generated real embeddings** - Used sentence transformers to convert text to 384D vectors
2. **Solved the token limit problem** - Learned three strategies to chunk long documents
3. **Built a production-quality search engine** - 15 movies, 357 vectors, multiple search strategies
4. **Compared approaches empirically** - Saw how chunking affects search quality
5. **Implemented advanced queries** - Filters, grouping, and multi-field payloads
6. **Created a named vector collection** - Stored multiple representations per document

## 💡 Key Takeaways & Benefits

### What You Learned

**Embeddings are the bridge from text to vectors:**
- Sentence transformers convert natural language into numerical representations
- Similar meanings produce similar vectors (high cosine similarity)
- The `all-MiniLM-L6-v2` model is lightweight (80MB) and fast

**Chunking strategies matter:**
- **Fixed-size**: Fast and predictable, but may split ideas awkwardly
- **Sentence-based**: Preserves grammar, good middle ground
- **Semantic**: Most coherent chunks, but slowest to generate

**Named vectors enable experimentation:**
- Store multiple embeddings for the same document
- Test different strategies without rebuilding collections
- Query specific vector fields with `using="fixed"`

**Grouping prevents duplicate results:**
- Chunks from the same movie can dominate results
- `query_points_groups()` returns one best chunk per movie
- Essential for user-facing search applications

### Why This Matters

You've built the core architecture behind:
- **Documentation search**: Find relevant docs by asking questions in natural language
- **Product recommendations**: "Find products similar to this description"
- **Content discovery**: "More movies like this" on streaming platforms
- **Knowledge bases**: Semantic search over company wikis or help centers

### Real-World Insights

**Chunking trade-offs:**
- Smaller chunks = more precise but less context
- Larger chunks = more context but may be less focused
- Overlap helps preserve continuity across chunk boundaries

**When to use which strategy:**
- **Fixed**: When speed matters and text is uniform
- **Sentence**: Best general-purpose choice for natural language
- **Semantic**: When chunk coherence is critical (legal docs, technical writing)

## 🚀 What's Next

Day 1 used in-memory Qdrant (`:memory:`), which means:
- ✅ Fast and simple for learning
- ❌ No persistence (data lost when session ends)
- ❌ No HNSW indexing (exact search only)
- ❌ Doesn't scale beyond ~10K vectors

**Continue to Day 2** to learn:
- How the HNSW algorithm powers fast approximate search
- Performance tuning with `m` and `ef_construct` parameters
- When and how to build payload indexes
- Measuring filtering overhead
- Scaling to 100K+ vectors on Qdrant Cloud

## 🔍 Deep Dive: How Semantic Search Works

When you search for "alien invasion", here's what happens:

1. **Query embedding**: Your search text is converted to a 384D vector
2. **Vector comparison**: Qdrant computes cosine similarity against all stored vectors
3. **Ranking**: Results are sorted by similarity score (0 to 1)
4. **Filtering** (if specified): Only vectors matching payload conditions are considered
5. **Grouping** (if specified): Best match per group is returned

**Example from the notebook:**
```
Query: "alien invasion"
Top result: "E.T. the Extra-Terrestrial" (score: 0.5720)
Matching chunk: "The film opens with a group of botanist aliens 
visiting Earth, only to be interrupted by government agents..."
```

The model didn't match the exact words "alien invasion" - it understood the *semantic meaning* and found relevant content about aliens arriving on Earth.

## 🎓 Pro Tips from Day 1

1. **Tokenizer matters**: Always use the same tokenizer as your embedding model
2. **Warm-up queries**: First query may be slow while model loads
3. **Chunk size sweet spot**: 50-200 tokens works well for most use cases
4. **Test chunking strategies**: Your data may perform better with one approach
5. **Payload fields are cheap**: Add metadata liberally (year, author, category)

You've graduated from basics to building real semantic applications. The concepts you've learned are used in production systems at companies worldwide!
