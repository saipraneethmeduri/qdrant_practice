# Day 4: Large-Scale Ingestion & Quantization

## 🎯 Agenda

Scale from thousands to millions of vectors while maintaining performance. Learn enterprise-grade techniques for handling massive datasets and optimizing memory usage through quantization.

**Learning objectives:**
- Understand large-scale data ingestion strategies (400M vectors)
- Learn production-grade collection configurations
- Master quantization techniques (scalar, binary, product)
- Optimize the oversampling + rescoring pipeline
- Measure accuracy vs speed tradeoffs
- Deploy memory-optimized production systems

## 📋 Requirements

### Dependencies
- `qdrant-client` v1.13.4+
- `pandas`, `numpy`, `pyarrow`, `fastparquet` (for data processing)
- `tqdm` (progress tracking)
- `sentence-transformers` (for project notebook)
- `python-dotenv`

### Hardware Requirements

**For notebook 1 (LAION-400M - DON'T RUN LOCALLY):**
- ⚠️ **Minimal usable config**: 64GB RAM, 8 CPU cores, 1TB disk
- ⚠️ **Recommended**: Cloud VM or dedicated server
- ⚠️ **Storage**: 500GB+ for dataset download
- ⚠️ **Network**: Fast, stable connection for downloading 400M vectors

**For notebook 2 (Quantization project - CAN RUN LOCALLY):**
- ✅ **Local-friendly**: 8GB RAM, any modern CPU
- ✅ **Storage**: < 1GB for small dataset
- ✅ **Qdrant**: Cloud or local instance

### Knowledge Prerequisites
- Completed Day 0-3
- Understanding of HNSW and vector indexing
- Basic statistics (percentiles, means)
- Production deployment concepts (helpful but not required)

## 📓 Notebooks in This Folder

### 1. `day4_large_scale_ingestion(don't_run_locally).ipynb` ⚠️

**🚨 IMPORTANT: This notebook requires enterprise-grade resources. Do NOT run locally!**

**What it covers:**

This notebook demonstrates how to ingest and benchmark the LAION-400M dataset - a real-world, production-scale vector database challenge.

**Part 1: Dataset Overview**
- LAION-400M: 400 million image-text pairs
- 512-dimensional CLIP embeddings per image
- Split into 409 parts for manageable processing
- Use case: Large-scale image similarity search

**Part 2: Production Collection Configuration**
```python
vectors_config=models.VectorParams(
    size=512,
    distance=models.Distance.COSINE,
    datatype=models.Datatype.FLOAT16,  # Memory optimization
    on_disk=True  # Store vectors on disk
)
```

Key configuration choices explained:
- **Float16 datatype**: Reduces memory by 50% with minimal accuracy loss
- **On-disk storage**: Handles datasets larger than RAM
- **Binary quantization**: Further 32x compression for fast filtering
- **Large segments**: `max_segment_size=5_000_000` for faster search
- **Reduced HNSW m**: `m=6` to lower memory usage
- **Optimizers tuned**: Larger segments prioritize search speed over indexing

**Part 3: Bulk Upload Strategy**
- Download dataset parts incrementally (each ~1M vectors)
- Load embeddings as NumPy arrays
- Load metadata as Pandas DataFrames
- Upload to Qdrant with parallel workers
- Clear local files after upload to save disk space
- Track progress across 409 parts

**Part 4: Ground Truth Generation**
- Full scan implementation for exact search results
- Memory-efficient chunked processing
- Cosine similarity calculation across entire dataset
- Used for benchmarking HNSW accuracy

**Part 5: Evaluation & Benchmarking**
- Compare Qdrant search vs ground truth
- Measure precision@k metrics
- Test rescoring with different limits
- Evaluate quantization impact on accuracy

**Key concepts introduced:**
- **Float16 precision**: Trading precision for memory (2x savings)
- **On-disk vectors**: Storing beyond RAM capacity
- **Binary quantization**: Extreme compression (32x) for fast pre-filtering
- **Segment optimization**: Tuning for scale
- **Parallel upload**: Using multiple workers for bulk ingestion
- **Ground truth**: Generating reference results for accuracy measurement
- **Precision@k**: Measuring search quality

**Why you shouldn't run locally:**
- 400M vectors × 512 dimensions × 4 bytes = ~800GB uncompressed
- Even with optimizations, requires 50-100GB RAM minimum
- Download size: 300-500GB across 409 parts
- Processing time: Hours to days depending on hardware
- Risk of system crashes, out-of-memory errors

**How to learn from this notebook:**
1. **Read and understand** the code structure
2. **Study the configurations** for production use cases
3. **Note the upload strategy** for batch processing
4. **Understand the tradeoffs** (memory vs accuracy vs speed)
5. **Apply patterns** to your smaller-scale projects
6. **Use as reference** when building production systems

**Real-world applications:**
- Image search engines (Pinterest, Google Images)
- Video recommendation systems (YouTube, TikTok)
- Content moderation at scale
- Reverse image search
- Visual product discovery (e-commerce)

### 2. `day4_project.ipynb` ✅

**👍 This notebook is local-friendly - run and experiment!**

**What it covers:**

Build a quantization-optimized search engine for a recipe collection and measure real-world performance impacts.

**Part 1: Baseline Measurement**
- Create unquantized collection with recipe data (12 recipes)
- Generate 384D embeddings with `all-MiniLM-L6-v2`
- Measure baseline latency across diverse test queries
- Establish performance benchmark

**Part 2: Quantization Methods**

Test three quantization approaches:

**Scalar Quantization (INT8):**
- Converts float32 → int8 (4x compression)
- Learns min/max per dimension
- 99th percentile clipping to handle outliers
- Best balance of speed and accuracy

**Binary Quantization:**
- Converts vectors to binary (1 bit per dimension)
- Extreme compression (32x)
- Fastest for pre-filtering
- Requires rescoring for accuracy

**Product Quantization (PQ):**
- Splits vectors into subspaces
- Learns codebook per subspace
- Configurable compression ratio (x4, x8, x16, x32)
- Good memory/accuracy tradeoff

**Part 3: Performance Benchmarking**
- Measure latency for each quantization method
- Test without rescoring (fast but less accurate)
- Test with rescoring (slower but accurate)
- Compare speedup vs baseline
- Calculate P95 latencies for reliability

**Part 4: Accuracy vs Speed Tradeoff**
- Test different oversampling factors (2x, 3x, 5x, 8x, 10x)
- Measure accuracy retention vs baseline
- Find optimal oversampling for your use case
- Balance between speed and precision

**Part 5: Analysis & Recommendations**
- Summarize findings for each method
- Identify best configuration for specific scenarios
- Document tradeoffs and decision criteria

**Key concepts introduced:**
- **Scalar quantization**: Discretizing continuous values
- **Binary quantization**: Extreme compression for filtering
- **Product quantization**: Subspace decomposition
- **Oversampling**: Fetching more candidates before rescoring
- **Rescoring**: Re-ranking with full-precision vectors
- **Always_ram**: Keeping quantized vectors in memory
- **Accuracy retention**: Measuring quality degradation
- **P95 latency**: Reliability metric (95% of queries faster than this)

**Example results from the notebook:**
```
Baseline: 929ms average latency

Scalar quantization:   189ms (4.9x speedup)
Binary quantization:   190ms (4.9x speedup)
Product quantization:  190ms (4.9x speedup)

All methods: 100% accuracy retention with rescoring
```

**Test queries used:**
- Semantic: "quick weeknight dinner", "comforting winter meal"
- Keyword: "French beef wine sauce", "miso glazed salmon"
- Hybrid: "easy Asian recipe under 30 minutes"
- Constraints: "quick recipe less than 30 minutes"

## ✅ What You've Accomplished

By completing Day 4, you've:

1. **Learned enterprise-scale patterns** - Configurations for 400M+ vector datasets
2. **Mastered quantization** - Three compression methods with different tradeoffs
3. **Optimized memory usage** - Reduced storage by 4-32x
4. **Tuned accuracy pipelines** - Oversampling and rescoring strategies
5. **Measured real-world impact** - Latency, memory, and accuracy metrics
6. **Production-ready knowledge** - Deploying scalable vector search systems

## 💡 Key Takeaways & Benefits

### What You Learned

**Large-scale ingestion strategies:**
- **Incremental processing**: Download and upload in chunks to manage resources
- **Parallel workers**: Use multiple threads for faster uploads
- **Disk-based storage**: Handle datasets larger than RAM
- **Segment optimization**: Tune for your read/write ratio
- **Memory efficiency**: Float16, quantization, on-disk vectors

**Quantization fundamentals:**

| Method | Compression | Speed | Accuracy | Use Case |
|--------|------------|-------|----------|----------|
| **Scalar** | 4x | Fast | High | General purpose |
| **Binary** | 32x | Fastest | Medium* | Pre-filtering |
| **Product** | 4-32x | Fast | Medium-High | Configurable balance |

*With proper rescoring, accuracy approaches full-precision

**The rescoring pipeline:**
1. **Pre-filter** with quantized vectors (fast, approximate)
2. **Oversample** candidates (e.g., fetch 3x more than needed)
3. **Rescore** top candidates with full-precision vectors
4. **Return** final top-k results

**Why rescoring works:**
- Quantization is fast but approximate
- Most candidates are clearly not relevant (quantization is good enough)
- Only the top candidates need precise scoring
- Oversampling ensures true positives aren't missed

**Configuration patterns for scale:**

**Memory-constrained:**
```python
- datatype=FLOAT16
- on_disk=True
- quantization=BinaryQuantization
- always_ram=True  # quantized only
```

**Speed-optimized:**
```python
- on_disk=False
- quantization=ScalarQuantization
- hnsw.m=16 or higher
- max_segment_size=large
```

**Balanced production:**
```python
- datatype=FLOAT16
- quantization=ScalarQuantization
- hnsw.on_disk=False
- vectors.on_disk=True
```

### Why This Matters

**Production systems need quantization:**
- **Cost reduction**: 4-32x less RAM/storage = 4-32x lower cloud bills
- **Faster search**: Comparing smaller vectors is faster
- **Better scaling**: Fit more vectors per node
- **Maintained accuracy**: With rescoring, quality is preserved

**Real-world numbers:**
- 1M vectors × 384 dims × 4 bytes = 1.5GB uncompressed
- With scalar quantization: 384MB (4x reduction)
- With binary quantization: 48MB (32x reduction)
- At 100M scale, this is 150GB → 12GB → 4.6GB

**Industry adoption:**
- Every major vector database uses quantization
- Critical for mobile/edge deployments
- Standard in production RAG systems
- Essential for cost-effective scaling

### Real-World Applications

**Day 4 techniques power:**
- **Reverse image search** at Google/Pinterest scale (billions of images)
- **E-commerce search** with massive product catalogs
- **Video recommendation** across millions of videos
- **Mobile apps** with on-device vector search (extreme compression needed)
- **Real-time personalization** (latency-critical quantized search)
- **Archive search** (historical data on cheap disk storage)

## 🚀 What's Next

You've completed the core Qdrant curriculum! Here's how to continue:

**Advanced Topics:**
- **Multi-tenancy**: Partitioning collections by user/tenant
- **Distributed Qdrant**: Running clusters for high availability
- **Custom sharding**: Geo-distributed data placement
- **Streaming updates**: Real-time index updates
- **Cross-encoder reranking**: Adding a second-stage ranking model
- **Multi-modal search**: Combining text, image, audio vectors
- **Monitoring & observability**: Metrics, logging, alerting

**Build Production Projects:**
- Full-stack RAG application with quantized search
- Multi-tenant SaaS with vector search
- Real-time recommendation engine
- Visual search for e-commerce
- Semantic documentation portal
- Content moderation pipeline

**Performance Optimization:**
- A/B test quantization methods on your data
- Profile memory usage patterns
- Optimize for your query distribution
- Implement caching strategies
- Load test at scale

## 🔍 Deep Dive: Quantization Mathematics

### Scalar Quantization

**Concept:** Map continuous float range to discrete integers

```python
# Per-dimension quantization
min_val, max_val = vector.min(), vector.max()
scale = (max_val - min_val) / 255  # INT8 range

quantized = ((vector - min_val) / scale).astype(int8)

# Dequantization for rescoring
dequantized = quantized * scale + min_val
```

**Memory:** float32 (4 bytes) → int8 (1 byte) = **4x compression**

### Binary Quantization

**Concept:** Reduce each dimension to 1 bit (positive or negative)

```python
# Simplified version
binary = (vector > 0).astype(uint8)  # 1 bit per dimension
# In practice, uses median or learned threshold

# Distance approximation
hamming_distance = np.sum(binary_a != binary_b)
# Correlates with cosine distance
```

**Memory:** float32 (4 bytes) → 1 bit = **32x compression**

**Why it works:** High-dimensional vectors have statistical properties that preserve relative distances even with 1-bit quantization

### Product Quantization

**Concept:** Split vector into subspaces, quantize each separately

```python
# Split 384D vector into 8 subspaces of 48D each
subspaces = vector.reshape(8, 48)

# Learn codebook of 256 centroids per subspace (during training)
codebooks = [kmeans(subspace_i) for subspace_i in training_data]

# Quantize: find nearest centroid in each subspace
pq_codes = [find_nearest_centroid(sub, codebook[i]) 
            for i, sub in enumerate(subspaces)]
# Store 8 bytes (one index per subspace) instead of 384 floats
```

**Memory:** 384 dims × 4 bytes → 8 indices × 1 byte = **192x → 8 bytes (48x compression)**

## 🎓 Pro Tips from Day 4

1. **Always benchmark on your data** - Quantization impact varies by domain
2. **Start with scalar quantization** - Best default choice
3. **Binary needs rescoring** - Never use binary results directly
4. **Oversampling is cheap** - 3-5x is usually sufficient
5. **Monitor accuracy** - Track precision@k in production
6. **Float16 is low-hanging fruit** - 2x compression with negligible accuracy loss
7. **Tune segment sizes** - Larger segments = faster search for large datasets
8. **Use `always_ram=True` for quantized data** - Keep fast pre-filter in memory
9. **On-disk is for vectors** - Keep HNSW index in RAM if possible
10. **Test before scaling** - Validate configuration on subset before full upload

## 🧪 Experiment Ideas

1. **Compare your domain**: Test quantization on your specific dataset
2. **Vary oversampling**: Find optimal factor for your accuracy requirements
3. **Mix methods**: Try scalar for some collections, binary for others
4. **Benchmark reranking**: Add cross-encoder second-stage ranking
5. **Measure memory**: Track actual RAM usage with different configs
6. **Stress test**: How many queries/second can quantized search handle?
7. **Accuracy curves**: Plot accuracy vs oversampling factor
8. **Cost analysis**: Calculate cloud cost savings from quantization

## 🏆 Mastery Achieved

You now understand:
- **Enterprise-scale ingestion**: Techniques for millions to billions of vectors
- **Memory optimization**: Reducing footprint by 4-32x
- **Accuracy preservation**: Maintaining quality with compression
- **Production configuration**: Deploying efficient, scalable systems
- **Performance tuning**: Balancing speed, memory, and accuracy

**The complete production stack:**
```
┌─────────────────────────────────────────┐
│  Application Layer (RAG, Search, etc)   │
├─────────────────────────────────────────┤
│  Quantization + Rescoring Pipeline      │
│  (Oversampling → Full-precision)        │
├─────────────────────────────────────────┤
│  Hybrid Search (Dense + Sparse)         │
│  (Reciprocal Rank Fusion)               │
├──────────────────┬──────────────────────┤
│  Dense HNSW      │  Sparse Inverted     │
│  (quantized)     │  (full-precision)    │
├──────────────────┴──────────────────────┤
│  Payload Filters & Indexes              │
├─────────────────────────────────────────┤
│  Storage (RAM + Disk, optimized)        │
└─────────────────────────────────────────┘
```

## 📚 Further Learning

**Research Papers:**
- "Product Quantization for Nearest Neighbor Search" (Jégou et al.)
- "Billion-scale similarity search with GPUs" (Johnson et al.)
- "Learning to Quantize Deep Networks" (various)

**Qdrant Resources:**
- [Quantization Documentation](https://qdrant.tech/documentation/guides/quantization/)
- [Binary Quantization Blog](https://qdrant.tech/articles/binary-quantization/)
- [Benchmarks](https://qdrant.tech/benchmarks/)

**Production Guides:**
- Qdrant Cloud deployment best practices
- Monitoring and alerting setup
- Disaster recovery strategies
- Scaling patterns and anti-patterns

## 🎉 Congratulations!

You've completed the comprehensive Qdrant practice curriculum, progressing from basic vector operations to production-grade systems handling hundreds of millions of vectors with optimized memory and blazing-fast search.

**Your journey:**
- **Day 0**: First vectors and similarity search
- **Day 1**: Semantic search with real embeddings
- **Day 2**: HNSW performance tuning at scale
- **Day 3**: Sparse vectors and hybrid search
- **Day 4**: Enterprise ingestion and quantization

You're now equipped to build world-class vector search applications that power modern AI systems. These skills are directly applicable to production systems at companies building the future of search, recommendations, and AI-powered experiences!

**Next steps:** Build something amazing and share it with the community! 🚀
