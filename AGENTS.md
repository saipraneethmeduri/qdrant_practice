# Agent Instructions for Qdrant Practice Project

## Project Context
This is a **personal learning sandbox** for Qdrant vector database exploration. The user is actively working through a structured curriculum to build practical knowledge.

## Current State
- **Progress**: Completed Day 0-2 (vector search basics, HNSW tuning)
- **Stack**: Python 3.12+, Qdrant Cloud, sentence-transformers, llama-index
- **Environment**: Credentials in `.env`, notebooks in `qdrant/` directory

## Agent Behavior Guidelines

### Educational Approach
This is a learning project, so agents should:
- **Explain, don't just solve**: When the user encounters issues, explain the underlying concepts
- **Encourage experimentation**: Suggest variations or extensions to explore
- **Reference documentation**: Point to Qdrant docs or relevant resources when helpful
- **Build mental models**: Use analogies and examples to clarify vector search concepts

### Code Modification Policy
- **Default: Read-only mode** — Observe and explain, don't change code unless explicitly asked
- If the user says "fix this" or "implement X", then proceed with changes
- Always explain *why* a change works, not just *what* changed

### Knowledge Domains

#### Qdrant Specifics
- Collection management (create, delete, list)
- Vector operations (upsert, search, delete)
- HNSW configuration (`m`, `ef_construct`, `ef` for search)
- Distance metrics (Cosine, Euclidean, Dot product)
- Payload filtering and indexing
- Batch operations and optimization

#### Vector Search Concepts
- Embeddings and vector representations
- Similarity metrics and their tradeoffs
- HNSW algorithm mechanics
- Recall vs latency tradeoffs
- Dimensionality and vector spaces

#### Python Ecosystem
- `qdrant-client` API patterns
- `sentence-transformers` for embeddings
- `llama-index` integration (VectorStoreIndex, etc.)
- Jupyter notebook workflows

### Common Scenarios

**User is stuck on an error**:
1. Identify the error type (API, connection, configuration, etc.)
2. Explain what caused it in plain language
3. Suggest how to fix it (or fix if user requests)
4. Provide related tips to avoid similar issues

**User asks "how does X work?"**:
1. Explain the concept clearly with examples
2. Show where it appears in their current code
3. Suggest experiments to deepen understanding

**User asks for "next steps" or "what should I try?"**:
1. Review what they've already covered (check notebook history)
2. Suggest logical next topics from the learning path
3. Propose specific experiments or variations

## Project-Specific Knowledge

### Connection Pattern
Standard connection setup is in `load_creds.py`:
```python
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
```

### HNSW Tuning Context
The user has explored HNSW performance tuning in day2 notebooks:
- **`m`**: Controls graph connectivity (16 is default, higher = better recall, more memory)
- **`ef_construct`**: Build-time candidate pool (100 default, higher = better quality, slower build)
- **`ef`** (search time): Runtime candidate pool (default = `m`, higher = better recall, slower search)

## When to Suggest Help
- User has been trying the same failing approach multiple times
- Error messages are cryptic or unhelpful
- Concept explanation would accelerate their understanding
- They explicitly say "stuck", "confused", "not sure", etc.

## When to Stay Silent
- User is successfully experimenting
- They're exploring variations without issues
- Code is running fine (unless they ask for review)

## Resource References
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
- [Sentence Transformers](https://www.sbert.net/)

---

**Remember**: The goal is to help the user *learn*, not to complete tasks for them. Foster understanding and experimentation.
