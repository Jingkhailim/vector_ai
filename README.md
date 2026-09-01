# Build-Your-Own Vector Database (Learning Project)

This is a hands-on way to understand what vector databases actually do,
by building a minimal one yourself instead of just reading about Pinecone
or Weaviate.

## Files

- **`vector_db.py`** — the database itself, built from scratch with numpy.
  Read this file top to bottom; it's fully commented.
- **`demo.py`** — inserts sample text into the DB and runs semantic search
  queries against it. Run this first: `python3 demo.py`

## The core idea

A vector database does three things:

1. **Embed**: turn content (text, images, audio) into a list of numbers
   (a "vector") that captures its meaning. Similar content → similar vectors.
2. **Store**: keep the vector alongside the original content and any metadata.
3. **Search**: given a query vector, find the stored vectors that are most
   *similar* to it — usually with **cosine similarity** (the angle between
   two vectors) rather than exact matching.

That's it. Everything else (indexing tricks, sharding, filtering, hybrid
search) is optimization on top of that core loop.

## Why not just use a normal database?

A normal database finds *exact* or *range* matches (`WHERE price < 100`).
A vector database finds *similar* things — "documents like this one,"
even if they don't share a single keyword. That's what makes it the
backbone of semantic search, recommendation systems, and RAG (Retrieval-
Augmented Generation) for LLMs.

## Try this yourself

1. Run `python3 demo.py` and read the output.
2. Notice the "economy" query returns nothing — because we used TF-IDF,
   which only matches literal words. Change the query to `"inflation"` or
   `"stock market"` and watch it work again.
3. Open `vector_db.py` and change `top_k` or add your own documents in
   `demo.py`.
4. Try adding a *bad* filter in `search(..., filter_fn=...)` and see how
   metadata filtering works.

## Leveling up: real embeddings

Swap the `embed()` function in `demo.py` for a real embedding model, and
you get *semantic* (meaning-based) search instead of *lexical* (keyword)
search:

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return model.encode(text)
```

With this swap, a query like "economy" would correctly match documents
about "inflation" and "stock market" — because the model understands
they're related concepts, not just different words.

## Leveling up: real vector databases

Once you understand the brute-force version here, the next step is an
**Approximate Nearest Neighbor (ANN)** index, which is how real vector
databases stay fast at millions of vectors instead of comparing against
every single one:

| Tool | Type | Good for |
|---|---|---|
| **FAISS** (Meta) | Library, in-process | Local prototyping, full control, no server |
| **Chroma** | Embedded / lightweight server | Quick RAG projects, works great with LangChain |
| **Qdrant** | Self-hosted or cloud | Production apps, filtering, good docs |
| **Weaviate** | Self-hosted or cloud | Hybrid search (keyword + vector), GraphQL API |
| **Pinecone** | Fully managed cloud | No infra to manage, scales automatically |

They all do the same three things as `vector_db.py` — embed, store,
search — just with a much faster index (commonly **HNSW**: Hierarchical
Navigable Small World graphs) so search stays fast even at huge scale.

## Suggested next project

Build a tiny RAG pipeline:
1. Chunk a PDF or set of notes into paragraphs.
2. Embed each chunk with `sentence-transformers`.
3. Store in Chroma or FAISS.
4. On a user question, embed the question, retrieve the top 3 chunks,
   and pass them to an LLM as context to answer with.

That's the pattern behind most "chat with your documents" tools.
