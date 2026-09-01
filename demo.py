"""
demo.py
=======
Turns real sentences into vectors and runs semantic search against
SimpleVectorDB (see vector_db.py).

We use TF-IDF (scikit-learn) instead of a neural embedding model here
on purpose: it needs no big model download, runs instantly, and still
demonstrates the exact same workflow you'd use with OpenAI/Cohere/
sentence-transformers embeddings later:

    text --> vector --> store --> search-by-similarity

Swap in real embeddings later by replacing `embed()` below.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from vector_db import SimpleVectorDB

# 1. Some sample "knowledge base" documents
documents = [
    ("doc1", "The cat sat quietly on the warm windowsill in the afternoon sun."),
    ("doc2", "Dogs are loyal companions and love to play fetch in the park."),
    ("doc3", "Python is a popular programming language for data science and AI."),
    ("doc4", "Machine learning models learn patterns from large datasets."),
    ("doc5", "The stock market rallied today after strong earnings reports."),
    ("doc6", "Investors are watching interest rates and inflation closely."),
    ("doc7", "Cats and dogs are the most common household pets worldwide."),
    ("doc8", "Neural networks are the foundation of modern deep learning."),
]

# 2. Fit a TF-IDF vectorizer on our documents (this is our "embedding model")
texts = [text for _, text in documents]
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(texts)


def embed(text: str):
    """Turn a string into a vector. Swap this out for a real embedding
    model (OpenAI, Cohere, sentence-transformers) when you're ready."""
    return vectorizer.transform([text]).toarray()[0]


# 3. Build the vector DB and insert every document
db = SimpleVectorDB()
for doc_id, text in documents:
    db.add(id=doc_id, vector=embed(text), text=text, metadata={"length": len(text)})

print(f"Inserted {len(db)} documents into the vector DB.\n")

# 4. Run some semantic search queries
queries = [
    "tell me about pets",
    "how does AI learn from data",
    "what's happening with the economy",
]

for query in queries:
    print(f"QUERY: {query!r}")
    results = db.search(embed(query), top_k=3)
    for r in results:
        print(f"   [{r['score']:.4f}] {r['id']}: {r['text']}")
    print()

# 5. Example of metadata filtering (like a WHERE clause)
print("QUERY with filter (only docs longer than 60 chars): 'pets'")
results = db.search(embed("pets"), top_k=3, filter_fn=lambda m: m["length"] > 60)
for r in results:
    print(f"   [{r['score']:.4f}] {r['id']}: {r['text']}")

# 6. Persistence demo
db.save("my_vector_db.pkl")
print("\nSaved DB to my_vector_db.pkl")

db2 = SimpleVectorDB()
db2.load("my_vector_db.pkl")
print(f"Reloaded DB has {len(db2)} records.")

print("""
NOTE: TF-IDF is *lexical* search - it only matches literal words. Try
changing the economy query to say "inflation" instead of "economy" and
watch the results improve dramatically. A real embedding model (e.g.
sentence-transformers, OpenAI embeddings) captures *meaning*, so
"economy" and "inflation" would score as related even without sharing
a single word. That's the whole value proposition of vector search
over old-school keyword search - see README.md for how to upgrade.
""")
