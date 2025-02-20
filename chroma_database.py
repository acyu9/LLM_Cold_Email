import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection")

collection.upsert(
    documents=[
        "This is a document about New York",
        "This is a document about Delhi"
    ],
    ids=["id1", "id2"],
    metadatas=[
        {'url': 'https://en.wikipedia.org/wiki/New_York_City'},
        {'url': 'https://en.wikipedia.org/wiki/Delhi'}
    ]
)

all_docs = collection.get()
# print(all_docs)

results = collection.query(
    query_texts=['Query is about India'],
    n_results=2
)

# Semantic search - Delhi has smaller Euclidean distance
print(results)