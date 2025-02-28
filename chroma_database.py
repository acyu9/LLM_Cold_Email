import pandas as pd
import uuid
import chromadb

def get_chromadb_collection(collection_name):
    """Initialize ChromaDB client and get the collection."""
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def load_data_to_chromadb(csv_file, collection_name):
    """Load data from CSV into ChromaDB collection."""
    df = pd.read_csv(csv_file)

    collection = get_chromadb_collection(collection_name)

    # If the collection is empty, add data
    if not collection.count():
        for _, row in df.iterrows():
            collection.add(
                documents=row['Techstack'],
                metadatas={'links': row['Links']},
                ids=[str(uuid.uuid4())]
            )

def query_chromadb(collection_name, query_texts, n_results=2):
    """Query the ChromaDB collection for relevant links."""
    collection = get_chromadb_collection(collection_name)

    links = collection.query(
        query_texts=query_texts,
        n_results=n_results
    ).get('metadatas', [])

    return links