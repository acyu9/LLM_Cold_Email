import pandas as pd
import uuid
import chromadb


class Portfolio:
    def __init__(self, file_path='resource/my_portfolio.csv'):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(name='portfolio')
    
    def load_data_to_chromadb(self):
        """Load data from CSV into ChromaDB collection."""
        # If the collection is empty, add data
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=row['Techstack'],
                    metadatas={'links': row['Links']},
                    ids=[str(uuid.uuid4())]
                )
    
    def query_links(self, skills):
        """Query the ChromaDB collection for relevant links."""
        # Get links with query texts
        links = self.collection.query(
            query_texts=skills,
            n_results=2
        ).get('metadatas', [])

        return links