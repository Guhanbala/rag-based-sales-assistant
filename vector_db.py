import chromadb
from chromadb.utils import embedding_functions

class VectorDB:
    def __init__(self, collection_name="sales_knowledge"):
        # Persistent storage so data is saved even after you close the program
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Using the embedding model we downloaded
        self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url="http://localhost:11434/api/embeddings"         # ← Correct URL
        )
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents: list, ids: list, metadatas: list = None):
        """Add documents + embeddings into the vector database"""
        if metadatas:
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
        else:
            self.collection.add(
                documents=documents,
                ids=ids
            )
        print(f"✅ Successfully added {len(documents)} documents to vector database.")

    def search(self, query: str, n_results: int = 5):
        """Search for similar past records"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results["documents"][0] if results["documents"] else []