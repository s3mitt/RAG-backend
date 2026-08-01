import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import google.genai as genai
import tiktoken

load_dotenv()

class PineconeManager:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "rag-knowledge-base")
        self.dimension = 768  # Gemini embedding dimension
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes().indexes]
        
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Created index: {self.index_name}")
        else:
            print(f"Index {self.index_name} already exists")
    
    def get_index(self):
        """Get the Pinecone index"""
        return self.pc.Index(self.index_name)
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Google Gemini"""
        response = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
        )
        return response.embeddings[0].values
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks with overlap"""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap
        
        return chunks
    
    def upload_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upload documents to Pinecone
        documents: List of dicts with 'text', 'metadata', 'id' keys
        """
        index = self.get_index()
        
        vectors = []
        for doc in documents:
            chunks = self.chunk_text(doc['text'])
            for i, chunk in enumerate(chunks):
                embedding = self.get_embedding(chunk)
                vector_id = f"{doc['id']}_chunk_{i}"
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        **doc['metadata'],
                        "chunk_index": i,
                        "text": chunk
                    }
                })
                
                if len(vectors) >= batch_size:
                    index.upsert(vectors=vectors)
                    print(f"Uploaded batch of {len(vectors)} vectors")
                    vectors = []
        
        if vectors:
            index.upsert(vectors=vectors)
            print(f"Uploaded final batch of {len(vectors)} vectors")
        
        print(f"Total documents uploaded: {len(documents)}")

if __name__ == "__main__":
    # Example usage
    manager = PineconeManager()
    manager.create_index()
    
    # Sample documents
    sample_docs = [
        {
            "id": "doc1",
            "text": "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.",
            "metadata": {"source": "fastapi_docs", "category": "framework"}
        },
        {
            "id": "doc2", 
            "text": "Pinecone is a vector database optimized for machine learning applications. It allows you to store, search, and manage vector embeddings at scale.",
            "metadata": {"source": "pinecone_docs", "category": "database"}
        }
    ]
    
    manager.upload_documents(sample_docs)
