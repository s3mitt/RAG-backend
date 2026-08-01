from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from pinecone_upload import PineconeManager
from ingest import DocumentIngestor
import google.genai as genai

load_dotenv()

app = FastAPI(title="RAG Knowledge Assistant API")

# CORS configuration
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
pinecone_manager = PineconeManager()
ingestor = DocumentIngestor()

# Configure Gemini
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class DocumentUploadResponse(BaseModel):
    message: str
    document_count: int

@app.get("/")
async def root():
    return {"message": "RAG Knowledge Assistant API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "pinecone_index": pinecone_manager.index_name}

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using RAG
    """
    try:
        # Generate embedding for the query
        query_embedding = pinecone_manager.get_embedding(request.query)
        
        # Search Pinecone for similar documents
        index = pinecone_manager.get_index()
        results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True
        )
        
        # Extract relevant context
        contexts = []
        sources = []
        for match in results['matches']:
            if match['score'] > 0.7:  # Relevance threshold
                contexts.append(match['metadata']['text'])
                sources.append({
                    "source": match['metadata'].get('source', 'Unknown'),
                    "score": match['score'],
                    "chunk_index": match['metadata'].get('chunk_index', 0)
                })
        
        if not contexts:
            return QueryResponse(
                answer="I couldn't find relevant information in the knowledge base to answer your question.",
                sources=[]
            )
        
        # Generate answer using Gemini
        context_text = "\n\n".join(contexts)
        prompt = f"""
        Based on the following context, answer the user's question. 
        If the answer is not in the context, say so.
        
        Context:
        {context_text}
        
        Question: {request.query}
        
        Provide a helpful, accurate answer based on the context above.
        """
        
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        answer = response.text
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF or TXT)
    """
    try:
        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Ingest the document
        doc = ingestor.ingest_single_file(file_path)
        ingestor.process_and_upload([doc])
        
        # Clean up
        os.remove(file_path)
        
        return DocumentUploadResponse(
            message=f"Successfully uploaded and indexed {file.filename}",
            document_count=1
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-directory")
async def upload_directory(directory_path: str):
    """
    Upload all documents from a directory
    """
    try:
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=400, detail="Directory does not exist")
        
        documents = ingestor.ingest_directory(directory_path)
        ingestor.process_and_upload(documents)
        
        return DocumentUploadResponse(
            message=f"Successfully uploaded and indexed {len(documents)} documents from {directory_path}",
            document_count=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-index")
async def clear_index():
    """
    Clear all documents from the Pinecone index
    """
    try:
        index = pinecone_manager.get_index()
        index.delete(delete_all=True)
        
        return {"message": "Index cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000))
    )
