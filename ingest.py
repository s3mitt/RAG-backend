import os
from typing import List, Dict, Any
from pypdf import PdfReader
from pinecone_upload import PineconeManager

class DocumentIngestor:
    def __init__(self):
        self.pinecone_manager = PineconeManager()
    
    def read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def read_text_file(self, file_path: str) -> str:
        """Read text from plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def ingest_directory(self, directory: str, file_extensions: List[str] = ['.pdf', '.txt']):
        """
        Ingest all documents from a directory
        """
        documents = []
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                
                if ext in file_extensions:
                    try:
                        if ext == '.pdf':
                            text = self.read_pdf(file_path)
                        else:
                            text = self.read_text_file(file_path)
                        
                        doc = {
                            "id": filename.replace(ext, ''),
                            "text": text,
                            "metadata": {
                                "source": filename,
                                "file_type": ext[1:],
                                "file_path": file_path
                            }
                        }
                        documents.append(doc)
                        print(f"Processed: {filename}")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
        
        return documents
    
    def ingest_single_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single file"""
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.pdf':
            text = self.read_pdf(file_path)
        else:
            text = self.read_text_file(file_path)
        
        doc = {
            "id": filename.replace(ext, ''),
            "text": text,
            "metadata": {
                "source": filename,
                "file_type": ext[1:],
                "file_path": file_path
            }
        }
        
        return doc
    
    def process_and_upload(self, documents: List[Dict[str, Any]]):
        """Process documents and upload to Pinecone"""
        self.pinecone_manager.create_index()
        self.pinecone_manager.upload_documents(documents)

if __name__ == "__main__":
    ingestor = DocumentIngestor()
    
    # Example: Ingest from a directory
    # documents = ingestor.ingest_directory("./documents")
    # ingestor.process_and_upload(documents)
    
    # Example: Ingest single file
    # doc = ingestor.ingest_single_file("./sample.pdf")
    # ingestor.process_and_upload([doc])
    
    print("Document ingestor ready. Use the methods above to ingest your documents.")
