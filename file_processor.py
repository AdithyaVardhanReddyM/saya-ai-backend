import os
import json
import httpx
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere
from dotenv import load_dotenv

load_dotenv()

class FileProcessor:
    def __init__(self):
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    async def download_file(self, url: str) -> bytes:
        """Download file from URL"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            from io import BytesIO
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_txt(self, txt_content: bytes) -> str:
        """Extract text from TXT content"""
        try:
            return txt_content.decode('utf-8')
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return txt_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            raise Exception("Unable to decode text file with any common encoding")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere (1024 dimensions)"""
        try:
            response = self.cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    async def process_file(self, url: str, filename: str, agent_id: str) -> List[Dict[str, Any]]:
        """Process file and return chunks with embeddings"""
        # Download file
        file_content = await self.download_file(url)
        
        # Extract text based on file type
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            text = self.extract_text_from_pdf(file_content)
        elif file_extension in ['txt', 'text']:
            text = self.extract_text_from_txt(file_content)
        else:
            raise Exception(f"Unsupported file type: {file_extension}")
        
        if not text.strip():
            raise Exception("No text content found in file")
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        if not chunks:
            raise Exception("No chunks generated from text")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Prepare data for database
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            processed_chunks.append({
                "agentId": agent_id,
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": file_extension
                },
                "vector": embedding  # Store as list directly for TiDB vector column
            })
        
        return processed_chunks