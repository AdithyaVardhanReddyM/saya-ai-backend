# Saya AI Backend

A FastAPI backend service with CrewAI agents and RAG (Retrieval-Augmented Generation) file processing capabilities using TiDB for vector storage.

## Features

- **CrewAI Integration**: Customer support agent powered by Gemini
- **RAG File Processing**: Process PDF and TXT files for knowledge base
- **Vector Storage**: Store embeddings in TiDB with Cohere embeddings
- **RESTful API**: Clean API endpoints for chat and file processing

## Setup

### 1. Install Dependencies

```bash
# Install uv if you haven't already
pip install uv

# Install project dependencies
uv sync
```

### 2. Environment Configuration

Make sure your `.env` file contains all required variables:

```env
# AI Models
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key

# TiDB Configuration
TIDB_HOST=your_tidb_host
TIDB_PORT=4000
TIDB_USER=your_tidb_user
TIDB_PASSWORD=your_tidb_password
TIDB_DB_NAME=your_database_name
CA_PATH=/etc/ssl/cert.pem
```

### 3. Initialize Database

```bash
# Initialize database and create tables
python init_db.py
```

### 4. Test File Processing (Optional)

```bash
# Test the file processing functionality
python test_file_processing.py
```

### 5. Run the Server

```bash
# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Chat Endpoint

```http
POST /chat
Content-Type: application/json

{
    "message": "Hello, I need help with my order"
}
```

### File Processing Endpoint

```http
POST /process-file
Content-Type: application/json

{
    "url": "https://example.com/document.pdf",
    "filename": "document.pdf",
    "agentId": "agent-123"
}
```

### Get Embeddings

```http
GET /embeddings/{agent_id}
```

### Health Check

```http
GET /health
```

## File Processing Flow

1. **Download**: File is downloaded from the provided URL
2. **Extract**: Text is extracted (PDF or TXT files supported)
3. **Chunk**: Text is split into manageable chunks (1000 chars with 200 overlap)
4. **Embed**: Each chunk is embedded using Cohere's embed-english-v3.0 model
5. **Store**: Embeddings are stored in TiDB with metadata

## Database Schema

The `embeddings` table structure:

- `id`: UUID primary key
- `agentId`: String identifier for the agent
- `text`: The text chunk (LONGTEXT)
- `metadata`: JSON metadata about the chunk
- `vector`: JSON string of the embedding vector (1024 dimensions)
- `createdAt`: Timestamp

## Supported File Types

- **PDF**: Text extraction using PyPDF2
- **TXT**: Plain text files with encoding detection

## Error Handling

The API includes comprehensive error handling for:

- Invalid file types
- Download failures
- Text extraction errors
- Embedding generation failures
- Database connection issues
