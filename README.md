# Saya AI Backend

A comprehensive AI-powered customer support backend that leverages advanced agent orchestration and retrieval-augmented generation (RAG) to deliver intelligent, context-aware customer service. The system dynamically creates specialized AI agents using CrewAI that can understand customer queries, access relevant knowledge from processed documents, and execute actions across multiple integrated business platforms.

## Key Capabilities

- **Intelligent Agent Orchestration**: Uses CrewAI to create dynamic customer support agents that adapt their behavior based on the conversation context and available tools
- **Document Processing & Knowledge Base**: Processes PDF and text documents to create searchable knowledge bases using vector embeddings stored in TiDB
- **Multi-Platform Integration**: Seamlessly connects with payment processing systems, team communication platforms, and scheduling services to perform real-world actions
- **Contextual Responses**: Combines vector search results with conversational AI to provide accurate, helpful responses to customer inquiries
- **Actionable Support**: Agents can perform tasks like creating payment links, posting team updates, retrieving customer information, and facilitating meeting scheduling

The backend provides RESTful APIs for chat interactions, file processing, and system health monitoring, making it easy to integrate into existing customer support workflows.

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
