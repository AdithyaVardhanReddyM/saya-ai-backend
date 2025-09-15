import os
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from typing import List

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

from database import get_db, Embeddings
from file_processor import FileProcessor
from tools.vector_search_tool import vector_search
from tools.stripe_mcp_tool import stripe_mcp
from tools.slack_tools import (
    slack_list_channels,
    slack_post_message,
    slack_reply_to_thread,
    slack_add_reaction,
    slack_get_channel_history,
    slack_get_thread_replies,
    slack_get_users,
    slack_get_user_profile
)

# Load .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CAL_EVENT_URL = os.getenv("CAL_EVENT_URL")

# Configure Gemini LLM via LiteLLM wrapper
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",   # LiteLLM expects provider/model format
    api_key=GEMINI_API_KEY,
)

# FastAPI app
app = FastAPI()

# Initialize file processor
file_processor = FileProcessor()

class Message(BaseModel):
    message: str
    agentId: str
    CalEnabled: bool | None = None
    StripeEnabled: bool | None = None
    SlackEnabled: bool | None = None
    CalUrl: str | None = None
    STRIPE_API_KEY: str | None = None
    SLACK_BOT_TOKEN: str | None = None
    SLACK_TEAM_ID: str | None = None
    SLACK_CHANNEL_IDS: str | None = None

class ProcessFileRequest(BaseModel):
    url: str
    filename: str
    agentId: str

class ProcessFileResponse(BaseModel):
    success: bool
    message: str
    chunks_processed: int

# Agent will be created per-request in /chat based on enabled tools and provided credentials


@app.post("/chat")
async def chat(msg: Message):
    # Determine enabled capabilities strictly from request (no env fallbacks)
    stripe_enabled = bool(msg.StripeEnabled) and bool(msg.STRIPE_API_KEY)
    slack_enabled = bool(msg.SlackEnabled) and bool(msg.SLACK_BOT_TOKEN)
    cal_enabled = bool(msg.CalEnabled) and bool(msg.CalUrl)

    # Build tool list
    tools_to_use = [vector_search]
    if stripe_enabled:
        tools_to_use.append(stripe_mcp)
    if slack_enabled:
        tools_to_use.extend([
            slack_list_channels,
            slack_post_message,
            slack_reply_to_thread,
            slack_add_reaction,
            slack_get_channel_history,
            slack_get_thread_replies,
            slack_get_users,
            slack_get_user_profile
        ])

    # Dynamic agent profile
    capabilities = ["RAG (vector search)"]
    if stripe_enabled:
        capabilities.append("Stripe MCP")
    if slack_enabled:
        capabilities.append("Slack")
    role = "Customer Support Agent"
    goal = f"Assist customers using {', '.join(capabilities)}."
    backstory_parts = [
        "You are a skilled support agent who can answer policy questions using the knowledge base (RAG)."
    ]
    if stripe_enabled:
        backstory_parts.append("You can take actions in Stripe via the Stripe MCP tool.")
    if slack_enabled:
        backstory_parts.append("You can communicate with the team and perform operations in Slack.")
    backstory = " ".join(backstory_parts)

    # Create an agent per request with the exact tools enabled
    support_agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools_to_use,
        verbose=True,
        memory=True,
        llm=gemini_llm
    )

    # Tool configuration that MUST be respected by the agent when calling tools
    tool_config_lines = []
    if stripe_enabled:
        tool_config_lines.append(f"- StripeMCPTool: include api_key='{msg.STRIPE_API_KEY}' in every call.")
    if slack_enabled:
        tool_config_lines.extend([
            f"- SlackListChannelsTool: include token='{msg.SLACK_BOT_TOKEN}', and either team_id='{msg.SLACK_TEAM_ID}' or channel_ids='{msg.SLACK_CHANNEL_IDS}'.",
            f"- SlackPostMessageTool: include token='{msg.SLACK_BOT_TOKEN}'.",
            f"- SlackReplyToThreadTool: include token='{msg.SLACK_BOT_TOKEN}'.",
            f"- SlackAddReactionTool: include token='{msg.SLACK_BOT_TOKEN}'.",
            f"- SlackGetChannelHistoryTool: include token='{msg.SLACK_BOT_TOKEN}'.",
            f"- SlackGetThreadRepliesTool: include token='{msg.SLACK_BOT_TOKEN}'.",
            f"- SlackGetUsersTool: include token='{msg.SLACK_BOT_TOKEN}', team_id='{msg.SLACK_TEAM_ID}'.",
            f"- SlackGetUserProfileTool: include token='{msg.SLACK_BOT_TOKEN}'.",
        ])
    tool_config = "\n".join(tool_config_lines) if tool_config_lines else "No external tool usage is enabled."

    # Calendar behavior: only when explicitly enabled and provided via request
    if cal_enabled:
        cal_sentence = f"If the customer wants to schedule a meeting, provide this calendar URL: {msg.CalUrl}."
    else:
        cal_sentence = "Do not mention any scheduling or calendars."

    # Define the support task with strictly specified tool parameter passing
    support_task = Task(
        description=(
            f"Respond to the customer message: {msg.message}\n"
            f"Agent context: agent_id={msg.agentId}.\n"
            f"{cal_sentence}\n"
            "ToolConfig (must be followed exactly when calling tools):\n"
            f"{tool_config}\n"
            "When calling tools, strictly pass parameters as listed in ToolConfig. Do not invent or omit parameters."
        ),
        expected_output="A well-structured, polite, and clear customer response.",
        agent=support_agent,
    )

    # Create and run the crew
    crew = Crew(
        agents=[support_agent],
        tasks=[support_task],
        process=Process.sequential
    )

    result = crew.kickoff()
    return {"response": result.raw}

@app.post("/process-file", response_model=ProcessFileResponse)
async def process_file(request: ProcessFileRequest, db: Session = Depends(get_db)):
    """
    Process a file for RAG: download, chunk, embed, and store in TiDB
    """
    try:
        # Validate input
        if not request.url or not request.filename or not request.agentId:
            raise HTTPException(status_code=400, detail="URL, filename, and agentId are required")
        
        # Check file extension
        file_extension = request.filename.lower().split('.')[-1]
        if file_extension not in ['pdf', 'txt', 'text']:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Process file
        processed_chunks = await file_processor.process_file(
            url=request.url,
            filename=request.filename,
            agent_id=request.agentId
        )
        
        # Store in database
        stored_count = 0
        for chunk_data in processed_chunks:
            embedding_record = Embeddings(
                id=str(uuid.uuid4()),  # Generate UUID for id
                agentId=chunk_data["agentId"],
                text=chunk_data["text"],
                vectorMetadata=chunk_data["metadata"],
                vector=chunk_data["vector"],
                createdAt=datetime.now()  # Set current timestamp
            )
            db.add(embedding_record)
            stored_count += 1
        
        # Commit all records
        db.commit()
        
        return ProcessFileResponse(
            success=True,
            message=f"Successfully processed and stored {stored_count} chunks from {request.filename}",
            chunks_processed=stored_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# @app.get("/embeddings/{agent_id}")
# async def get_embeddings(agent_id: str, db: Session = Depends(get_db)):
#     """
#     Get all embeddings for a specific agent
#     """
#     try:
#         embeddings = db.query(Embeddings).filter(Embeddings.agentId == agent_id).all()
        
#         result = []
#         for embedding in embeddings:
#             result.append({
#                 "id": embedding.id,
#                 "agentId": embedding.agentId,
#                 "text": embedding.text[:200] + "..." if len(embedding.text) > 200 else embedding.text,
#                 "metadata": embedding.vectorMetadata,
#                 "createdAt": embedding.createdAt
#             })
        
#         return {
#             "agent_id": agent_id,
#             "total_embeddings": len(result),
#             "embeddings": result
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error retrieving embeddings: {str(e)}")

# @app.post("/search-similar")
# async def search_similar(
#     query: str, 
#     agent_id: str, 
#     limit: int = 5, 
#     db: Session = Depends(get_db)
# ):
#     """
#     Search for similar text chunks using vector similarity
#     """
#     try:
#         # Generate embedding for the query
#         query_embedding = file_processor.generate_embeddings([query])[0]
        
#         # Search for similar vectors using cosine distance
#         distance = Embeddings.vector.cosine_distance(query_embedding).label('distance')
        
#         results = db.query(
#             Embeddings, distance
#         ).filter(
#             Embeddings.agentId == agent_id
#         ).order_by(distance).limit(limit).all()
        
#         # Format results
#         similar_chunks = []
#         for embedding, dist in results:
#             similar_chunks.append({
#                 "id": embedding.id,
#                 "text": embedding.text,
#                 "metadata": embedding.vectorMetadata,
#                 "distance": float(dist),
#                 "createdAt": embedding.createdAt
#             })
        
#         return {
#             "query": query,
#             "agent_id": agent_id,
#             "results": similar_chunks,
#             "total_results": len(similar_chunks)
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error searching similar chunks: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG processing service is running"}
