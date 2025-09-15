from crewai.tools import tool
from fastapi import Depends
from sqlalchemy.orm import Session
import sys
import os

# Add the root directory to the path to import from root level modules

from database import get_db, Embeddings
from file_processor import FileProcessor


@tool("VectorSearchTool")
def vector_search(query: str, agent_id: str, limit: int = 5) -> str:
    """
    Performs a vector similarity search in the knowledge base.
    Returns relevant chunks of text that can help answer customer queries.
    args: query is str and the query you want to search against, agent_id is provided by user, limit is the number of chunks you want
    """
    try:
        # Get database session
        db: Session = next(get_db())
        
        # Initialize file processor to generate query embedding
        file_processor = FileProcessor()
        
        # Generate embedding for the query
        query_embedding = file_processor.generate_embeddings([query])[0]
        
        # Search for similar vectors using cosine distance
        distance = Embeddings.vector.cosine_distance(query_embedding).label('distance')
        
        results = db.query(
            Embeddings, distance
        ).filter(
            Embeddings.agentId == agent_id
        ).order_by(distance).limit(limit).all()
        
        if not results:
            return "No relevant context found."

        # Format results
        formatted = "\n".join(
            f"- {embedding.text} (score: {float(dist):.4f})"
            for embedding, dist in results
        )
        return f"Relevant context:\n{formatted}"

    except Exception as e:
        return f"Error during vector search: {str(e)}"
