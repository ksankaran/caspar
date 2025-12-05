# Save as: src/caspar/api/main.py

"""
CASPAR API - Production-ready FastAPI application.

Provides REST endpoints for the customer service agent.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid

from caspar.config import settings, get_logger
from caspar.agent import create_checkpointer_context, create_agent, create_initial_state
from caspar.knowledge import get_retriever

logger = get_logger(__name__)

# Store active conversations (use Redis in production for horizontal scaling)
conversations: dict = {}
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize resources on startup, cleanup on shutdown.
    
    The checkpointer context manager MUST wrap the yield to keep
    the database connection open during the server's lifetime.
    """
    global agent
    
    logger.info("starting_caspar_api", version="1.0.0")
    
    # Initialize knowledge base first (validates it's ready)
    retriever = get_retriever()
    logger.info("knowledge_base_ready")
    
    # Create checkpointer context - connection stays open until shutdown
    # If DATABASE_URL is set, conversations will persist across restarts
    async with create_checkpointer_context() as checkpointer:
        # Create the agent with the checkpointer
        agent = await create_agent(checkpointer=checkpointer)
        logger.info(
            "agent_initialized",
            persistence_enabled=checkpointer is not None
        )
        
        # Server runs while we're inside this 'async with' block
        yield
        
        # Cleanup on shutdown
        logger.info("shutting_down_caspar_api")
        conversations.clear()
    
    # Checkpointer connection closes automatically here


app = FastAPI(
    title="CASPAR API",
    description="Customer Service AI Agent powered by LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if hasattr(settings, 'cors_origins') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class StartConversationRequest(BaseModel):
    """Request to start a new conversation."""
    customer_id: str = Field(..., description="Customer identifier")
    initial_message: str | None = Field(None, description="Optional first message")


class StartConversationResponse(BaseModel):
    """Response with new conversation details."""
    conversation_id: str
    message: str


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    message: str = Field(..., min_length=1, max_length=10000)


class SendMessageResponse(BaseModel):
    """Response from the agent."""
    response: str
    intent: str | None = None
    needs_escalation: bool = False
    ticket_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class ConversationStatus(BaseModel):
    """Current status of a conversation."""
    conversation_id: str
    customer_id: str
    message_count: int
    intent: str | None
    needs_escalation: bool
    created_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    agent_ready: bool


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the service is healthy."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agent_ready=agent is not None,
    )


@app.post("/conversations", response_model=StartConversationResponse, tags=["Conversations"])
async def start_conversation(request: StartConversationRequest):
    """
    Start a new conversation with CASPAR.
    
    Returns a conversation ID to use for subsequent messages.
    """
    conversation_id = f"conv-{uuid.uuid4().hex[:12]}"
    
    # Initialize conversation state
    state = create_initial_state(
        conversation_id=conversation_id,
        customer_id=request.customer_id,
    )
    
    conversations[conversation_id] = {
        "state": state,
        "customer_id": request.customer_id,
    }
    
    logger.info(
        "conversation_started",
        conversation_id=conversation_id,
        customer_id=request.customer_id,
    )
    
    # If there's an initial message, process it
    if request.initial_message:
        response = await _process_message(conversation_id, request.initial_message)
        return StartConversationResponse(
            conversation_id=conversation_id,
            message=response.response,
        )
    
    return StartConversationResponse(
        conversation_id=conversation_id,
        message="Hello! I'm CASPAR, your customer service assistant. How can I help you today?",
    )


@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=SendMessageResponse,
    tags=["Conversations"],
)
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message in an existing conversation.
    
    The agent will process the message and return a response.
    """
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found",
        )
    
    return await _process_message(conversation_id, request.message)


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationStatus,
    tags=["Conversations"],
)
async def get_conversation(conversation_id: str):
    """Get the current status of a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found",
        )
    
    conv = conversations[conversation_id]
    state = conv["state"]
    
    return ConversationStatus(
        conversation_id=conversation_id,
        customer_id=conv["customer_id"],
        message_count=len(state.get("messages", [])),
        intent=state.get("intent"),
        needs_escalation=state.get("needs_escalation", False),
        created_at=state.get("started_at", "unknown"),
    )


@app.delete("/conversations/{conversation_id}", tags=["Conversations"])
async def end_conversation(conversation_id: str):
    """End a conversation and clean up resources."""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found",
        )
    
    del conversations[conversation_id]
    
    logger.info("conversation_ended", conversation_id=conversation_id)
    
    return {"status": "ended", "conversation_id": conversation_id}


async def _process_message(conversation_id: str, message: str) -> SendMessageResponse:
    """Process a message through the agent."""
    from langchain_core.messages import HumanMessage
    
    conv = conversations[conversation_id]
    state = conv["state"]
    
    # Add the user message
    state["messages"].append(HumanMessage(content=message))
    
    # Run the agent
    config = {"configurable": {"thread_id": conversation_id}}
    
    try:
        result = await agent.ainvoke(state, config)
        
        # Update stored state
        conv["state"] = result
        
        # Extract response
        ai_response = result["messages"][-1].content if result["messages"] else "I apologize, but I couldn't process your request."
        
        logger.info(
            "message_processed",
            conversation_id=conversation_id,
            intent=result.get("intent"),
            needs_escalation=result.get("needs_escalation", False),
        )
        
        return SendMessageResponse(
            response=ai_response,
            intent=result.get("intent"),
            needs_escalation=result.get("needs_escalation", False),
            ticket_id=result.get("ticket_id"),
            metadata={
                "sentiment_score": result.get("sentiment_score"),
                "frustration_level": result.get("frustration_level"),
            },
        )
        
    except Exception as e:
        logger.error("message_processing_error", error=str(e), conversation_id=conversation_id)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your message. Please try again.",
        )