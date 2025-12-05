# Save as: src/caspar/agent/persistence.py

"""
Conversation persistence using PostgreSQL.

This module provides checkpointing functionality that allows
conversations to survive restarts and be resumed later.
"""

import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from caspar.config import get_logger

logger = get_logger(__name__)


async def create_checkpointer() -> AsyncPostgresSaver | None:
    """
    Create a PostgreSQL checkpointer for conversation persistence.
    
    Returns:
        AsyncPostgresSaver if database is configured, None otherwise
    
    Environment Variables:
        DATABASE_URL: PostgreSQL connection string
                     Format: postgresql://user:pass@host:port/dbname
    """
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        logger.warning(
            "no_database_url",
            message="DATABASE_URL not set - conversations won't persist across restarts"
        )
        return None
    
    try:
        # Create the checkpointer
        checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
        
        # Set up the required tables (safe to call multiple times)
        await checkpointer.setup()
        
        logger.info("checkpointer_initialized", database="postgresql")
        return checkpointer
        
    except Exception as e:
        logger.error(
            "checkpointer_failed",
            error=str(e),
            message="Falling back to in-memory state (no persistence)"
        )
        return None


async def create_agent_with_persistence():
    """
    Convenience function to create an agent with database persistence.
    
    This is the recommended way to create the agent for production use.
    
    Returns:
        Compiled agent with checkpointing enabled (if database available)
    
    Usage:
        agent = await create_agent_with_persistence()
        
        # Invoke with thread_id to enable persistence
        result = await agent.ainvoke(
            state,
            config={"configurable": {"thread_id": "conversation-123"}}
        )
    """
    from .graph import create_agent
    
    checkpointer = await create_checkpointer()
    return await create_agent(checkpointer=checkpointer)