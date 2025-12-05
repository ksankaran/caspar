# Update: src/caspar/agent/__init__.py

"""CASPAR Agent Module"""

from .state import AgentState, create_initial_state, ConversationMetadata
from .graph import build_graph, create_agent, get_graph_diagram
from .persistence import create_checkpointer, create_agent_with_persistence
from .nodes import (
    classify_intent,
    handle_faq,
    handle_order_inquiry,
    handle_account,
    handle_complaint,
    handle_general,
    check_sentiment,
    respond,
    human_handoff,
)

__all__ = [
    # State
    "AgentState",
    "create_initial_state",
    "ConversationMetadata",
    # Graph
    "build_graph",
    "create_agent",
    "get_graph_diagram",
    #Persistence
    "create_checkpointer",
    "create_agent_with_persistence",
    # Nodes
    "classify_intent",
    "handle_faq",
    "handle_order_inquiry",
    "handle_account",
    "handle_complaint",
    "handle_general",
    "check_sentiment",
    "respond",
    "human_handoff",
]