# Update: src/caspar/agent/graph.py

"""
CASPAR Agent Graph

Defines the LangGraph workflow that orchestrates the agent's behavior.
"""

from langgraph.graph import StateGraph, END

from caspar.config import get_logger
from .state import AgentState
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

logger = get_logger(__name__)


def route_by_intent(state: AgentState) -> str:
    """Route to the appropriate handler based on classified intent."""
    intent = state.get("intent") or "general"
    
    routes = {
        "faq": "handle_faq",
        "order_inquiry": "handle_order_inquiry",
        "account": "handle_account",
        "complaint": "handle_complaint",
        "handoff_request": "human_handoff",
        "general": "handle_general",
    }
    
    return routes.get(intent, "handle_general")


def route_after_sentiment(state: AgentState) -> str:
    """Route based on sentiment check - escalate if needed."""
    if state.get("needs_escalation") and state.get("intent") != "handoff_request":
        return "human_handoff"
    return "respond"


def build_graph() -> StateGraph:
    """
    Build the CASPAR agent graph.
    
    Returns:
        Configured StateGraph ready for compilation
    """
    # Create the graph with our state schema
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_faq", handle_faq)
    graph.add_node("handle_order_inquiry", handle_order_inquiry)
    graph.add_node("handle_account", handle_account)
    graph.add_node("handle_complaint", handle_complaint)
    graph.add_node("handle_general", handle_general)
    graph.add_node("check_sentiment", check_sentiment)
    graph.add_node("respond", respond)
    graph.add_node("human_handoff", human_handoff)
    
    # Set entry point
    graph.set_entry_point("classify_intent")
    
    # Add conditional routing after intent classification
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_faq": "handle_faq",
            "handle_order_inquiry": "handle_order_inquiry",
            "handle_account": "handle_account",
            "handle_complaint": "handle_complaint",
            "handle_general": "handle_general",
            "human_handoff": "human_handoff",
        }
    )
    
    # All handlers go to sentiment check
    for handler in ["handle_faq", "handle_order_inquiry", "handle_account", "handle_complaint", "handle_general"]:
        graph.add_edge(handler, "check_sentiment")
    
    # Sentiment check routes to respond or escalate
    graph.add_conditional_edges(
        "check_sentiment",
        route_after_sentiment,
        {
            "respond": "respond",
            "human_handoff": "human_handoff",
        }
    )
    
    # End nodes
    graph.add_edge("respond", END)
    graph.add_edge("human_handoff", END)
    
    return graph


async def create_agent(checkpointer=None):
    """
    Create and compile the CASPAR agent.
    
    Args:
        checkpointer: Optional checkpointer for state persistence
        
    Returns:
        Compiled agent ready for invocation
    """
    graph = build_graph()
    
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    
    return graph.compile()


def get_graph_diagram() -> str:
    """Get a Mermaid diagram of the agent graph."""
    return """
```mermaid
graph TD
    START([Start]) --> classify_intent[Classify Intent]
    
    classify_intent -->|faq| handle_faq[Handle FAQ]
    classify_intent -->|order_inquiry| handle_order_inquiry[Handle Order]
    classify_intent -->|account| handle_account[Handle Account]
    classify_intent -->|complaint| handle_complaint[Handle Complaint]
    classify_intent -->|general| handle_general[Handle General]
    classify_intent -->|handoff_request| human_handoff[Human Handoff]
    
    handle_faq --> check_sentiment[Check Sentiment]
    handle_order_inquiry --> check_sentiment
    handle_account --> check_sentiment
    handle_complaint --> check_sentiment
    handle_general --> check_sentiment
    
    check_sentiment -->|ok| respond[Generate Response]
    check_sentiment -->|escalate| human_handoff
    
    respond --> END([End])
    human_handoff --> END
```
"""