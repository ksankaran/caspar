# Update: src/caspar/agent/nodes.py

"""
CASPAR Agent Nodes

Each node is a function that processes state and returns updates.
This file contains all node implementations for the agent graph.
"""

from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from caspar.config import settings, get_logger
from caspar.knowledge import get_retriever
from caspar.tools import get_order_status, create_ticket, get_account_info
from .state import AgentState
from caspar.handoff import (
    check_escalation_triggers,
    get_handoff_queue,
    package_context_for_agent,
    notify_available_agents,
    format_context_for_display,
)
from caspar.tools import get_account_info

logger = get_logger(__name__)


async def classify_intent(state: AgentState) -> dict:
    """
    Classify the customer's intent from their message.
    
    Determines whether the customer is asking about:
    - faq: General questions answerable from knowledge base
    - order_inquiry: Questions about specific orders
    - complaint: Expressing dissatisfaction or issues
    - general: Other queries
    - handoff_request: Explicitly asking for human agent
    
    Returns:
        Dict with 'intent' and 'confidence' fields
    """
    logger.info("classify_intent_start", conversation_id=state["conversation_id"])
    
    llm = ChatOpenAI(
        model=settings.default_model,
        api_key=settings.openai_api_key,
        temperature=0  # Deterministic for classification
    )
    
    # Get the latest message
    messages = state["messages"]
    if not messages:
        return {
            "intent": "general",
            "confidence": 0.5,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    last_message = messages[-1].content
    
    classification_prompt = f"""Classify this customer service message into ONE of these categories:

- faq: Questions about policies, products, how things work (e.g., "What's your return policy?", "Do you ship internationally?")
- order_inquiry: Questions about a specific order (e.g., "Where's my order?", "When will order #12345 arrive?")
- complaint: Expressing dissatisfaction, frustration, or problems (e.g., "This is broken!", "I've been waiting forever")
- account: Questions about their account, points, profile (e.g., "What's my loyalty status?", "Update my address")
- handoff_request: Explicitly asking for human help (e.g., "Let me talk to a person", "I need a human agent")
- general: Anything else, greetings, unclear intent

Customer message: "{last_message}"

Respond with ONLY the category name, nothing else."""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    
    intent = response.content.strip().lower()
    
    # Validate intent
    valid_intents = ["faq", "order_inquiry", "complaint", "account", "handoff_request", "general"]
    if intent not in valid_intents:
        intent = "general"
    
    logger.info(
        "classify_intent_complete",
        conversation_id=state["conversation_id"],
        intent=intent
    )
    
    return {
        "intent": intent,
        "confidence": 0.85,  # Simplified - could calculate from model logprobs
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


async def handle_faq(state: AgentState) -> dict:
    """
    Handle FAQ-type questions using the knowledge base.
    
    Retrieves relevant information from ChromaDB and prepares
    context for response generation.
    
    Returns:
        Dict with 'retrieved_context' field
    """
    logger.info("handle_faq_start", conversation_id=state["conversation_id"])
    
    messages = state["messages"]
    if not messages:
        return {
            "retrieved_context": None,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    query = messages[-1].content
    
    # Retrieve relevant documents
    retriever = get_retriever()
    documents = retriever.retrieve(query=query, k=4)
    
    # Format context for the LLM
    context = retriever.format_context(documents)
    
    logger.info(
        "handle_faq_complete",
        conversation_id=state["conversation_id"],
        documents_found=len(documents)
    )
    
    return {
        "retrieved_context": context,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


async def handle_order_inquiry(state: AgentState) -> dict:
    """
    Handle order-related inquiries.
    
    Extracts order ID from the message and looks up order status.
    
    Returns:
        Dict with 'order_info' field containing order details
    """
    logger.info("handle_order_inquiry_start", conversation_id=state["conversation_id"])
    
    messages = state["messages"]
    if not messages:
        return {
            "order_info": None,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    last_message = messages[-1].content
    customer_id = state.get("customer_id")
    
    # Use LLM to extract order ID from message
    llm = ChatOpenAI(
        model=settings.default_model,
        api_key=settings.openai_api_key,
        temperature=0
    )
    
    extract_prompt = f"""Extract the order ID from this customer message.
Order IDs look like: TF-10001, TF-12345, or just numbers like 10001, 12345

Customer message: "{last_message}"

If you find an order ID, respond with ONLY the ID (include TF- prefix if not present).
If no order ID is found, respond with "NONE"."""

    response = llm.invoke([HumanMessage(content=extract_prompt)])
    extracted_id = response.content.strip()
    
    order_info = None
    
    if extracted_id and extracted_id != "NONE":
        # Look up the order
        result = get_order_status(extracted_id, customer_id)
        
        if result["found"]:
            order_info = {
                "order_id": extracted_id,
                "status": result["order"]["status"],
                "summary": result["summary"],
                "full_order": result["order"]
            }
        else:
            order_info = {
                "order_id": extracted_id,
                "error": result["error"]
            }
    else:
        order_info = {
            "error": "I couldn't find an order number in your message. Could you please provide your order ID? It looks like TF-XXXXX."
        }
    
    logger.info(
        "handle_order_inquiry_complete",
        conversation_id=state["conversation_id"],
        order_found=order_info.get("status") is not None if order_info else False
    )
    
    return {
        "order_info": order_info,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


async def handle_account(state: AgentState) -> dict:
    """
    Handle account-related inquiries.
    
    Retrieves customer account information.
    
    Returns:
        Dict with account information
    """
    logger.info("handle_account_start", conversation_id=state["conversation_id"])
    
    customer_id = state.get("customer_id")
    
    if not customer_id:
        return {
            "retrieved_context": "I'd be happy to help with your account, but I need to verify your identity first. Could you please provide your customer ID or the email associated with your account?",
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    result = get_account_info(customer_id)
    
    if result["found"]:
        context = result["summary"]
    else:
        context = result["error"]
    
    logger.info(
        "handle_account_complete",
        conversation_id=state["conversation_id"],
        account_found=result["found"]
    )
    
    return {
        "retrieved_context": context,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


async def handle_complaint(state: AgentState) -> dict:
    """
    Handle customer complaints.
    
    Acknowledges the issue, retrieves relevant information,
    and creates a ticket if needed.
    
    Returns:
        Dict with complaint handling context and ticket info
    """
    logger.info("handle_complaint_start", conversation_id=state["conversation_id"])
    
    messages = state["messages"]
    customer_id = state.get("customer_id") or "UNKNOWN"
    conversation_id = state.get("conversation_id")
    
    if not messages:
        return {
            "retrieved_context": None,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    last_message = messages[-1].content
    
    # Try to get relevant KB info for the complaint
    retriever = get_retriever()
    documents = retriever.retrieve(query=last_message, k=2)
    kb_context = retriever.format_context(documents) if documents else ""
    
    # Create a ticket for the complaint
    ticket_result = create_ticket(
        customer_id=customer_id,
        category="general",  # Could use LLM to classify more specifically
        subject=f"Customer Complaint: {last_message[:50]}...",
        description=last_message,
        priority="high",  # Complaints get high priority
        conversation_id=conversation_id,
    )
    
    context_parts = []
    
    if kb_context:
        context_parts.append(f"Relevant Information:\n{kb_context}")
    
    context_parts.append(f"\n{ticket_result['confirmation']}")
    
    logger.info(
        "handle_complaint_complete",
        conversation_id=state["conversation_id"],
        ticket_id=ticket_result["ticket"]["ticket_id"]
    )
    
    return {
        "retrieved_context": "\n\n".join(context_parts),
        "ticket_id": ticket_result["ticket"]["ticket_id"],
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


async def handle_general(state: AgentState) -> dict:
    """
    Handle general inquiries that don't fit other categories.
    
    Falls back to knowledge base search for potential matches.
    
    Returns:
        Dict with any found context
    """
    logger.info("handle_general_start", conversation_id=state["conversation_id"])
    
    messages = state["messages"]
    if not messages:
        return {
            "retrieved_context": None,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    query = messages[-1].content
    
    # Try knowledge base anyway - might find something useful
    retriever = get_retriever()
    documents = retriever.retrieve(query=query, k=2)
    
    context = None
    if documents:
        context = retriever.format_context(documents)
    
    logger.info(
        "handle_general_complete",
        conversation_id=state["conversation_id"],
        found_context=context is not None
    )
    
    return {
        "retrieved_context": context,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


# Update in: src/caspar/agent/nodes.py

async def check_sentiment(state: AgentState) -> dict:
    """Analyze customer sentiment and check all escalation triggers."""
    
    logger.info("check_sentiment_start", conversation_id=state["conversation_id"])
    
    messages = state["messages"]
    if not messages:
        return {"sentiment_score": 0.0, "frustration_level": "low", "last_updated": datetime.now(timezone.utc).isoformat()}
    
    # Get last few messages for context
    recent_messages = messages[-3:] if len(messages) >= 3 else messages
    conversation_text = "\n".join([
        f"{'Customer' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in recent_messages
    ])
    
    llm = ChatOpenAI(model=settings.default_model, api_key=settings.openai_api_key, temperature=0)
    
    sentiment_prompt = f"""Analyze the customer's emotional state in this conversation.

Conversation:
{conversation_text}

Provide your analysis in this exact format:
SENTIMENT: [number from -1.0 to 1.0, where -1 is very negative, 0 is neutral, 1 is very positive]
FRUSTRATION: [low, medium, or high]"""

    response = llm.invoke([HumanMessage(content=sentiment_prompt)])
    
    # Parse response
    sentiment_score = 0.0
    frustration_level = "low"
    
    for line in response.content.strip().split("\n"):
        if line.startswith("SENTIMENT:"):
            try:
                sentiment_score = float(line.split(":")[1].strip())
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
            except ValueError:
                pass
        elif line.startswith("FRUSTRATION:"):
            level = line.split(":")[1].strip().lower()
            if level in ["low", "medium", "high"]:
                frustration_level = level
    
    result = {
        "sentiment_score": sentiment_score,
        "frustration_level": frustration_level,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    # Check for sensitive topics in the last message
    from caspar.handoff import check_sensitive_topics
    last_message = messages[-1].content if messages else ""
    if check_sensitive_topics(last_message):
        result["needs_escalation"] = True
        result["escalation_reason"] = "Sensitive topic detected - requires human handling"
        logger.warning("sensitive_topic_detected", conversation_id=state["conversation_id"])
    
    # Check if escalation needed based on sentiment
    elif sentiment_score < settings.sentiment_threshold or frustration_level == "high":
        result["needs_escalation"] = True
        result["escalation_reason"] = f"High frustration detected (sentiment: {sentiment_score}, frustration: {frustration_level})"
        logger.warning("escalation_triggered", conversation_id=state["conversation_id"])
    
    logger.info("check_sentiment_complete", sentiment_score=sentiment_score, frustration_level=frustration_level)
    
    return result

async def respond(state: AgentState) -> dict:
    """
    Generate the final response to the customer.
    
    Synthesizes all context gathered by previous nodes into
    a helpful, friendly response.
    
    Returns:
        Dict with new AI message added to messages
    """
    logger.info("respond_start", conversation_id=state["conversation_id"])
    
    llm = ChatOpenAI(
        model=settings.default_model,
        api_key=settings.openai_api_key,
        temperature=0.7
    )
    
    # Build system prompt with context
    system_prompt = """You are CASPAR, TechFlow's friendly customer service assistant.

TechFlow is an online electronics retailer. You help customers with:
- Product questions
- Order status and tracking
- Returns and refunds  
- Shipping information
- Technical support

Guidelines:
- Be warm, helpful, and professional
- Keep responses concise but complete
- If you have specific information (order details, policies), share it clearly
- If you don't have enough information, ask clarifying questions
- Never make up order numbers, tracking info, or policies
- If the customer seems frustrated, acknowledge their feelings

"""
    
    # Add context from previous nodes
    context_parts = []
    
    if state.get("retrieved_context"):
        context_parts.append(f"Relevant Information:\n{state['retrieved_context']}")
    
    if state.get("order_info"):
        order = state["order_info"]
        if "summary" in order:
            context_parts.append(f"Order Information:\n{order['summary']}")
        elif "error" in order:
            context_parts.append(f"Order Lookup Result: {order['error']}")
    
    if state.get("ticket_id"):
        context_parts.append(f"A support ticket has been created: {state['ticket_id']}")
    
    if context_parts:
        system_prompt += "\nContext for this response:\n" + "\n\n".join(context_parts)
    
    # Build messages for the LLM
    llm_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            llm_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            llm_messages.append({"role": "assistant", "content": msg.content})
    
    response = llm.invoke(llm_messages)
    
    logger.info(
        "respond_complete",
        conversation_id=state["conversation_id"],
        response_length=len(response.content)
    )
    
    return {
        "messages": [AIMessage(content=response.content)],
        "turn_count": (state.get("turn_count") or 0) + 1,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


# Update in: src/caspar/agent/nodes.py

async def human_handoff(state: AgentState) -> dict:
    """
    Handle escalation to a human agent.
    
    This node:
    1. Checks escalation triggers
    2. Creates a handoff request
    3. Packages context for the human agent
    4. Notifies available agents
    5. Informs the customer
    """
    logger.info("human_handoff_start", conversation_id=state["conversation_id"])
    
    customer_id = state.get("customer_id") or "UNKNOWN"
    conversation_id = state.get("conversation_id")
    
    # Get customer info for context
    customer_info = None
    if customer_id != "UNKNOWN":
        account_result = get_account_info(customer_id)
        if account_result["found"]:
            customer_info = account_result["account"]
    
    # Check escalation triggers
    customer_tier = customer_info.get("loyalty_tier") if customer_info else None
    escalation_result = check_escalation_triggers(state, customer_tier)
    
    # Create ticket for tracking
    from caspar.tools import create_ticket
    ticket_result = create_ticket(
        customer_id=customer_id,
        category="general",
        subject="Human Agent Requested",
        description=escalation_result.reason,
        priority=escalation_result.priority,
        conversation_id=conversation_id,
    )
    
    # Add to handoff queue
    queue = get_handoff_queue()
    handoff_request = queue.add(
        conversation_id=conversation_id,
        customer_id=customer_id,
        priority=escalation_result.priority,
        triggers=[t.value for t in escalation_result.triggers],
        reason=escalation_result.reason,
        ticket_id=ticket_result["ticket"]["ticket_id"],
    )
    
    # Package context for human agent
    state_with_triggers = {
        **state,
        "escalation_triggers": [t.value for t in escalation_result.triggers],
    }
    context = package_context_for_agent(
        state=state_with_triggers,
        request_id=handoff_request.request_id,
        customer_info=customer_info,
    )
    
    # Notify available agents
    notifications = notify_available_agents(handoff_request, context)
    
    # Log the full context (in production, this would go to the agent dashboard)
    context_display = format_context_for_display(context)
    logger.info("handoff_context_prepared", context_length=len(context_display))
    
    # Build customer-facing message
    position = queue.get_queue_position(handoff_request.request_id)
    wait_time = handoff_request.estimated_wait or 5
    
    handoff_message = _build_handoff_message(
        ticket_id=ticket_result["ticket"]["ticket_id"],
        position=position,
        wait_time=wait_time,
        priority=escalation_result.priority,
    )
    
    logger.info(
        "human_handoff_complete",
        request_id=handoff_request.request_id,
        ticket_id=ticket_result["ticket"]["ticket_id"],
        agents_notified=len(notifications)
    )
    
    return {
        "messages": [AIMessage(content=handoff_message)],
        "needs_escalation": True,
        "escalation_reason": escalation_result.reason,
        "ticket_id": ticket_result["ticket"]["ticket_id"],
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


def _build_handoff_message(
    ticket_id: str,
    position: int,
    wait_time: int,
    priority: str,
) -> str:
    """Build the customer-facing handoff message."""
    
    priority_messages = {
        "urgent": "I've flagged this as urgent, and a team member will be with you very shortly.",
        "high": "I've marked this as high priority. A team member will be with you soon.",
        "medium": "A team member will be with you as soon as possible.",
        "low": "A team member will reach out to help you.",
    }
    
    message_parts = [
        "I understand you'd like to speak with a human agent, and I've arranged that for you.",
        "",
        f"**Your Reference Number: {ticket_id}**",
        "",
        priority_messages.get(priority, priority_messages["medium"]),
        "",
    ]
    
    if position > 0:
        message_parts.append(f"You're currently #{position} in our queue.")
    
    message_parts.extend([
        f"Estimated wait time: approximately {wait_time} minutes.",
        "",
        "While you wait:",
        "• You don't need to stay on this chat - we'll reach out to you",
        "• You can reference your ticket number in any follow-up",
        "• Our team has the full context of our conversation",
        "",
        "Is there anything else I can help you with while you wait?",
    ])
    
    return "\n".join(message_parts)