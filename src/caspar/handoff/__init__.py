# Save as: src/caspar/handoff/__init__.py

"""CASPAR Human Handoff Module

Manages escalation to human agents, including queue management,
context packaging, and agent notifications.
"""

from .triggers import EscalationTrigger, check_escalation_triggers, check_sensitive_topics
from .queue import HandoffQueue, HandoffRequest, get_handoff_queue
from .context import ConversationContext, package_context_for_agent, format_context_for_display
from .notifications import notify_available_agents

__all__ = [
    "EscalationTrigger",
    "check_escalation_triggers",
    "HandoffQueue",
    "HandoffRequest",
    "get_handoff_queue",
    "ConversationContext",
    "package_context_for_agent",
    "notify_available_agents",
]