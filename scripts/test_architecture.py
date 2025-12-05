# Save as: scripts/test_architecture.py

"""
Test script to verify the agent architecture is properly configured.

Note: Make sure you've run 'pip install -e .' from the project root first!
"""

import asyncio

from langchain_core.messages import HumanMessage
from caspar.agent import create_agent, create_initial_state, get_graph_diagram
from caspar.config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def test_basic_flow():
    """Test that the agent can process a simple message."""
    
    print("=" * 60)
    print("🧪 Testing CASPAR Architecture")
    print("=" * 60)
    
    # Create the agent (without persistence for this test)
    agent = await create_agent(checkpointer=None)
    print("✅ Agent compiled successfully")
    
    # Create initial state
    state = create_initial_state(
        conversation_id="test-123",
        customer_id="customer-456"
    )
    print("✅ Initial state created")
    
    # Add a test message
    state["messages"] = [HumanMessage(content="What is your return policy?")]
    
    # Run the agent
    print("\n📤 Sending test message: 'What is your return policy?'")
    print("-" * 60)
    
    config = {"configurable": {"thread_id": "test-123"}}
    
    try:
        result = await agent.ainvoke(state, config)
        
        print(f"\n📋 Results:")
        print(f"   Intent: {result.get('intent', 'Not set')}")
        print(f"   Sentiment: {result.get('sentiment_score', 'Not set')}")
        print(f"   Frustration: {result.get('frustration_level', 'Not set')}")
        print(f"   Needs Escalation: {result.get('needs_escalation', False)}")
        
        # Get the response
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            print(f"\n🤖 CASPAR's Response:")
            print(f"   {last_message.content[:200]}...")
        
        print("\n✅ Agent flow completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during agent execution: {e}")
        raise


async def test_intent_classification():
    """Test different intents are properly classified."""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Intent Classification")
    print("=" * 60)
    
    test_cases = [
        ("What's your return policy?", "faq"),
        ("Where is my order #12345?", "order_inquiry"),
        ("This is unacceptable! I want a refund!", "complaint"),
        ("Hello!", "general"),
        ("I want to speak to a human", "handoff_request"),
    ]
    
    agent = await create_agent(checkpointer=None)
    
    for message, expected_intent in test_cases:
        state = create_initial_state(conversation_id=f"test-{hash(message)}")
        state["messages"] = [HumanMessage(content=message)]
        
        config = {"configurable": {"thread_id": f"test-{hash(message)}"}}
        result = await agent.ainvoke(state, config)
        
        actual_intent = result.get("intent", "unknown")
        status = "✅" if actual_intent == expected_intent else "⚠️"
        
        print(f"{status} '{message[:40]}...'")
        print(f"   Expected: {expected_intent}, Got: {actual_intent}")


def show_graph_diagram():
    """Display the graph structure."""
    
    print("\n" + "=" * 60)
    print("📊 Graph Diagram (Mermaid format)")
    print("=" * 60)
    print(get_graph_diagram())
    print("\n💡 Tip: Paste this into https://mermaid.live to visualize")


async def main():
    """Run all architecture tests."""
    
    show_graph_diagram()
    await test_basic_flow()
    await test_intent_classification()
    
    print("\n" + "=" * 60)
    print("🎉 Architecture tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())