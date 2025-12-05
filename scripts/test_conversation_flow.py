# Save as: scripts/test_conversation_flow.py

"""
Test complete conversation flows through CASPAR.

This script tests various customer scenarios to verify
all components work together correctly.
"""

import asyncio

from langchain_core.messages import HumanMessage

from caspar.agent import create_agent, create_initial_state
from caspar.config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def test_faq_flow():
    """Test FAQ handling."""
    print("\n" + "=" * 60)
    print("📋 Test: FAQ Flow")
    print("=" * 60)
    
    agent = await create_agent()
    state = create_initial_state(
        conversation_id="test-faq-001",
        customer_id="CUST-1000"
    )
    
    # Add a FAQ question
    state["messages"] = [HumanMessage(content="What is your return policy?")]
    
    # Run the agent
    config = {"configurable": {"thread_id": "test-faq-001"}}
    result = await agent.ainvoke(state, config)
    
    print(f"Intent: {result['intent']}")
    print(f"Sentiment: {result['sentiment_score']}")
    print(f"\nResponse:\n{result['messages'][-1].content}")
    
    return result


async def test_order_inquiry_flow():
    """Test order inquiry handling."""
    print("\n" + "=" * 60)
    print("📦 Test: Order Inquiry Flow")
    print("=" * 60)
    
    agent = await create_agent()
    state = create_initial_state(
        conversation_id="test-order-001",
        customer_id="CUST-1000"
    )
    
    # Ask about an order
    state["messages"] = [HumanMessage(content="Where is my order TF-10001?")]
    
    config = {"configurable": {"thread_id": "test-order-001"}}
    result = await agent.ainvoke(state, config)
    
    print(f"Intent: {result['intent']}")
    print(f"Order Info: {result.get('order_info', {}).get('status', 'N/A')}")
    print(f"\nResponse:\n{result['messages'][-1].content}")
    
    return result


async def test_account_flow():
    """Test account inquiry handling."""
    print("\n" + "=" * 60)
    print("👤 Test: Account Inquiry Flow")
    print("=" * 60)
    
    agent = await create_agent()
    state = create_initial_state(
        conversation_id="test-account-001",
        customer_id="CUST-1001"
    )
    
    state["messages"] = [HumanMessage(content="What's my loyalty status?")]
    
    config = {"configurable": {"thread_id": "test-account-001"}}
    result = await agent.ainvoke(state, config)
    
    print(f"Intent: {result['intent']}")
    print(f"\nResponse:\n{result['messages'][-1].content}")
    
    return result


async def test_complaint_flow():
    """Test complaint handling."""
    print("\n" + "=" * 60)
    print("😤 Test: Complaint Flow")
    print("=" * 60)
    
    agent = await create_agent()
    state = create_initial_state(
        conversation_id="test-complaint-001",
        customer_id="CUST-1002"
    )
    
    state["messages"] = [HumanMessage(content="My laptop arrived damaged! The screen is cracked and I'm very upset!")]
    
    config = {"configurable": {"thread_id": "test-complaint-001"}}
    result = await agent.ainvoke(state, config)
    
    print(f"Intent: {result['intent']}")
    print(f"Sentiment: {result['sentiment_score']}")
    print(f"Frustration: {result['frustration_level']}")
    print(f"Ticket Created: {result.get('ticket_id', 'N/A')}")
    print(f"Escalated: {result.get('needs_escalation', False)}")
    print(f"\nResponse:\n{result['messages'][-1].content}")
    
    return result


async def test_escalation_flow():
    """Test human handoff request."""
    print("\n" + "=" * 60)
    print("🙋 Test: Human Handoff Flow")
    print("=" * 60)
    
    agent = await create_agent()
    state = create_initial_state(
        conversation_id="test-handoff-001",
        customer_id="CUST-1003"
    )
    
    state["messages"] = [HumanMessage(content="I want to speak to a human agent please")]
    
    config = {"configurable": {"thread_id": "test-handoff-001"}}
    result = await agent.ainvoke(state, config)
    
    print(f"Intent: {result['intent']}")
    print(f"Escalated: {result.get('needs_escalation', False)}")
    print(f"Ticket Created: {result.get('ticket_id', 'N/A')}")
    print(f"\nResponse:\n{result['messages'][-1].content}")
    
    return result


async def test_multi_turn_conversation():
    """Test a multi-turn conversation."""
    print("\n" + "=" * 60)
    print("💬 Test: Multi-Turn Conversation")
    print("=" * 60)
    
    agent = await create_agent()
    config = {"configurable": {"thread_id": "test-multi-001"}}
    
    conversation = [
        "Hi there!",
        "What laptops do you have?",
        "How much is the TechFlow Pro 15?",
        "What's the return policy if I don't like it?",
    ]
    
    state = create_initial_state(
        conversation_id="test-multi-001",
        customer_id="CUST-1000"
    )
    
    for i, message in enumerate(conversation):
        print(f"\n--- Turn {i + 1} ---")
        print(f"Customer: {message}")
        
        state["messages"].append(HumanMessage(content=message))
        result = await agent.ainvoke(state, config)
        
        # Update state with result
        state = result
        
        print(f"Intent: {result['intent']}")
        print(f"CASPAR: {result['messages'][-1].content[:200]}...")
    
    print(f"\nTotal turns: {result['turn_count']}")
    return result


async def interactive_test():
    """Interactive conversation mode."""
    print("\n" + "=" * 60)
    print("🎮 Interactive CASPAR Test")
    print("=" * 60)
    print("Chat with CASPAR! Type 'quit' to exit.\n")
    
    agent = await create_agent()
    config = {"configurable": {"thread_id": "interactive-001"}}
    
    state = create_initial_state(
        conversation_id="interactive-001",
        customer_id="CUST-1000"
    )
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "quit":
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        state["messages"].append(HumanMessage(content=user_input))
        result = await agent.ainvoke(state, config)
        state = result
        
        print(f"\nCASPAR: {result['messages'][-1].content}")
        print(f"\n[Intent: {result['intent']} | Sentiment: {result['sentiment_score']:.2f}]")


async def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CASPAR conversation flows")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--test", "-t", choices=["faq", "order", "account", "complaint", "handoff", "multi", "all"],
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    if args.interactive:
        await interactive_test()
        return
    
    tests = {
        "faq": test_faq_flow,
        "order": test_order_inquiry_flow,
        "account": test_account_flow,
        "complaint": test_complaint_flow,
        "handoff": test_escalation_flow,
        "multi": test_multi_turn_conversation,
    }
    
    if args.test == "all":
        for test_func in tests.values():
            try:
                await test_func()
            except Exception as e:
                print(f"❌ Test failed: {e}")
    else:
        await tests[args.test]()
    
    print("\n" + "=" * 60)
    print("✅ Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())