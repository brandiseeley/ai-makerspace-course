#!/usr/bin/env python3
"""
Test script for the LangGraph-based client agent using the A2A protocol.
Make sure your A2A server is running first with: uv run python -m app
"""

from client_agent.langgraph_client import LangGraphClientAgent


def test_simple_langgraph_client():
    """Test the LangGraph client agent using the A2A protocol."""
    print("🚀 Starting LangGraph Client Agent Test (A2A Protocol)")
    print("=" * 60)
    
    try:
        # Create the LangGraph client agent
        client_agent = LangGraphClientAgent()
        
        # Test with a question
        question = "What are the latest developments in artificial intelligence?"
        print(f"\n🤖 LangGraph Client Agent: Processing query: {question}")
        
        response = client_agent.process_query(question)
        
        print(f"\n📝 A2A Server Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        print(f"\n✅ LangGraph client agent test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure your A2A server is running with:")
        print("   uv run python -m app")
        print("   The server should be available at http://localhost:10000")


if __name__ == "__main__":
    test_simple_langgraph_client()
