import logging
import httpx
import json
from typing import Dict, Any, Annotated, TypedDict, List
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool


class IntelligentClientState(TypedDict):
    """State schema for the intelligent client agent graph."""
    messages: Annotated[List, add_messages]
    query_type: str  # "simple", "research", "complex"
    needs_a2a: bool
    a2a_response: str
    final_response: str
    task_completed: bool
    task_id: str
    context_id: str


class A2AClientWrapper:
    """Wrapper for A2A client to handle async operations in sync context."""
    
    def __init__(self):
        self.base_url = 'http://localhost:10000'
        self.client = None
        self.agent_card = None
        self.httpx_client = None
        
    async def _initialize_client(self):
        """Initialize the A2A client asynchronously."""
        if self.client is None:
            from a2a.client import A2ACardResolver, A2AClient
            
            self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
            resolver = A2ACardResolver(httpx_client=self.httpx_client, base_url=self.base_url)
            self.agent_card = await resolver.get_agent_card()
            self.client = A2AClient(httpx_client=self.httpx_client, agent_card=self.agent_card)
    
    async def close(self):
        """Close the httpx client."""
        if self.httpx_client:
            await self.httpx_client.aclose()
            self.httpx_client = None
            self.client = None
    
    async def send_message(self, question: str, task_id: str = None, context_id: str = None):
        """Send a message using the A2A protocol."""
        from a2a.types import MessageSendParams, SendMessageRequest
        
        await self._initialize_client()
        
        message_payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': question}],
                'message_id': uuid4().hex,
            },
        }
        
        # Add task_id and context_id for conversation continuation
        if task_id:
            message_payload['message']['task_id'] = task_id
        if context_id:
            message_payload['message']['context_id'] = context_id
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**message_payload)
        )
        
        response = await self.client.send_message(request)
        result = response.root.result
        
        # Extract content from artifacts
        if hasattr(result, 'artifacts') and result.artifacts:
            artifact = result.artifacts[0]
            if hasattr(artifact, 'parts') and artifact.parts:
                part = artifact.parts[0]
                if hasattr(part, 'text'):
                    return part.text, result.id, result.context_id
                else:
                    return getattr(part, 'text', str(part)), result.id, result.context_id
        else:
            return "No response received from A2A agent", result.id, result.context_id


# Global A2A client wrapper
_a2a_client = A2AClientWrapper()


def analyze_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the user query to determine its type and whether A2A is needed."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        user_query = last_message.content.lower()
        
        # Simple decision logic for query classification
        research_keywords = ['research', 'papers', 'latest', 'developments', 'find', 'search', 'academic']
        complex_keywords = ['compare', 'analyze', 'evaluate', 'comprehensive', 'detailed']
        
        if any(keyword in user_query for keyword in research_keywords):
            query_type = "research"
            needs_a2a = True
        elif any(keyword in user_query for keyword in complex_keywords):
            query_type = "complex"
            needs_a2a = True
        else:
            query_type = "simple"
            needs_a2a = False  # Handle simple queries locally
        
        return {
            "messages": [AIMessage(content=f"Query analyzed: {query_type} type")],
            "query_type": query_type,
            "needs_a2a": needs_a2a,
            "a2a_response": "",
            "final_response": "",
            "task_completed": False,
            "task_id": state.get("task_id"),
            "context_id": state.get("context_id")
        }
    
    return state


def local_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle simple queries locally without using A2A."""
    messages = state["messages"]
    
    # Find the user's question
    user_question = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    if user_question:
        # Simple local responses for basic queries
        if "hello" in user_question.lower() or "hi" in user_question.lower():
            response = "Hello! I'm your intelligent client agent. I can help you with research questions and complex queries by connecting to specialized AI agents."
        elif "help" in user_question.lower():
            response = "I can help you with:\n- Research questions (I'll use the A2A research agent)\n- Complex analysis (I'll use the A2A agent with tools)\n- Simple questions (I'll answer directly)\n\nJust ask me anything!"
        else:
            response = f"I understand you're asking: '{user_question}'. This seems like a simple query that I can handle locally. For more complex research or analysis, I'd recommend rephrasing your question to include keywords like 'research', 'papers', 'latest developments', or 'analyze'."
        
        return {
            "messages": [AIMessage(content=response)],
            "query_type": state.get("query_type", "simple"),
            "needs_a2a": False,
            "a2a_response": "",
            "final_response": response,
            "task_completed": True,
            "task_id": state.get("task_id"),
            "context_id": state.get("context_id")
        }
    
    return state


def a2a_communication_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Communicate with the A2A server for complex queries."""
    messages = state["messages"]
    task_id = state.get("task_id")
    context_id = state.get("context_id")
    
    # Find the user's question
    user_question = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    if user_question:
        try:
            import asyncio
            response_text, new_task_id, new_context_id = asyncio.run(
                _a2a_client.send_message(user_question, task_id, context_id)
            )
            
            return {
                "messages": [AIMessage(content=f"A2A Response: {response_text}")],
                "query_type": state.get("query_type", "complex"),
                "needs_a2a": True,
                "a2a_response": response_text,
                "final_response": "",
                "task_completed": False,
                "task_id": new_task_id,
                "context_id": new_context_id
            }
        except Exception as e:
            error_msg = f"Error communicating with A2A server: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)],
                "query_type": state.get("query_type", "complex"),
                "needs_a2a": True,
                "a2a_response": error_msg,
                "final_response": error_msg,
                "task_completed": True,
                "task_id": task_id,
                "context_id": context_id
            }
    
    return state


def postprocess_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process A2A responses with client-specific enhancements."""
    a2a_response = state.get("a2a_response", "")
    query_type = state.get("query_type", "complex")
    
    if a2a_response and not a2a_response.startswith("Error"):
        # Add client-specific enhancements based on query type
        if query_type == "research":
            enhanced_response = f"🔬 Research Results:\n\n{a2a_response}\n\n---\n💡 Tip: I used the A2A research agent to find this information for you."
        elif query_type == "complex":
            enhanced_response = f"🧠 Analysis Results:\n\n{a2a_response}\n\n---\n💡 Tip: I used the A2A agent with advanced tools to provide this comprehensive analysis."
        else:
            enhanced_response = f"📋 Results:\n\n{a2a_response}\n\n---\n💡 Tip: I used the A2A agent to help answer your question."
        
        return {
            "messages": [AIMessage(content=enhanced_response)],
            "query_type": query_type,
            "needs_a2a": True,
            "a2a_response": a2a_response,
            "final_response": enhanced_response,
            "task_completed": True,
            "task_id": state.get("task_id"),
            "context_id": state.get("context_id")
        }
    
    return state


def route_decision(state: Dict[str, Any]) -> str:
    """Route to appropriate node based on query analysis."""
    needs_a2a = state.get("needs_a2a", False)
    a2a_response = state.get("a2a_response", "")
    final_response = state.get("final_response", "")
    
    if not needs_a2a:
        return "local_response"
    elif needs_a2a and not a2a_response:
        return "a2a_communication"
    elif a2a_response and not final_response:
        return "postprocess"
    else:
        return "end"


def build_intelligent_client_graph():
    """Build the intelligent LangGraph for the client agent."""
    
    # Create the graph
    workflow = StateGraph(IntelligentClientState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("local_response", local_response_node)
    workflow.add_node("a2a_communication", a2a_communication_node)
    workflow.add_node("postprocess", postprocess_response_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "analyze_query",
        route_decision,
        {
            "local_response": "local_response",
            "a2a_communication": "a2a_communication",
            "postprocess": "postprocess",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "local_response",
        lambda state: "end",
        {"end": END}
    )
    
    workflow.add_conditional_edges(
        "a2a_communication",
        route_decision,
        {
            "postprocess": "postprocess",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "postprocess",
        lambda state: "end",
        {"end": END}
    )
    
    # Set entry point
    workflow.set_entry_point("analyze_query")
    
    return workflow.compile()


class IntelligentLangGraphClientAgent:
    """An intelligent LangGraph-based client agent that makes decisions about A2A usage."""
    
    def __init__(self):
        self.graph = build_intelligent_client_graph()
        self.task_id = None
        self.context_id = None
        
    def process_query(self, query: str) -> str:
        """Process a user query through the intelligent LangGraph."""
        # Initialize the graph with the user's query
        inputs = {
            "messages": [HumanMessage(content=query)],
            "query_type": "",
            "needs_a2a": False,
            "a2a_response": "",
            "final_response": "",
            "task_completed": False,
            "task_id": self.task_id,
            "context_id": self.context_id
        }
        
        # Run the graph
        final_response = ""
        for event in self.graph.stream(inputs):
            for node_name, node_output in event.items():
                if node_name == "local_response":
                    final_response = node_output.get("final_response", "")
                elif node_name == "postprocess":
                    final_response = node_output.get("final_response", "")
                    # Update task_id for future conversations
                    self.task_id = node_output.get("task_id")
                    self.context_id = node_output.get("context_id")
                elif node_name == "a2a_communication":
                    # Update task_id for future conversations
                    self.task_id = node_output.get("task_id")
                    self.context_id = node_output.get("context_id")
        
        return final_response


def main():
    """Test the intelligent LangGraph client agent."""
    logging.basicConfig(level=logging.INFO)
    
    # Create the intelligent client agent
    client_agent = IntelligentLangGraphClientAgent()
    
    # Test with different types of queries
    test_queries = [
        # "Hello, how are you?",
        "What are the latest developments in AI?",
        # "Can you help me?",
        # "Find recent papers on transformer architectures"
    ]
    
    for query in test_queries:
        print(f"\n🤖 Client Agent: Processing: {query}")
        response = client_agent.process_query(query)
        print(f"📝 Response: {response}")
        print("-" * 50)


if __name__ == "__main__":
    main()
