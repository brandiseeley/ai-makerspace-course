import logging
import httpx
import json
from typing import Dict, Any, Annotated, TypedDict, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool


class ClientAgentState(TypedDict):
    """State schema for the client agent graph."""
    messages: Annotated[List, add_messages]
    a2a_response: str
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
        from uuid import uuid4
        
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


def client_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Main client agent node that processes user input and decides what to ask."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        # This is a user message, we need to ask the A2A server
        user_question = last_message.content
        return {
            "messages": [AIMessage(content=f"I'll ask the A2A server: {user_question}")],
            "a2a_response": "",
            "task_completed": False,
            "task_id": state.get("task_id"),
            "context_id": state.get("context_id")
        }
    
    return state


def a2a_communication_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node that communicates with the A2A server."""
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
        # Call the A2A server directly using the wrapper
        try:
            import asyncio
            response_text, new_task_id, new_context_id = asyncio.run(
                _a2a_client.send_message(user_question, task_id, context_id)
            )
            
            return {
                "messages": [AIMessage(content=f"A2A Server Response: {response_text}")],
                "a2a_response": response_text,
                "task_completed": True,
                "task_id": new_task_id,
                "context_id": new_context_id
            }
        except Exception as e:
            error_msg = f"Error communicating with A2A server: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)],
                "a2a_response": error_msg,
                "task_completed": True,
                "task_id": task_id,
                "context_id": context_id
            }
    
    return state


def should_continue(state: Dict[str, Any]) -> str:
    """Decide whether to continue or end the graph."""
    if state.get("task_completed", False):
        return "end"
    return "continue"


def build_client_agent_graph():
    """Build the LangGraph for the client agent."""
    
    # Create the graph
    workflow = StateGraph(ClientAgentState)
    
    # Add nodes
    workflow.add_node("client_agent", client_agent_node)
    workflow.add_node("a2a_communication", a2a_communication_node)
    
    # Add edges
    workflow.add_edge("client_agent", "a2a_communication")
    workflow.add_conditional_edges(
        "a2a_communication",
        should_continue,
        {
            "continue": "client_agent",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("client_agent")
    
    return workflow.compile()


class LangGraphClientAgent:
    """A LangGraph-based client agent that interacts with the A2A server via REST API."""
    
    def __init__(self):
        self.graph = build_client_agent_graph()
        self.task_id = None
        self.context_id = None
        
    def process_query(self, query: str) -> str:
        """Process a user query through the LangGraph."""
        # Initialize the graph with the user's query
        inputs = {
            "messages": [HumanMessage(content=query)],
            "a2a_response": "",
            "task_completed": False,
            "task_id": self.task_id,
            "context_id": self.context_id
        }
        
        # Run the graph
        result = None
        for event in self.graph.stream(inputs):
            for node_name, node_output in event.items():
                if node_name == "a2a_communication":
                    result = node_output.get("a2a_response", "")
                    # Update task_id for future conversations
                    self.task_id = node_output.get("task_id")
                    break
        
        return result


def main():
    """Test the LangGraph client agent."""
    logging.basicConfig(level=logging.INFO)
    
    # Create the client agent
    client_agent = LangGraphClientAgent()
    
    # Test with a question
    question = "What are the latest developments in AI?"
    print(f"🤖 Client Agent: Asking: {question}")
    
    response = client_agent.process_query(question)
    print(f"📝 A2A Server Response: {response}")


if __name__ == "__main__":
    main()
