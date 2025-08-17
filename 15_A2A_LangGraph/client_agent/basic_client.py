import asyncio
import logging
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)


class BasicClientAgent:
    """A very basic client agent that interacts with the A2A server."""
    
    def __init__(self, server_url: str = "http://localhost:10000"):
        self.server_url = server_url
        self.client = None
        self.agent_card = None
        
    async def setup(self):
        """Setup the A2A client connection."""
        # Create HTTP client with longer timeout
        httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        
        # Resolve the agent card from the server
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=self.server_url,
        )
        
        # Get the public agent card
        self.agent_card = await resolver.get_agent_card()
        print(f"Connected to agent: {self.agent_card.name}")
        
        # Create the A2A client
        self.client = A2AClient(
            httpx_client=httpx_client,
            agent_card=self.agent_card
        )
        
    async def ask_question(self, question: str):
        """Ask a simple question to the A2A server agent."""
        if not self.client:
            raise RuntimeError("Client not setup. Call setup() first.")
            
        # Create the message payload
        message_payload = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': question}
                ],
                'message_id': uuid4().hex,
            },
        }
        
        # Create the request
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**message_payload)
        )
        
        # Send the message and get response
        print(f"🤖 Client Agent: Asking: {question}")
        response = await self.client.send_message(request)
        
        # Extract the result
        result = response.root.result
        
        # Get the content from artifacts
        if hasattr(result, 'artifacts') and result.artifacts:
            # Get the first artifact's text content
            artifact = result.artifacts[0]
            if hasattr(artifact, 'parts') and artifact.parts:
                # Access the text content properly
                part = artifact.parts[0]
                if hasattr(part, 'text'):
                    content = part.text
                else:
                    # Try to access as dictionary
                    content = getattr(part, 'text', str(part))
                print(f"📝 Server Agent Response: {content}")
            else:
                content = "No content found in response"
                print(f"📝 Server Agent Response: {content}")
        else:
            content = "No artifacts found in response"
            print(f"📝 Server Agent Response: {content}")
        
        return result


async def main():
    """Simple test of the basic client agent."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and setup the client agent
    client_agent = BasicClientAgent()
    await client_agent.setup()
    
    # Ask a simple question
    question = "What are the latest developments in AI?"
    result = await client_agent.ask_question(question)
    
    print(f"\n✅ Basic interaction completed!")
    print(f"Task ID: {result.id}")
    print(f"Context ID: {result.context_id}")


if __name__ == "__main__":
    asyncio.run(main())
