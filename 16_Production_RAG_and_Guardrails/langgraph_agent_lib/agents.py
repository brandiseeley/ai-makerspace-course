"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os
import logging

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]
    guard_failures: Annotated[List[str], "guard_failures"]
    validation_attempts: Annotated[int, "validation_attempts"]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()


def create_guardrails_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    max_refinement_attempts: int = 3
):
    """Create a production-safe LangGraph agent with Guardrails validation.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        max_refinement_attempts: Maximum number of refinement attempts
        
    Returns:
        Compiled LangGraph agent with guardrails
    """
    try:
        from guardrails import Guard
        from guardrails.hub import (
            RestrictToTopic,
            DetectJailbreak,
            ProfanityFree,
            GuardrailsPII,
            LlmRagEvaluator,
            HallucinationPrompt
        )
        guardrails_available = True
    except ImportError:
        logger.warning("Guardrails not available - creating agent without guards")
        guardrails_available = False
    
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    # Initialize guards if available
    if guardrails_available:
        # Input validation guards
        topic_guard = Guard().use(
            RestrictToTopic(
                valid_topics=["student loans", "financial aid", "education financing", "loan repayment", "education"],
                invalid_topics=["investment advice", "crypto", "gambling", "politics", "illegal activities"],
                disable_classifier=True,
                disable_llm=False,
                on_fail="exception"
            )
        )
        
        jailbreak_guard = Guard().use(DetectJailbreak())
        
        pii_guard = Guard().use(
            GuardrailsPII(
                entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"],
                on_fail="fix"
            )
        )
        
        # Output validation guards
        profanity_guard = Guard().use(
            ProfanityFree(threshold=0.8, validation_method="sentence", on_fail="exception")
        )
        
        if rag_chain:
            factuality_guard = Guard().use(
                LlmRagEvaluator(
                    eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                    llm_evaluator_fail_response="hallucinated",
                    llm_evaluator_pass_response="factual",
                    llm_callable=model_name,
                    on_fail="exception",
                    on="prompt"
                )
            )
        else:
            factuality_guard = None
    
    def validate_input(state: AgentState) -> Dict[str, Any]:
        """Validate user input using guardrails."""
        if not guardrails_available:
            return {"guard_failures": [], "validation_attempts": 0}
        
        messages = state["messages"]
        user_message = None
        
        # Find the most recent human message
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"guard_failures": [], "validation_attempts": 0}
        
        guard_failures = []
        
        try:
            # Topic validation
            topic_guard.validate(user_message)
            logger.info("✅ Input passed topic validation")
        except Exception as e:
            guard_failures.append(f"Topic validation failed: {str(e)}")
            logger.warning(f"❌ Topic validation failed: {e}")
        
        try:
            # Jailbreak detection
            jailbreak_result = jailbreak_guard.validate(user_message)
            if not jailbreak_result.validation_passed:
                guard_failures.append("Jailbreak attempt detected")
                logger.warning("❌ Jailbreak attempt detected")
        except Exception as e:
            guard_failures.append(f"Jailbreak detection failed: {str(e)}")
            logger.warning(f"❌ Jailbreak detection error: {e}")
        
        try:
            # PII detection and redaction
            pii_result = pii_guard.validate(user_message)
            if pii_result.validated_output != user_message:
                logger.info("✅ PII detected and redacted")
        except Exception as e:
            guard_failures.append(f"PII validation failed: {str(e)}")
            logger.warning(f"❌ PII validation error: {e}")
        
        return {
            "guard_failures": guard_failures,
            "validation_attempts": 0
        }
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def validate_output(state: AgentState) -> Dict[str, Any]:
        """Validate model output using guardrails."""
        if not guardrails_available:
            return {"guard_failures": [], "validation_attempts": 0}
        
        messages = state["messages"]
        ai_message = None
        
        # Find the most recent AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                ai_message = msg.content
                break
        
        if not ai_message:
            return {"guard_failures": [], "validation_attempts": 0}
        
        guard_failures = []
        current_attempts = state.get("validation_attempts", 0)
        
        try:
            # Content moderation
            profanity_guard.validate(ai_message)
            logger.info("✅ Output passed content moderation")
        except Exception as e:
            guard_failures.append(f"Content moderation failed: {str(e)}")
            logger.warning(f"❌ Content moderation failed: {e}")
        
        try:
            # Factuality check (if RAG chain is available)
            if factuality_guard and rag_chain:
                factuality_guard.validate(ai_message)
                logger.info("✅ Output passed factuality check")
        except Exception as e:
            guard_failures.append(f"Factuality check failed: {str(e)}")
            logger.warning(f"❌ Factuality check failed: {e}")
        
        return {
            "guard_failures": guard_failures,
            "validation_attempts": current_attempts + 1
        }
    
    def should_continue(state: AgentState):
        """Route based on tool calls and guard failures."""
        last_message = state["messages"][-1]
        
        # Check for guard failures
        guard_failures = state.get("guard_failures", [])
        validation_attempts = state.get("validation_attempts", 0)
        
        if guard_failures and validation_attempts < max_refinement_attempts:
            logger.info(f"🔄 Guard failures detected, attempting refinement (attempt {validation_attempts + 1})")
            return "refine"
        
        if guard_failures and validation_attempts >= max_refinement_attempts:
            logger.warning("❌ Max refinement attempts reached, returning error")
            return "error"
        
        # Check for tool calls
        if getattr(last_message, "tool_calls", None):
            return "action"
        
        return END
    
    def refine_response(state: AgentState) -> Dict[str, Any]:
        """Refine the response when guards fail."""
        guard_failures = state.get("guard_failures", [])
        
        # Create a refinement prompt
        refinement_prompt = f"""
The previous response failed validation checks. Please provide a corrected response that addresses these issues:

Guard Failures:
{chr(10).join(f"- {failure}" for failure in guard_failures)}

Please ensure your response:
1. Stays on topic (student loans, financial aid, education)
2. Uses professional, appropriate language
3. Is factual and accurate
4. Does not contain sensitive personal information

Provide a helpful, safe response:
"""
        
        # Create a new message for refinement
        refinement_message = HumanMessage(content=refinement_prompt)
        
        # Get the model response
        response = model_with_tools.invoke([refinement_message])
        
        return {"messages": [response]}
    
    def handle_error(state: AgentState) -> Dict[str, Any]:
        """Handle cases where guards fail after max attempts."""
        guard_failures = state.get("guard_failures", [])
        
        error_message = AIMessage(content=f"""
I apologize, but I cannot provide a response that meets our safety and quality standards. 

The following validation checks failed:
{chr(10).join(f"- {failure}" for failure in guard_failures)}

Please try rephrasing your question to focus on student loans, financial aid, or education-related topics, and I'll be happy to help you.
""")
        
        return {"messages": [error_message]}
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    # Add nodes
    graph.add_node("validate_input", validate_input)
    graph.add_node("agent", call_model)
    graph.add_node("validate_output", validate_output)
    graph.add_node("action", tool_node)
    graph.add_node("refine", refine_response)
    graph.add_node("error", handle_error)
    
    # Set entry point
    graph.set_entry_point("validate_input")
    
    # Add edges
    graph.add_edge("validate_input", "agent")
    graph.add_edge("agent", "validate_output")
    graph.add_conditional_edges("validate_output", should_continue, {
        "action": "action",
        "refine": "refine",
        "error": "error",
        END: END
    })
    graph.add_edge("action", "agent")
    graph.add_edge("refine", "validate_output")
    graph.add_edge("error", END)
    
    return graph.compile()
