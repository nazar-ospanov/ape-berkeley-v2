import os
import hashlib
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor as LangChainAgentExecutor
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from a2a.types import TextPart
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message


# Load environment variables
load_dotenv()


# LangChain Tools
@tool
def math_calculator(expression: str) -> str:
    """
    Perform elementary math operations and calculations.
    
    Args:
        expression: The math expression or problem to solve (e.g., "25 + 37", "What is 15 * 8?")
    
    Returns:
        The result of the math calculation
    """
#     # Initialize ChatOpenAI for math operations
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
#     prompt = f"""You are a helpful math assistant. Please solve the following elementary math problem step by step and provide the final answer.

# Math problem: {expression}

# Please show your work and provide a clear final answer."""
    
#     try:
#         response = llm.invoke([HumanMessage(content=prompt)])
#         return response.content
#     except Exception as e:
#         return f"Error performing math operation: {str(e)}"
    return f"Performing math operation... {expression}"


@tool
def md5_hash(text: str) -> str:
    """
    Generate MD5 hash of the input text.
    
    Args:
        text: The text to hash
    
    Returns:
        The MD5 hash of the input text as a hexadecimal string
    """
    try:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    except Exception as e:
        return f"Error generating MD5 hash: {str(e)}"


@tool
def sha512_hash(text: str) -> str:
    """
    Generate SHA-512 hash of the input text.
    
    Args:
        text: The text to hash
    
    Returns:
        The SHA-512 hash of the input text as a hexadecimal string
    """
    try:
        return hashlib.sha512(text.encode('utf-8')).hexdigest()
    except Exception as e:
        return f"Error generating SHA-512 hash: {str(e)}"


class MultiPurposeToolAgent:
    """Multi-purpose tool-calling agent using LangChain's create_tool_calling_agent."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the Multi-Purpose Tool Agent.
        
        Args:
            model: OpenAI model to use for the agent.
        """
        self.llm = ChatOpenAI(model=model, temperature=0)
        
        # Define available tools
        self.tools = [math_calculator, md5_hash, sha512_hash]
        
        # Create the prompt template following LangChain's requirements
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with access to multiple tools:
1. math_calculator: For performing math calculations and solving math problems
2. md5_hash: For generating MD5 hashes of text
3. sha512_hash: For generating SHA-512 hashes of text

When a user provides a series of operations like "1. md5hash 2. sha-512 hash 3. md5 hash 4. md5 hash", 
you should perform them sequentially on the provided text, using the result of each operation as input for the next when appropriate.

For hashing operations, apply them to the original text unless the user specifies otherwise.
For math operations, solve the mathematical expression or problem provided.

Always use the appropriate tools to complete the requested operations."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the tool-calling agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # Create the agent executor
        self.agent_executor = LangChainAgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def process_request(self, user_input: str) -> str:
        """Process a user request using the tool-calling agent.
        
        Args:
            user_input: The user's input text
            
        Returns:
            The agent's response
        """
        try:
            result = self.agent_executor.invoke({"input": user_input})
            return result.get("output", "No output generated")
        except Exception as e:
            return f"Error processing request: {str(e)}"


class MultiPurposeAgentExecutor(AgentExecutor):
    """A2A Agent Executor for multi-purpose tool operations."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the Multi-Purpose Agent Executor.
        
        Args:
            model: OpenAI model to use for operations.
        """
        self.agent = MultiPurposeToolAgent(model=model)
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the multi-purpose tool request.
        
        Args:
            context: The request context containing the message
            event_queue: Event queue for sending responses
        """
        # Extract the user input from the message
        user_message = context.message
        if not user_message.parts:
            error_msg = "No message content provided."
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        # Get the text content from all parts
        user_input = ""
        for part in user_message.parts:
            if isinstance(part.root, TextPart):
                user_input += part.root.text + " "
        
        user_input = user_input.strip()
        if not user_input:
            error_msg = "No input provided. Please provide a request for the agent to process."
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        # Process the request using the tool-calling agent
        result = await self.agent.process_request(user_input)
        print(f"User Input: {user_input}")
        print(f"Agent Result: {result}")
        await event_queue.enqueue_event(new_agent_text_message(result))
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel the operation (not supported for tool operations)."""
        raise Exception('cancel not supported for tool operations')


class MathToolAgent:
    """Math Tool Agent that uses ChatOpenAI for elementary math operations."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the Math Tool Agent.
        
        Args:
            model: OpenAI model to use for math operations.
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=0  # Use low temperature for consistent math results
        )

    async def perform_math(self, expression: str) -> str:
        """Perform elementary math operations using the LLM.
        
        Args:
            expression: The math expression or problem to solve
            
        Returns:
            The result of the math operation as a string
        """
        prompt = f"""
        You are a helpful math assistant. Please solve the following elementary math problem step by step and provide the final answer.

        Math problem: {expression}

        Return only the answer, no other text.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error performing math operation: {str(e)}"


class MathToolAgentExecutor(AgentExecutor):
    """A2A Agent Executor for math tool operations."""

    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize the Math Tool Agent Executor.
        
        Args:
            openai_api_key: OpenAI API key. If None, will use OPENAI_API_KEY from environment.
            model: OpenAI model to use for math operations.
        """
        self.agent = MathToolAgent(model=model)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the math tool request.
        
        Args:
            context: The request context containing the message
            event_queue: Event queue for sending responses
        """
        # Extract the math expression from the user's message
        user_message = context.message
        if not user_message.parts:
            error_msg = "No message content provided for math operation."
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        # Get the text content from the first part
        expression = ""
        for part in user_message.parts:
            if isinstance(part.root, TextPart):
                expression += part.root.text + " "
        
        expression = expression.strip()
        if not expression:
            error_msg = "No math expression provided. Please provide a math problem to solve."
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        # Perform the math operation
        result = await self.agent.perform_math(expression)
        print(f"Expression: {expression}")
        print(f"Result: {result}")
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel the operation (not supported for synchronous math operations)."""
        raise Exception('cancel not supported for math operations')


# Keep the old HelloWorld classes for backward compatibility
class HelloWorldAgent:
    """Hello World Agent."""

    async def invoke(self) -> str:
        return 'Hello World'


class HelloWorldAgentExecutor(AgentExecutor):
    """Test AgentProxy Implementation."""

    def __init__(self):
        self.agent = HelloWorldAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        result = await self.agent.invoke()
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported')
