import os
import hashlib
import time
import re
import signal
import atexit
import base64
import io
import subprocess
import tempfile
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor as LangChainAgentExecutor
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup  # Not currently used, but available for HTML parsing
from a2a.types import TextPart, FilePart
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_task


# Load environment variables
load_dotenv()


# Global webdriver instance for persistence across tool calls
_driver = None

def get_webdriver():
    """Get or create a webdriver instance."""
    global _driver
    if _driver is None:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # Use new headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--remote-debugging-port=9222")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        
        # Try to use system Chromium/Chrome driver
        try:
            # First try system chromedriver
            service = Service("/usr/bin/chromedriver")
            _driver = webdriver.Chrome(service=service, options=chrome_options)
        except:
            try:
                # Fallback to ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                _driver = webdriver.Chrome(service=service, options=chrome_options)
            except Exception as e:
                raise Exception(f"Failed to initialize Chrome/Chromium WebDriver: {str(e)}")
    return _driver

def close_webdriver():
    """Close the webdriver instance."""
    global _driver
    if _driver:
        try:
            _driver.quit()
        except:
            pass  # Ignore errors during cleanup
        _driver = None

# Register cleanup handlers
def cleanup_handler(signum=None, frame=None):
    """Handle cleanup on exit."""
    close_webdriver()

signal.signal(signal.SIGTERM, cleanup_handler)
signal.signal(signal.SIGINT, cleanup_handler)
atexit.register(cleanup_handler)

def minimax(board, depth, is_maximizing, player='X', opponent='O'):
    """
    Minimax algorithm for optimal tic-tac-toe play.
    Returns the best score for the current board state.
    """
    # Check for terminal states
    winner = check_winner(board)
    if winner == player:
        return 10 - depth
    elif winner == opponent:
        return depth - 10
    elif is_board_full(board):
        return 0
    
    if is_maximizing:
        best_score = float('-inf')
        for i in range(9):
            if board[i] == '':
                board[i] = player
                score = minimax(board, depth + 1, False, player, opponent)
                board[i] = ''
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i] == '':
                board[i] = opponent
                score = minimax(board, depth + 1, True, player, opponent)
                board[i] = ''
                best_score = min(score, best_score)
        return best_score

def get_best_move(board, player='X', opponent='O'):
    """Get the best move for the current player using minimax."""
    best_score = float('-inf')
    best_move = -1
    
    for i in range(9):
        if board[i] == '':
            board[i] = player
            score = minimax(board, 0, False, player, opponent)
            board[i] = ''
            if score > best_score:
                best_score = score
                best_move = i
    
    return best_move

def check_winner(board):
    """Check if there's a winner on the board."""
    # Winning combinations
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != '':
            return board[combo[0]]
    return None

def is_board_full(board):
    """Check if the board is full."""
    return '' not in board

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


@tool
def analyze_image_for_cat_or_dog(image_base64: str) -> str:
    """
    Analyze an image to detect if it contains a cat or a dog using OpenAI Vision API.
    
    Args:
        image_base64: Base64 encoded image data
    
    Returns:
        "cat" if a cat is detected, "dog" if a dog is detected, or "neither" if neither is found
    """
    try:
        # Initialize ChatOpenAI with vision capabilities
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Create the message with image
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Look at this image and determine if there is a cat or a dog present. Respond with exactly one word: 'cat' if you see a cat, 'dog' if you see a dog, or 'neither' if you see neither a cat nor a dog."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        )
        
        # Get response from OpenAI Vision
        response = llm.invoke([message])
        result = response.content.lower().strip()
        
        # Ensure we return only valid responses
        if "cat" in result:
            return "cat"
        elif "dog" in result:
            return "dog"
        else:
            return "neither"
            
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


@tool
def record_in_memory(text: str) -> str:
    """
    Record information in persistent memory that survives across agent sessions.
    
    Args:
        text: The text to remember and store in memory
        
    Returns:
        str: Confirmation message about what was recorded
    """
    try:
        memory_file = "/app/memory.txt"
        
        # Append the new memory entry with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory_entry = f"[{timestamp}] {text}\n"
        
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(memory_entry)
        
        return f"Successfully recorded in memory: {text}"
        
    except Exception as e:
        return f"Error recording memory: {str(e)}"


@tool
def check_memory() -> str:
    """
    Retrieve all stored memories from previous sessions.
    
    Returns:
        str: All stored memories or message if no memories exist
    """
    try:
        memory_file = "/app/memory.txt"
        
        # Check if memory file exists
        if not os.path.exists(memory_file):
            return "No memories found. Memory file doesn't exist yet."
        
        # Read all memories
        with open(memory_file, "r", encoding="utf-8") as f:
            memories = f.read().strip()
        
        if not memories:
            return "No memories stored yet."
        
        return f"Stored memories:\n\n{memories}"
        
    except Exception as e:
        return f"Error reading memory: {str(e)}"


@tool
def generate_and_execute_code(problem_description: str) -> str:
    """
    Generate Python code to solve a given coding problem and execute it safely.
    
    Args:
        problem_description: Description of the coding problem to solve
        
    Returns:
        str: The output of the executed code or error message
    """
    try:
        # Load OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return "Error: OpenAI API key not found"
        
        # Create ChatOpenAI instance
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=openai_api_key
        )
        
        # Create a prompt for code generation
        code_generation_prompt = f"""
You are a Python programming expert. Given the following problem description, write Python code to solve it.

Problem: {problem_description}

Requirements:
1. Write clean, efficient Python code
2. Include comments explaining the logic
3. Make sure the code handles edge cases
4. Print the final result clearly
5. Don't use any external libraries unless absolutely necessary (stick to standard library)
6. The code should be complete and executable

Respond with ONLY the Python code, no explanations or markdown formatting.
"""
        
        # Generate code
        response = llm.invoke([HumanMessage(content=code_generation_prompt)])
        generated_code = response.content.strip()
        
        # Clean up the code (remove markdown formatting if present)
        if generated_code.startswith("```python"):
            generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        elif generated_code.startswith("```"):
            generated_code = generated_code.replace("```", "").strip()
        
        print(f"Generated code:\n{generated_code}")
        
        # Execute the code safely in a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(generated_code)
            temp_file_path = temp_file.name
        
        try:
            # Execute the code with a timeout
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=tempfile.gettempdir()  # Run in a safe directory
            )
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if result.stderr:
                    output += f"\nWarnings: {result.stderr.strip()}"
                return f"Code executed successfully!\n\nGenerated Code:\n{generated_code}\n\nOutput:\n{output}"
            else:
                error_output = result.stderr.strip() if result.stderr else "Unknown error"
                return f"Code execution failed!\n\nGenerated Code:\n{generated_code}\n\nError:\n{error_output}"
                
        except subprocess.TimeoutExpired:
            os.unlink(temp_file_path)
            return f"Code execution timed out (30s limit)!\n\nGenerated Code:\n{generated_code}"
        except Exception as exec_error:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            return f"Error executing code: {str(exec_error)}\n\nGenerated Code:\n{generated_code}"
            
    except Exception as e:
        return f"Error generating or executing code: {str(e)}"


@tool
def start_tictactoe_game(url: str) -> str:
    """
    Start a new tic-tac-toe game session by navigating to the URL.
    This should be called first before other tic-tac-toe operations.
    
    Args:
        url: The URL of the tic-tac-toe game website
    
    Returns:
        Success message and initial board state
    """
    try:
        driver = get_webdriver()
        print(f"Starting new game session at {url}")
        driver.get(url)
        
        # Wait for the game board to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "gameBoard"))
        )
        
        # Get initial board state
        cells = driver.find_elements(By.CSS_SELECTOR, ".cell")
        board = [''] * 9
        
        for cell in cells:
            index = int(cell.get_attribute("data-index"))
            text = cell.text.strip()
            if text in ['X', 'O']:
                board[index] = text
        
        return f"Game session started. Initial board state: {board}"
        
    except Exception as e:
        return f"Error starting game session: {str(e)}"


@tool
def parse_tictactoe_board(url: str) -> str:
    """
    Parse the current state of the tic-tac-toe board from the website.
    
    Args:
        url: The URL of the tic-tac-toe game website
    
    Returns:
        A string representation of the board state and game status
    """
    try:
        driver = get_webdriver()
        
        # Only navigate if we're not already on the correct page
        current_url = driver.current_url
        if not current_url or url not in current_url:
            print(f"Navigating to {url}")
            driver.get(url)
            # Wait for the game board to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "gameBoard"))
            )
        else:
            print(f"Already on correct page: {current_url}")
        
        # Get the board state
        cells = driver.find_elements(By.CSS_SELECTOR, ".cell")
        board = [''] * 9
        
        for cell in cells:
            index = int(cell.get_attribute("data-index"))
            text = cell.text.strip()
            if text in ['X', 'O']:
                board[index] = text
            elif cell.get_attribute("disabled"):
                # If disabled but no text, might be a placed piece
                board[index] = text if text else ''
        
        # Check if board is full
        board_full = all(cell.get_attribute("disabled") for cell in cells)
        
        if board_full and '' in board:
            # If board appears full but we have empty spots, try to reset
            try:
                reset_button = driver.find_element(By.ID, "resetButton")
                reset_button.click()
                time.sleep(1)
                
                # Re-parse after reset
                cells = driver.find_elements(By.CSS_SELECTOR, ".cell")
                board = [''] * 9
                for cell in cells:
                    index = int(cell.get_attribute("data-index"))
                    text = cell.text.strip()
                    if text in ['X', 'O']:
                        board[index] = text
                
                return f"Board reset and parsed: {board}"
            except:
                pass
        
        return f"Board state: {board}"
        
    except Exception as e:
        return f"Error parsing board: {str(e)}"


@tool  
def place_x_on_board(position: int) -> str:
    """
    Place an X on the tic-tac-toe board at the specified position.
    
    Args:
        position: The position (0-8) where to place X
    
    Returns:
        Success or error message
    """
    try:
        driver = get_webdriver()
        
        # Find the cell with the specified data-index
        cell = driver.find_element(By.CSS_SELECTOR, f'.cell[data-index="{position}"]')
        
        # Check if cell is available
        if cell.get_attribute("disabled"):
            return f"Position {position} is already occupied"
        
        # Click the cell
        cell.click()
        
        # Wait 2 seconds as requested
        time.sleep(2)
        
        return f"Successfully placed X at position {position}"
        
    except Exception as e:
        return f"Error placing X at position {position}: {str(e)}"


@tool
def check_win_and_extract_secret() -> str:
    """
    Check if the game is won and extract the secret number if available.
    
    Returns:
        The secret number if won, or status message if not won yet
    """
    try:
        driver = get_webdriver()
        
        # Check for congratulations message
        try:
            congratulations = driver.find_element(By.ID, "congratulations")
            if "show" in congratulations.get_attribute("class"):
                # Extract the secret number
                text = congratulations.text
                print(f"Congratulations text found: {text}")
                # Look for 14-digit number pattern
                match = re.search(r'\b\d{14}\b', text)
                if match:
                    return match.group(0)
                else:
                    # Look for any number in the congratulations message
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        return max(numbers, key=len)  # Return the longest number found
                    return f"Won but no secret number found in: {text}"
            else:
                return "Game not won yet - congratulations not showing"
        except:
            return "Game not won yet - no congratulations element found"
            
    except Exception as e:
        return f"Error checking win status: {str(e)}"


# @tool
# def play_tictactoe_until_win(url: str) -> str:
#     """
#     Automated tool to play tic-tac-toe until winning and extract secret.
    
#     Args:
#         url: The URL of the tic-tac-toe game website
    
#     Returns:
#         The secret number when won
#     """
#     try:
#         driver = get_webdriver()
#         driver.get(url)
        
#         max_games = 10  # Prevent infinite loops
        
#         for game_num in range(max_games):
#             # Check if already won
#             win_result = check_win_and_extract_secret(url)
#             if win_result and win_result.isdigit():
#                 return win_result
            
#             # Parse current board
#             board_result = parse_tictactoe_board(url)
#             if "Board state:" in board_result:
#                 board_str = board_result.replace("Board state: ", "")
#                 board = eval(board_str)  # Convert string representation back to list
                
#                 # Get best move
#                 best_move = get_best_move(board)
#                 if best_move != -1:
#                     # Place X at best position
#                     place_result = place_x_on_board(best_move)
                    
#                     # Check for win again after move
#                     win_result = check_win_and_extract_secret(url)
#                     if win_result and win_result.isdigit():
#                         return win_result
#                 else:
#                     # Board full, try reset
#                     try:
#                         reset_button = driver.find_element(By.ID, "resetButton")
#                         reset_button.click()
#                         time.sleep(1)
#                     except:
#                         pass
        
#         return "Could not win after maximum attempts"
        
#     except Exception as e:
#         return f"Error in automated play: {str(e)}"


class MultiPurposeToolAgent:
    """Multi-purpose tool-calling agent using LangChain's create_tool_calling_agent."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the Multi-Purpose Tool Agent.
        
        Args:
            model: OpenAI model to use for the agent.
        """
        self.llm = ChatOpenAI(model=model, temperature=0)
        
        # Define available tools
        self.tools = [
            math_calculator, 
            md5_hash, 
            sha512_hash,
            analyze_image_for_cat_or_dog,
            generate_and_execute_code,
            record_in_memory,
            check_memory,
            start_tictactoe_game,
            parse_tictactoe_board,
            place_x_on_board,
            check_win_and_extract_secret,
            # play_tictactoe_until_win
        ]
        
        # Create the prompt template following LangChain's requirements
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with access to multiple tools:

MATH & CRYPTO TOOLS:
1. math_calculator: For performing math calculations and solving math problems
2. md5_hash: For generating MD5 hashes of text
3. sha512_hash: For generating SHA-512 hashes of text

IMAGE UNDERSTANDING TOOLS:
4. analyze_image_for_cat_or_dog: Analyze images to detect cats or dogs using OpenAI Vision API

CODE EXECUTION TOOLS:
5. generate_and_execute_code: Generate Python code to solve coding problems and execute it safely

MEMORY TOOLS:
6. record_in_memory: Store information in persistent memory that survives across sessions
7. check_memory: Retrieve all stored memories from previous sessions

WEB AUTOMATION TOOLS:
8. start_tictactoe_game: Start a new game session (call this first)
9. parse_tictactoe_board: Parse the current state of a tic-tac-toe board from a website
10. place_x_on_board: Place an X at a specific position (0-8) on the tic-tac-toe board
11. check_win_and_extract_secret: Check if the game is won and extract the secret number (no URL needed)

USAGE INSTRUCTIONS:
- For sequential operations like "1. md5hash 2. sha-512 hash 3. md5 hash", perform them on the original text
- For image analysis: When you see [IMAGE_DATA:...] in the input, extract the base64 data and use analyze_image_for_cat_or_dog
- For coding problems: Use generate_and_execute_code to write and run Python code for any programming challenge
- For memory operations: Use record_in_memory when user asks to remember something, check_memory when they want to recall
- For tic-tac-toe games: ALWAYS start with start_tictactoe_game(url) to establish the session
- The tic-tac-toe board uses positions 0-8 in this layout: [[0,1,2], [3,4,5], [6,7,8]]
- Always extract the complete secret number when winning tic-tac-toe games
- The agent plays as X and uses optimal strategy to win. Agent plays until it wins and returns a secret back.

WORKFLOW FOR TIC-TAC-TOE:
1. start_tictactoe_game(url) - Initialize game session
2. check_win_and_extract_secret() - Check if already won
3. parse_tictactoe_board(url) - Get current board state
4. place_x_on_board(position) - Make optimal move
5. Repeat steps 2-4 until victory

Always use the appropriate tools to complete the requested operations.
            """),
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
            handle_parsing_errors=True,
            max_iterations=50
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
        
        # Get the text content from all parts and check for images
        user_input = ""
        image_data = None
        
        for part in user_message.parts:
            if isinstance(part.root, TextPart):
                user_input += part.root.text + " "
            elif isinstance(part.root, FilePart):
                file_part = part.root
                # Check if it's an image file
                if file_part.file.mime_type and file_part.file.mime_type.startswith('image/'):
                    print(f"Detected image file: {file_part.file.mime_type}")
                    if hasattr(file_part.file, 'bytes') and file_part.file.bytes:
                        # Store raw bytes for direct tool call
                        image_data = file_part.file.bytes
                        user_input += f"[Image file detected: {file_part.file.mime_type}] "
                    else:
                        user_input += f"[Image file detected but no data: {file_part.file.mime_type}] "
        
        user_input = user_input.strip()
        
        # If we have image data, bypass the agent chain and call the tool directly
        if image_data:
            print("Bypassing agent chain for image analysis - calling tool directly")
            try:
                # Convert bytes to base64 if it's not already
                if isinstance(image_data, bytes):
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                else:
                    image_base64 = image_data  # Assume it's already base64
                
                # Call the image analysis tool directly
                result = analyze_image_for_cat_or_dog(image_base64)
                print(f"Direct image analysis result: {result}")
                await event_queue.enqueue_event(new_task(new_agent_text_message(result, context_id=context.context_id, task_id=context.task_id)))
                return
            except Exception as e:
                error_msg = f"Error in direct image analysis: {str(e)}"
                print(error_msg)
                await event_queue.enqueue_event(new_task(new_agent_text_message(error_msg, context_id=context.context_id, task_id=context.task_id)))
                return
        
        if not user_input:
            error_msg = "No input provided. Please provide a request for the agent to process."
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        # Process the request using the tool-calling agent (non-image requests)
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
