import uvicorn
from starlette.middleware.cors import CORSMiddleware

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_executor import (
    MultiPurposeAgentExecutor,  # type: ignore[import-untyped]
)


if __name__ == '__main__':
    # Define multiple skills for the multi-purpose agent
    math_skill = AgentSkill(
        id='math_calculator',
        name='Math Calculator',
        description='Performs elementary-level mathematical calculations and problem solving',
        tags=['math', 'calculation', 'elementary', 'arithmetic', 'problem-solving'],
        examples=[
            'What is 25 + 37?',
            'Calculate 15 * 8',
            'Solve: 144 ÷ 12',
            'What is 2³?',
            'If I have 5 apples and give away 2, how many do I have left?'
        ],
    )
    
    hashing_skill = AgentSkill(
        id='hash_generator',
        name='Hash Generator',
        description='Generates MD5 and SHA-512 hashes of text input with support for sequential operations',
        tags=['hashing', 'md5', 'sha512', 'cryptography', 'security'],
        examples=[
            'Generate MD5 hash of "hello world"',
            'Create SHA-512 hash of "test string"',
            'For "mytext": 1. md5hash 2. sha512hash 3. md5hash',
            'Hash "password123" with MD5 then SHA-512'
        ],
    )
    
    web_automation_skill = AgentSkill(
        id='web_automation',
        name='Web Game Automation',
        description='Automates web-based tic-tac-toe games using optimal strategy until winning and extracts secret numbers',
        tags=['web-automation', 'selenium', 'tic-tac-toe', 'game-playing', 'scraping'],
        examples=[
            'Go to https://ttt.puppy9.com/ and play tic-tac-toe until you win, then get the secret number',
            'Play tic-tac-toe on the website https://example.com/tictactoe and extract the 14-digit secret',
            'Automate tic-tac-toe game at https://game-site.com and find the congratulation message number'
        ],
    )
    

    # This will be the public-facing agent card
    public_agent_card = AgentCard(
        name='Multi-Purpose Tool Agent',
        description='An AI agent with tool-calling capabilities for math calculations, hashing operations, and web game automation.',
        url='http://localhost:3000/',
        version='3.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[math_skill, hashing_skill, web_automation_skill],  # All available skills
        supports_authenticated_extended_card=False,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=MultiPurposeAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    # Build the app and add CORS middleware
    app = server.build()
    
    # Add CORS middleware to allow all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
        allow_headers=["*"],  # Allow all headers
    )

    uvicorn.run(app, host='0.0.0.0', port=3000)