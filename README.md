# Multi-Purpose Tool A2A Agent

This is an A2A (Agent-to-Agent) compliant agent with tool-calling capabilities that provides math calculations, hashing operations, and sequential multi-tool workflows using OpenAI's language models through LangChain.

## Features

- **🧮 Math Calculator**: Performs elementary-level mathematical calculations and problem solving
- **🔐 Hash Generator**: Creates MD5 and SHA-512 hashes of text input
- **🎮 Web Game Automation**: Automates tic-tac-toe games on websites using optimal strategy
- **👁️ Image Understanding**: Analyzes images to detect cats and dogs using OpenAI Vision API
- **🔗 Tool-Calling Agent**: LangChain-powered agent with sequential multi-tool operations
- **🔄 Sequential Operations**: Supports chained operations like "1. md5hash 2. sha512hash 3. md5hash"
- **🤖 LangChain Integration**: Uses ChatOpenAI with proper tool calling capabilities
- **🌐 Selenium WebDriver**: Headless browser automation for web-based games
- **📡 A2A Protocol Compliance**: Follows the Agent2Agent protocol specification
- **⚡ Synchronous Operations**: Simple request-response pattern without streaming
- **🌐 CORS Enabled**: Allows cross-origin requests from any domain

## Setup

### Option 1: Docker (Recommended)

1. **Install Docker**
   Make sure Docker is installed and running on your system.

2. **Configure OpenAI API Key**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Build and Run**
   ```bash
   make build
   make run
   ```

The agent will start on `http://localhost:3000`. Press `Ctrl+C` to stop.

### Option 2: Local Python

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API Key**
   
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   
   Or set the environment variable directly:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run the Agent**
   ```bash
   python -m ape-berkeley-v2
   ```

The agent will start on `http://localhost:3000`

## Usage

The agent accepts various types of queries through the A2A protocol:

### 🧮 **Math Operations**
- `What is 25 + 37?`
- `Calculate 15 * 8`
- `Solve: 144 ÷ 12`
- `What is 2³?`
- `If I have 5 apples and give away 2, how many do I have left?`

### 🔐 **Hashing Operations**
- `Generate MD5 hash of "hello world"`
- `Create SHA-512 hash of "test string"`
- `Hash "password123" with MD5`

### 🔄 **Sequential Multi-Tool Operations**
- `For "mytext": 1. md5hash 2. sha512hash 3. md5hash 4. md5hash`
- `Calculate 10+5 then generate MD5 hash of the result`
- `Solve 15*3 and then hash the answer with SHA-512`
- `Hash "data" with MD5 then SHA-512`

### 🎮 **Web Game Automation**
- `Go to https://ttt.puppy9.com/ and play tic-tac-toe until you win, then get the secret number`
- `Play tic-tac-toe on the website https://example.com/tictactoe and extract the 14-digit secret`
- `Automate tic-tac-toe game at https://game-site.com and find the congratulation message number`

### 👁️ **Image Understanding**
- Send an image file (PNG/JPEG) with the question: `Is there a cat or dog in this image?`
- `Analyze this photo to determine if it contains a cat or a dog`
- `Look at this image and tell me if you see a cat or dog`

## Agent Card

The agent exposes four main skills:

### 🧮 **Math Calculator**
- **ID**: `math_calculator`
- **Name**: Math Calculator
- **Description**: Performs elementary-level mathematical calculations and problem solving

### 🔐 **Hash Generator**
- **ID**: `hash_generator`
- **Name**: Hash Generator
- **Description**: Generates MD5 and SHA-512 hashes of text input with support for sequential operations

### 🎮 **Web Game Automation**
- **ID**: `web_automation`
- **Name**: Web Game Automation
- **Description**: Automates web-based tic-tac-toe games using optimal strategy until winning and extracts secret numbers

### 👁️ **Image Understanding**
- **ID**: `image_understanding`
- **Name**: Image Understanding
- **Description**: Analyzes images to detect cats and dogs using OpenAI Vision API

## Architecture

- **🔗 MultiPurposeToolAgent**: LangChain tool-calling agent using `create_tool_calling_agent`
- **🛠️ LangChain Tools**: Modular tools for math calculations, MD5/SHA-512 hashing, web automation, and image analysis
- **🎮 Selenium WebDriver**: Headless Chrome automation for tic-tac-toe game playing
- **🧠 Optimal Strategy**: Implements minimax-style strategy for winning tic-tac-toe games
- **⚙️ MultiPurposeAgentExecutor**: A2A executor that implements the protocol interfaces
- **📋 Agent Card**: Declares four distinct skills and capabilities
- **👁️ OpenAI Vision API**: GPT-4o with vision capabilities for image analysis
- **⚡ Synchronous Processing**: Uses simple request-response pattern for reliability
- **🌐 CORS Middleware**: Configured to allow all origins, methods, and headers for maximum compatibility
- **🤖 ChatOpenAI Integration**: Powered by OpenAI's language models with tool calling support

## Configuration

- **Model**: Defaults to `gpt-4o-mini` (configurable via environment variable `OPENAI_MODEL`)
- **Temperature**: Set to 0 for consistent math results
- **Port**: Runs on port 3000 by default

## A2A Protocol Compliance

This agent implements the core A2A protocol methods:
- `message/send` - Send math queries and receive responses
- `tasks/get` - Retrieve task status and results  
- `tasks/cancel` - Cancel ongoing operations (not supported for math)

The agent card is available at `http://localhost:3000/.well-known/agent-card.json`

## Docker Commands

The project includes a Makefile for easy Docker management:

- `make build` - Build the Docker image
- `make run` - Run the agent in foreground (Ctrl+C to stop)
- `make stop` - Stop the running container
- `make clean` - Remove Docker image and cleanup
- `make logs` - Show container logs
- `make ps` - Show running containers
- `make help` - Show available commands

### Development Workflow

1. **Initial setup**: `make build`
2. **Development**: `make run` (code changes are live-mounted, no rebuild needed)
3. **Dependency changes**: Only rebuild when `requirements.txt` changes

The Docker setup includes:
- **Volume mounting**: Code changes are reflected immediately without rebuild
- **Foreground execution**: Easy to stop with Ctrl+C
- **Environment file support**: Automatically loads `.env` file
- **Port mapping**: Agent accessible on `localhost:3000`
