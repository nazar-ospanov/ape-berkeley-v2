#!/usr/bin/env python3
"""
Simple test script for the Math Tool A2A Agent
"""

import asyncio
import aiohttp
import json
import sys
from typing import Dict, Any


async def test_agent(math_query: str, base_url: str = "http://localhost:3000") -> None:
    """Test the math agent with a given query."""
    
    # First, get the agent card to verify the agent is running
    print(f"🔍 Testing agent at {base_url}")
    print(f"📝 Math query: '{math_query}'")
    print("-" * 50)
    
    async with aiohttp.ClientSession() as session:
        try:
            # Get agent card
            async with session.get(f"{base_url}/.well-known/agent-card.json") as response:
                if response.status != 200:
                    print(f"❌ Failed to get agent card: HTTP {response.status}")
                    return
                
                agent_card = await response.json()
                print(f"✅ Agent Card Retrieved: {agent_card['name']}")
                print(f"📋 Available Skills: {[skill['name'] for skill in agent_card['skills']]}")
                print()
                
            # Send math query
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "parts": [
                            {
                                "type": "text",
                                "text": math_query
                            }
                        ]
                    }
                },
                "id": "test-request-1"
            }
            
            print("📤 Sending math query...")
            async with session.post(
                f"{base_url}/",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    print(f"❌ Failed to send message: HTTP {response.status}")
                    error_text = await response.text()
                    print(f"Error details: {error_text}")
                    return
                
                result = await response.json()
                print(f"✅ Message sent successfully")
                print(f"🆔 Task ID: {result['result']['id']}")
                
                task_id = result['result']['id']
                
            # Poll for task completion
            print("⏳ Waiting for response...")
            max_attempts = 30
            for attempt in range(max_attempts):
                await asyncio.sleep(1)
                
                task_payload = {
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {
                        "id": task_id
                    },
                    "id": "test-get-task"
                }
                
                async with session.post(
                    f"{base_url}/",
                    json=task_payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        print(f"❌ Failed to get task: HTTP {response.status}")
                        return
                    
                    task_result = await response.json()
                    task = task_result['result']
                    
                    if task['status']['state'] == 'completed':
                        print("✅ Task completed!")
                        
                        # Extract the math result from artifacts
                        if task.get('artifacts'):
                            for artifact in task['artifacts']:
                                if artifact.get('parts'):
                                    for part in artifact['parts']:
                                        if part.get('type') == 'text' and part.get('text'):
                                            print(f"🧮 Math Result:")
                                            print(f"{part['text']}")
                                            return
                        
                        print("⚠️  No result found in task artifacts")
                        return
                    
                    elif task['status']['state'] == 'failed':
                        print(f"❌ Task failed: {task['status'].get('error', 'Unknown error')}")
                        return
                    
                    elif attempt == max_attempts - 1:
                        print("⏰ Timeout waiting for task completion")
                        return
                    
                    print(f"⏳ Task status: {task['status']['state']} (attempt {attempt + 1}/{max_attempts})")
            
        except aiohttp.ClientError as e:
            print(f"❌ Connection error: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


async def main():
    """Main test function."""
    print("🧮 Math Tool A2A Agent Test")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        math_query = " ".join(sys.argv[1:])
    else:
        math_query = "What is 15 + 27 * 3?"
    
    await test_agent(math_query)


if __name__ == "__main__":
    asyncio.run(main())
