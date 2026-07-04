import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

# LLM
llm = ChatOllama(model="qwen2.5:3b")


@tool
def get_datetime():
    """Get today's date."""
    return datetime.now().strftime("%d-%m-%Y")


# Search Tool
search_tool = TavilySearch()

# System Prompt
system_prompt = """
You are a helpful AI assistant.

Rules:
1. Answer all user questions clearly.
2. Use the get_datetime tool only when the user explicitly asks for today's date.
3. Use the Tavily search tool when the question requires up-to-date information from the internet.
"""

# Create Agent
agent = create_agent(
    model=llm,
    tools=[get_datetime, search_tool],
    system_prompt=system_prompt,
)

# Chat Loop
while True:
    user_query = input("\nEnter your query (type 'exit' to quit): ")

    if user_query.lower() == "exit":
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query,
                }
            ]
        }
    )

    print("\nAssistant:")
    print(response["messages"][-1].content)
