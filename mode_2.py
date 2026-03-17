import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage, SystemMessage
import asyncio
from langchain.agents import create_agent
from marshmallow import pprint

model = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
)

chat = ChatHuggingFace(llm=model, verbose=True)

#Lesson 1 : Using MCP servers
#     
#using mcp to get tools and use them in the agent. we can use any tool that is available in the mcp server. for example, we can use a puppeteer tool to scrape data from the web and use it in our agent. we can also define our own tools and add them to the mcp server.
async def main():
    client = MultiServerMCPClient(
        {
            "puppeteer": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    print(tools)

    agent = create_agent(model=chat, tools=tools)
    response = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that can use tools to answer questions."},
                {"role": "user", "content": "What is the current tool used for?"},
            ]
        }
    )

    pprint(response)


asyncio.run(main())

#====================================================================================================

#Lesson 2: Creating Context and Using It in the Agent
#creating context using dataclasses and using it in the agent. we can define a dataclass to hold the context of the conversation and use it in the agent to generate more relevant responses. we can also use the context to store information about the user and use it in the conversation.
from dataclasses import dataclass

@dataclass
class ConversationContext:
    user_name: str = "alex"
    user_location: str = "paris"

#although we created the dataclass and can pass it, it cannot be used by the model as it does not know how to use it. we need to define a tool that can access the context and use it in the agent. we can define the tool to access the context and return the information to the agent.
from langchain.tools import tool , ToolRuntime
@tool("get_user_info")
def get_user_info(runtime: ToolRuntime) -> str:
    """Returns the user information from the context."""
    return f"User name is {runtime.context.user_name} and user location is {runtime.context.user_location}."


agent = create_agent(model=chat , context_schema=ConversationContext, tools=[get_user_info])

response = agent.invoke(
    {
        "messages" : [HumanMessage(content="What is my name and location?")]
    },
    context=ConversationContext()
)

pprint(response)

print("\n\n")
#since context is immutable, we can use state management to update the context and use it in the conversation. we can define a tool to update the context and use it in the agent to store information about the user and use it in the conversation.
from langchain.agents import AgentState
from langgraph.types import Command
from langchain_core.tools import InjectedToolCallId
from langchain.messages import ToolMessage
from typing import Annotated

class CustomState(AgentState):
        user_name : str 
        favorite_color : str 

@tool("update_user_info")
def update_user_info(runtime: ToolRuntime, user_name: str, favorite_color: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Updates the user information in the context."""
    return Command(update={
            "favorite_color" : favorite_color,
            "user_name" : user_name,
            "messages" : [ToolMessage(content=f"User name updated to {user_name} and favorite color updated to {favorite_color}", tool_call_id=tool_call_id)]
    })

@tool("get_user_info")
def get_user_info(runtime: ToolRuntime) -> str:
    """Returns the user information from the context."""
    return f"User name is {runtime.state.user_name} and favorite color is {runtime.state.favorite_color}."

from langgraph.checkpoint.memory import InMemorySaver

agent_with_state_management = create_agent(model=chat, state_schema=CustomState, tools=[update_user_info], checkpointer=InMemorySaver())

response = agent_with_state_management.invoke(
    {
        "messages" : [HumanMessage(content="My name is hammad and my favorite color is blue. Can you update my information?")],
    },   
    {
         "configurable" : {"thread_id" : "1"},
    }
)

pprint(response)

response = agent_with_state_management.invoke(
    {
        "messages" : [HumanMessage(content="What is my name and favorite color?")],
    },   
    {
         "configurable" : {"thread_id" : "1"},
    }
)
pprint(response)


#Lesson 3: Multi Agent Systems
#creating multiple agents and using them to communicate with each other. we can create multiple agents with different tools and use them to communicate with each other to solve complex tasks. we can also use the state management to share information between the agents and use it in the conversation.
import requests
from tavily import TavilyClient

@tool("get_weather")
def get_weather(location: str):
    """Returns the weather information for a given location."""
    weather_info = requests.get("http://api.weatherapi.com/v1/current.json", params={"key": "your_api_key", "q": location}).json() 
    return f"The weather in {location} is {weather_info}"

@tool("search_web")
def search_web(query: str):
    """Searches the web for a given query."""
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(
        query=query,
        num_results=3,
        filters={"type": "web"},
    )
    return f"Here are the results for '{query}': {response}"

agent1 = create_agent(model=chat, tools=[get_weather])
agent2 = create_agent(model=chat, tools=[search_web])

@tool("call_agent1")
def call_agent1(location: str):
    return agent1.invoke({"messages": [HumanMessage(content=f"What is the weather in {location}?")]})

@tool("call_agent2")
def call_agent2(query: str):
    return agent2.invoke({"messages": [HumanMessage(content=f"Search the web for {query}?")]})

#main agent
main_agent = create_agent(model=chat, tools=[call_agent1, call_agent2] , system_prompt=SystemMessage(content="You are a helpful assistant that can call other agents to get information about the weather and search the web."))

question = "What is the weather in New York and search the web for the latest news on technology?"
response = main_agent.invoke(
    {
        "messages" : [HumanMessage(content=question)]
    }
)

pprint(response)