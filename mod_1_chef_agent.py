from langchain.tools import tool
from tavily import TavilyClient
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from marshmallow import pprint
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage , AIMessage 
import os

model = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
)

chat = ChatHuggingFace(llm=model, verbose=True)


    
@tool("search_recipe")
def search_recipe(dish_name: str):
    """Searches for a recipe based on the dish name."""
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(
        query=dish_name,
        num_results=3,
        filters={"type": "recipe"},
    )
    
    return response if response else "No recipe found."


sys_prompt = SystemMessage(content="You are a helpful assistant that can search for recipes using the search_recipe tool and provide a detailed summary of each recipe with the given ingredients.")
agent = create_agent(model=chat, tools=[search_recipe] , system_prompt=sys_prompt)

response = agent.invoke(
    {
        "messages" : [HumanMessage(content="I have only chicken and rice. What can I cook with these ingredients?")]
    }
)

pprint(response)