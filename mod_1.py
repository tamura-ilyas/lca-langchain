from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
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

#lesson 1 : simple chat with model
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]

#response = chat.invoke(messages)
#print(response)

#====================================================================================================

#Lesson 2: Creating an Agent

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage , AIMessage

agent = create_agent(model=chat)

response = agent.invoke(
    {
        "messages" : [HumanMessage(content="What is the capital of france?")]
    }
)

print(response)

#==================================================================================================== 

#Lesson 3: Agent with System Prompts
system_prompt = SystemMessage(content="You are a helpful assistant that answers questions about geography.")
response = agent.invoke(
        {
        "messages" : [system_prompt, HumanMessage(content="What is the capital of france?")]
    }
)
print(response)

#====================================================================================================

#Lesson 4: Agent with Tools
from langchain.tools import tool

#can use tools to extend the capabilities of the agent. For example, we can define a tool to calculate the square of a number and use it in our agent. we can define the description and name within the decorator or in the function itself.
@tool("get_square")
def get_square1(num: int) -> int:
    """Returns the square of a number."""
    return num * num


agent_with_tools = create_agent(model=chat, tools=[get_square1])

response = agent_with_tools.invoke(
    {
        "messages": [
            HumanMessage(content="What is the square of 4?")
        ]
    }
)
print(response)

#====================================================================================================

#Lesson 5: Checkpointing and State Management
from langgraph.checkpoint.memory import InMemorySaver
agent_with_checkpoint = create_agent(model=chat, tools=[get_square1], checkpointer=InMemorySaver())

question = "hello, im hammad and my favorite number is 5, can you tell me what is the square of my favorite number?"  
config = {"configurable" : {"thread_id": "1"}}  # Example configuration for checkpointing

response = agent_with_checkpoint.invoke(
    {"messages": [question]},
    config,
)

pprint(response)

response = agent_with_checkpoint.invoke(
    {"messages": ["what was my previous question?"]},
    config,
)

pprint(response)