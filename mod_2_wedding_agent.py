import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
import asyncio
from langchain.agents import create_agent, AgentState
from marshmallow import pprint
from langchain.tools import tool, ToolRuntime
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient
from typing import Annotated

# ── Model Setup ──────────────────────────────────────────────────────────────

model = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
)

chat = ChatHuggingFace(llm=model, verbose=True)

# ── State Schema ─────────────────────────────────────────────────────────────
# using state management (as in mod_2 lesson 2) to maintain and update wedding preferences

class WeddingState(AgentState):
    destination: str
    origin: str
    departure_date: str
    return_date: str
    venue_min_price: int
    venue_max_price: int
    music_genre: str

# ── State Management Tools ───────────────────────────────────────────────────

@tool("update_wedding_preferences")
def update_wedding_preferences(
    runtime: ToolRuntime,
    destination: str,
    origin: str,
    departure_date: str,
    return_date: str,
    venue_min_price: int,
    venue_max_price: int,
    music_genre: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Updates the wedding planning preferences in state. Call this whenever the user provides or changes wedding details like destination, travel dates, budget, or music genre."""
    return Command(update={
        "destination": destination,
        "origin": origin,
        "departure_date": departure_date,
        "return_date": return_date,
        "venue_min_price": venue_min_price,
        "venue_max_price": venue_max_price,
        "music_genre": music_genre,
        "messages": [ToolMessage(
            content=(
                f"Wedding preferences updated — "
                f"Destination: {destination}, Origin: {origin}, "
                f"Travel: {departure_date} to {return_date}, "
                f"Venue budget: ${venue_min_price}-${venue_max_price}, "
                f"Music genre: {music_genre}"
            ),
            tool_call_id=tool_call_id,
        )],
    })


@tool("get_wedding_preferences")
def get_wedding_preferences(runtime: ToolRuntime) -> str:
    """Returns the current wedding planning preferences from state."""
    return (
        f"Destination: {runtime.state.destination}, Origin: {runtime.state.origin}, "
        f"Travel: {runtime.state.departure_date} to {runtime.state.return_date}, "
        f"Venue budget: ${runtime.state.venue_min_price}-${runtime.state.venue_max_price}, "
        f"Music genre: {runtime.state.music_genre}"
    )


# ── Tool Definitions ─────────────────────────────────────────────────────────

# --- Venue Agent Tools ---

@tool("search_venues")
def search_venues(destination: str, min_price: int, max_price: int) -> str:
    """Searches for wedding venues at a destination within a price range using Tavily web search."""
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"wedding venues in {destination} price range ${min_price} to ${max_price} reviews and availability"
    response = tavily.search(query=query, num_results=5, filters={"type": "web"})
    return f"Venue search results in {destination} (${min_price}-${max_price}): {response}"


# --- DJ Agent Tools ---

@tool("search_music")
def search_music(genre: str) -> str:
    """Searches for popular wedding music and playlists for a given genre using Tavily web search."""
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"best {genre} wedding songs playlist 2025 top tracks for wedding reception"
    response = tavily.search(query=query, num_results=5, filters={"type": "web"})
    return f"Music results for {genre} wedding playlist: {response}"


@tool("search_djs")
def search_djs(destination: str, genre: str) -> str:
    """Searches for DJs available at the wedding destination for a specific genre."""
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"{genre} wedding DJ available in {destination} reviews and pricing"
    response = tavily.search(query=query, num_results=5, filters={"type": "web"})
    return f"DJ search results for {genre} in {destination}: {response}"


# ── Sub-Agents & MCP Flight Agent ────────────────────────────────────────────

# the travel agent uses the Kiwi.com MCP server (https://mcp.kiwi.com/) for live flight search
# this follows the MCP pattern from mod_2 lesson 1, using http transport instead of stdio

async def create_travel_agent():
    """Creates the travel agent with Kiwi.com MCP flight search tools."""
    client = MultiServerMCPClient(
        {
            "kiwi_flights": {
                "url": "https://mcp.kiwi.com",
                "transport": "http",
            }
        }
    )
    flight_tools = await client.get_tools()
    print("Kiwi MCP flight tools loaded:", len(flight_tools), "tools")

    agent = create_agent(model=chat, tools=flight_tools)
    return agent, client


venue_agent = create_agent(
    model=chat,
    tools=[search_venues],
)

dj_agent = create_agent(
    model=chat,
    tools=[search_music, search_djs],
)


# ── Sub-Agent Caller Tools ───────────────────────────────────────────────────

# travel_agent is created async and stored here at startup
_travel_agent = None
_mcp_client = None


@tool("call_travel_agent")
async def call_travel_agent(origin: str, destination: str, departure_date: str, return_date: str):
    """Calls the travel agent to find flights to and from the wedding destination using the Kiwi.com MCP server."""
    response = await _travel_agent.ainvoke({
        "messages": [HumanMessage(content=(
            f"Search for flights from {origin} to {destination} departing on {departure_date}, "
            f"and also search for return flights from {destination} to {origin} on {return_date}. "
            f"Summarize the best options with prices and airlines."
        ))]
    })
    return response


@tool("call_venue_agent")
async def call_venue_agent(destination: str, min_price: int, max_price: int):
    """Calls the venue agent to search for wedding venues at the destination within a price range."""
    response = await venue_agent.ainvoke({
        "messages": [HumanMessage(content=(
            f"Search for wedding venues in {destination} within a budget of ${min_price} to ${max_price}. "
            f"Provide details on the top options including pricing, capacity, and reviews."
        ))]
    })
    return response


@tool("call_dj_agent")
async def call_dj_agent(destination: str, genre: str):
    """Calls the DJ agent to find music playlists and DJs for the wedding in a specific genre."""
    response = await dj_agent.ainvoke({
        "messages": [HumanMessage(content=(
            f"Find the best {genre} wedding songs and playlists, and also search for "
            f"available {genre} DJs in {destination}. Provide a summary of top song picks and DJ options."
        ))]
    })
    return response


# ── Main Wedding Planner Agent ───────────────────────────────────────────────

async def main():
    global _travel_agent, _mcp_client

    # initialize the travel agent with Kiwi MCP flight tools
    _travel_agent, _mcp_client = await create_travel_agent()

    # main agent with state management and checkpointer for multi-turn conversations
    main_agent = create_agent(
        model=chat,
        state_schema=WeddingState,
        tools=[
            update_wedding_preferences,
            get_wedding_preferences,
            call_travel_agent,
            call_venue_agent,
            call_dj_agent,
        ],
        checkpointer=InMemorySaver(),
        system_prompt=SystemMessage(content=(
            "You are a wedding planner assistant. You coordinate with three specialized agents: "
            "a travel agent (Kiwi.com flights), a venue agent, and a DJ agent. "
            "When a user provides wedding details, FIRST call update_wedding_preferences to save "
            "their preferences to state. Then call each sub-agent to gather information. "
            "Finally, compile a complete wedding plan with travel, venue, and music recommendations. "
            "If the user changes any preference later, update the state and re-query the relevant agent."
        )),
    )

    # ── Turn 1: Set preferences and plan ─────────────────────────────────────
    question = (
        "I am planning a wedding in Tuscany, Italy. "
        "We are flying from New York on June 15th 2026 and returning June 22nd 2026. "
        "Our venue budget is $5000 to $15000. "
        "We want a jazz genre for the music. "
        "Please put together a complete wedding plan."
    )

    response = await main_agent.ainvoke(
        {
            "messages": [HumanMessage(content=question)],
        },
        {
            "configurable": {"thread_id": "wedding-1"},
        },
    )
    pprint(response)

    # ── Turn 2: Update a preference using state ──────────────────────────────
    print("\n\n--- Updating music genre preference ---\n\n")

    response = await main_agent.ainvoke(
        {
            "messages": [HumanMessage(content="Actually, change the music genre to R&B instead of jazz. What DJ options are available now?")],
        },
        {
            "configurable": {"thread_id": "wedding-1"},
        },
    )
    pprint(response)

    # ── Turn 3: Retrieve current state ───────────────────────────────────────
    print("\n\n--- Checking current preferences ---\n\n")

    response = await main_agent.ainvoke(
        {
            "messages": [HumanMessage(content="What are my current wedding preferences?")],
        },
        {
            "configurable": {"thread_id": "wedding-1"},
        },
    )
    pprint(response)


asyncio.run(main())