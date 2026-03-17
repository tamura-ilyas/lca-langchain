# Intro to LangChain — Foundation Course

Hands-on code from the [LangChain Academy](https://academy.langchain.com/) Introduction to LangChain foundation course.

🎓 [View Certificate](https://academy.langchain.com/certificates/4rwbp9i6ln)

## Modules

### Module 1 — LangChain Basics

- **mod_1.py** — Core concepts: chat models, agents, tools, and checkpointing using HuggingFace + LangGraph.
- **mod_1_chef_agent.py** — A recipe-search agent that uses Tavily to suggest recipes based on available ingredients.

### Module 2 — Building Agents with LangGraph

- **mode_2.py** — Module 2 core exercises.
- **mod_2_wedding_agent.py** — A wedding planning agent built with LangGraph, demonstrating multi-step agentic workflows.

## Stack

- [LangChain](https://www.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/)
- [HuggingFace](https://huggingface.co/) — DeepSeek-R1 model
- [Tavily](https://tavily.com/) — web search tool

## Setup

```bash
pip install langchain langgraph langchain-community tavily-python
```

Set your environment variables:

```bash
export HUGGINGFACEHUB_API_TOKEN=your_token
export TAVILY_API_KEY=your_key
```

## Resources

- [LangChain Docs](https://python.langchain.com/docs/introduction/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain Academy](https://academy.langchain.com/)
