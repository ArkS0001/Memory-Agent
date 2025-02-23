# Memory-Agent

# LangMem

LangMem helps agents learn and adapt from their interactions over time.

It provides tooling to extract important information from conversations, optimize agent behavior through prompt refinement, and maintain long-term memory.

It offers both functional primitives you can use with any storage system and native integration with LangGraph's storage layer.

This lets your agents continuously improve, personalize their responses, and maintain consistent behavior across sessions.
Key features¶

    🧩 Core memory API that works with any storage system
    🧠 Memory management tools that agents can use to record and search information during active conversations "in the hot path"
    ⚙️ Background memory manager that automatically extracts, consolidates, and updates agent knowledge
    ⚡ Native integration with LangGraph's Long-term Memory Store, available by default in all LangGraph Platform deployments

Installation¶
    
    pip install -U langmem

Configure your environment with an API key for your favorite LLM provider:
    
    export ANTHROPIC_API_KEY="sk-..."  # Or another supported LLM provider

Creating an Agent¶

Here's how to create an agent that actively manages its own long-term memory in just a few lines:

# Import core components 

    
    from langgraph.prebuilt import create_react_agent
    from langgraph.store.memory import InMemoryStore
    from langmem import create_manage_memory_tool, create_search_memory_tool

# Set up storage 

    
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    ) 

# Create an agent with memory capabilities 
    
    
    agent = create_react_agent(
        "anthropic:claude-3-5-sonnet-latest",
        tools=[
            # Memory tools use LangGraph's BaseStore for persistence (4)
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",)),
        ],
        store=store,
    )

Then use the agent:

# Store a new memory 

  
      agent.invoke(
          {"messages": [{"role": "user", "content": "Remember that I prefer dark mode."}]}
      )

# Retrieve the stored memory 


    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What are my lighting preferences?"}]}
    )
    print(response["messages"][-1].content)
    # Output: "You've told me that you prefer dark mode."

The agent can now store important information from conversations, search its memory when relevant, and persist knowledge across conversations.
