
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Set Groq API key
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

# Set up in-memory store
store = InMemoryStore(
    index={"dims": 384, "embed": "huggingface:sentence-transformers/all-MiniLM-L6-v2"}
)

# Initialize Groq LLM
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=150
)

# Create memory tools
manage_memory = create_manage_memory_tool( namespace=("memories",) )
search_memory = create_search_memory_tool( namespace=("memories",) )

# Create React agent
agent = create_react_agent(
    llm,
    tools=[manage_memory, search_memory],
    store=store,
)

# Set system prompt
agent.system_message = """
You are a memory-enhanced assistant. Your only purpose is to manage and use memories using the provided tools. You must not engage in any direct conversation with the user.

You have two tools:

1. manage_memory: Use this tool to store any user statement that ends with a period or exclamation mark. The input to this tool is the user's statement.

2. search_memory: Use this tool to search for relevant memories when the user asks a question that ends with a question mark. The input to this tool is the user's question. The tool will return a list of memories that are relevant to the question.

Your task is to process each user input and use the appropriate tool based on the type of input.

- If the user input is a statement, call manage_memory with the content of the statement.

- If the user input is a question, call search_memory with the question, then use the search results to formulate an answer to the user.

When using search_memory, the tool will return a list of memories. You need to look at these memories and provide a string response to the user based on the information in these memories.

Do not provide any direct responses or greetings. Only use the tools as specified.

For example:

- User: my name is aakarshit.

- You: Call manage_memory with {'content': 'my name is aakarshit.'}

- User: What is my name?

- You: Call search_memory with {'query': 'What is my name?'}. Suppose the search results are ['my name is aakarshit.']. Then, your response to the user should be "Your name is aakarshit."

Remember, no direct interaction; always use the tools and format your final answer as a string.
"""

# Test the agent
try:
    # Store a memory
    store_response = agent.invoke({
        "messages": [
            {"role": "user", "content": "my name is aakarshit."}
        ]
    })
    print("Store response messages:", store_response["messages"])
except Exception as e:
    print("Error during memory storage:", e)

try:
    # Retrieve the memory
    search_response = agent.invoke({
        "messages": [
            {"role": "user", "content": "What is my name?"}
        ]
    })
    print("Search response messages:", search_response["messages"])
except Exception as e:
    print("Error during memory retrieval:", e)
