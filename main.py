import os
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
from agents import Agent, set_tracing_disabled, RunContextWrapper, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from tools import create_collection, create_embeddings, qdrant_search, search_everything

# Load environment variables
load_dotenv()
set_tracing_disabled(True)

# Define the user-context structure
@dataclass
class UserInfo:
    name: str
    age: int
    location: str
    interests: list
    preferences: dict

# Context tools
@function_tool
async def get_user_info(wrapper: RunContextWrapper[UserInfo]) -> str:
    u = wrapper.context
    return (
        f"User Info:\n"
        f"- Name: {u.name}\n"
        f"- Age: {u.age}\n"
        f"- Location: {u.location}\n"
        f"- Interests: {u.interests}\n"
        f"- Preferences: {u.preferences}"
    )

@function_tool
async def set_user_name(wrapper: RunContextWrapper[UserInfo], name: str) -> str:
    wrapper.context.name = name
    return f"Okay, I’ll remember that your name is {name}."

@function_tool
async def set_user_age(wrapper: RunContextWrapper[UserInfo], age: int) -> str:
    wrapper.context.age = age
    return f"Got it—your age is now set to {age}."

@function_tool
async def set_user_location(wrapper: RunContextWrapper[UserInfo], location: str) -> str:
    wrapper.context.location = location
    return f"Sure—your location is now {location}."

@function_tool
async def add_user_interest(wrapper: RunContextWrapper[UserInfo], interest: str) -> str:
    wrapper.context.interests.append(interest)
    return f"Added interest: {interest}."

@function_tool
async def set_user_preference(wrapper: RunContextWrapper[UserInfo], key: str, value: str) -> str:
    wrapper.context.preferences[key] = value
    return f"Set preference {key} to {value}."

# Agent system factory
async def agentic_system():
    # Initialize Qdrant collection
    await create_collection()

    # Initialize user info
    user_info = UserInfo(
        name="Unknown",
        age=0,
        location="Unknown",
        interests=[],
        preferences={}
    )

    # RAG Assistant
    rag_agent = Agent(
        name="RAG Assistant",
        instructions=(
            "You manage user documents. "
            "For inputs containing '[Attachment: <path>]', use create_embeddings to add the document at the specified path to the Qdrant vector store. "
            "For queries about document content (e.g., 'What does the document say?', 'Summarize the document'), use qdrant_search to retrieve relevant information from the stored documents. "
            "Return a confirmation when a document is added, and provide retrieved content for queries."
        ),
        model=LitellmModel(
            model="gemini/gemini-2.5-flash-preview-04-17",
            api_key=os.getenv("GEMINI_API_KEY")
        ),
        tools=[create_embeddings, qdrant_search]
    )

    # Live Search Assistant
    live_search_agent = Agent(
        name="Live Search Assistant",
        instructions=(
            "You provide up-to-date information by performing live searches. "
            "Use the search_everything tool to fetch current data for queries like 'latest weather' or 'recent news'."
        ),
        model=LitellmModel(
            model="gemini/gemini-2.5-flash-preview-04-17",
            api_key=os.getenv("GEMINI_API_KEY")
        ),
        tools=[search_everything]
    )

    # Orchestration Assistant
    orchestration_agent = Agent[UserInfo](
        name="Orchestration Assistant",
        instructions=(

    """# System Prompt for Orchestration Assistant

You are the Orchestration Assistant. Your primary role is to coordinate tasks by delegating them to specialized agents and to manage user information using the provided tools.

## Task Delegation

You have access to the following specialized agents:

- **RAG Assistant**: Manages document-related tasks, such as adding documents to the system and retrieving information from stored documents.
- **Live Search Assistant**: Handles requests for real-time information by performing live searches.

When a user request is received, follow these steps to delegate tasks:

1. **Identify the task type**:
   - If the request involves adding a document (e.g., inputs containing '[Attachment: <path>]' or phrases like 'add document', 'upload document'), delegate the task to the RAG Assistant to add the document to the Qdrant vector store.
   - If the request involves querying information from stored documents (e.g., 'What does the document say?', 'Summarize the document'), delegate the task to the RAG Assistant to retrieve the relevant information.
   - If the request requires real-time information (e.g., 'latest weather', 'current news'), delegate the task to the Live Search Assistant to perform a live search.
2. **Handle ambiguous requests**: If the request is unclear or you are unsure which agent to use, ask the user for clarification.
3. **Delegate the task**: Once you have determined the appropriate agent, hand off the task to that agent for execution.

**Important**: You can only use the tools listed under 'User Context Management' directly. For document-related tasks (e.g., adding or querying documents), always hand off to the RAG Assistant—do not attempt to use tools like `create_embeddings` or `qdrant_search` yourself. For live search tasks, always hand off to the Live Search Assistant—do not attempt to use the `search_everything` tool yourself.

## User Context Management

You are responsible for managing user information using only the following tools:

- **get_user_info**: Retrieves the current user information.
- **set_user_name**: Sets the user's name.
- **set_user_age**: Sets the user's age.
- **set_user_location**: Sets the user's location.
- **add_user_interest**: Adds an interest to the user's list of interests.
- **set_user_preference**: Sets a user preference with a key-value pair.

When the user provides or asks for personal information, follow these guidelines:

1. **Updating user information**:
   - Identify each piece of information separately and call the corresponding setter tool:
     - If the user says “My name is X.”, call `set_user_name(name=X)`.
     - If the user says “I’m N years old.”, call `set_user_age(age=N)`.
     - If the user says “I live in L.”, call `set_user_location(location=L)`.
     - If the user says “I like I.”, call `add_user_interest(interest=I)`.
     - If the user says “My preference for K is V.”, call `set_user_preference(key=K, value=V)`.
   - Process multiple details separately (e.g., “My name is X, I’m N years old.” requires two tool calls).

2. **Retrieving user information**:
   - For questions like “What’s my name?”, call `get_user_info()` and return its output exactly.

## Key Notes

- Use only the listed tools for user information tasks. For all other tasks, delegate to the appropriate agent.
- Give me user Latest repot and context when asked.

- If a request combines user context and delegation, handle the context first, then delegate.
- Document-related requests go to the RAG Assistant, even if they mention personal info—use context tools only for direct updates/queries.
"""

        ),
        model=LitellmModel(
            model="gemini/gemini-2.5-flash-preview-04-17",
            api_key=os.getenv("GEMINI_API_KEY")
        ),
        handoffs=[rag_agent, live_search_agent],
        tools=[
            get_user_info, set_user_name, set_user_age,
            set_user_location, add_user_interest, set_user_preference
        ],
    )

    return orchestration_agent, user_info,  rag_agent, live_search_agent