import chainlit as cl
from main import agentic_system, UserInfo
from agents import Runner
from agents.exceptions import ModelBehaviorError  # Import ModelBehaviorError
import os
import logging
import time
import litellm.exceptions  # Import for InternalServerError

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@cl.on_chat_start
async def start():
    start_time = time.time()
    await cl.Message(content="Hello! I'm your AI assistant. Upload documents or ask for real-time info!").send()
    cl.user_session.set("chat_history", [])
    user_info = UserInfo(name="Unknown", age=0, location="Unknown", interests=[], preferences={})
    cl.user_session.set("user_info", user_info)
    logging.info(f"Chat started in {time.time() - start_time:.2f}s")

@cl.on_message
async def main(message: cl.Message):
    start_time = time.time()
    history = cl.user_session.get("chat_history", [])
    user_info = cl.user_session.get("user_info")
    logging.debug(f"Received message: {message.content}")

    # Handle file uploads with absolute paths in C:\temp_uploads
    full_content = message.content
    if message.elements:
        temp_dir = os.path.abspath(r"C:\temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        for element in message.elements:
            if isinstance(element, cl.File):
                dest_path = os.path.join(temp_dir, element.name)
                file_start = time.time()
                try:
                    if element.content is not None:
                        with open(dest_path, "wb") as dst:
                            dst.write(element.content)
                    else:
                        with open(element.path, "rb") as src, open(dest_path, "wb") as dst:
                            dst.write(src.read())
                    logging.info(f"File saved to {dest_path} in {time.time() - file_start:.2f}s")
                    full_content += f" [Attachment: '{dest_path}']"
                except Exception as e:
                    logging.error(f"Failed to save file {element.name}: {str(e)}")
                    await cl.Message(content=f"Error saving file {element.name}: {str(e)}").send()
                    return

    history.append({"role": "user", "content": full_content})
    logging.debug(f"Full input to agent: {full_content}")

    # Show waiting message
    msg = await cl.Message(content="Processing your request...").send()

    try:
        # --- HERE’S THE FIX: unpack all four returns ---
        result = await agentic_system()  # returns (orchestration_agent, user_info, rag_agent, live_search_agent)
        logging.info(f"agentic_system() returned: {result}")
        if not (isinstance(result, tuple) and len(result) == 4):
            raise ValueError(f"Expected exactly 4 values from agentic_system(), got {result!r}")
        orchestration_agent, user_info, rag_agent, live_search_agent = result
        agent = orchestration_agent
        # -------------------------------------------------

        logging.info(f"Agent initialized in {time.time() - start_time:.2f}s")
        response = Runner.run_streamed(
            starting_agent=agent,
            input=history,
            context=user_info
        )
        full_response = ""
        async for event in response.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                full_response += event.data.delta
        msg.content = full_response
        await msg.update()
        logging.info(f"Response generated in {time.time() - start_time:.2f}s")

    except litellm.exceptions.InternalServerError as e:
        logging.error(f"Internal Server Error from Gemini API: {str(e)}")
        msg.content = "Sorry, there was an issue connecting to the AI service. Please try again later."
        await msg.update()

    except ModelBehaviorError as e:
        logging.error(f"ModelBehaviorError: {str(e)}")
        msg.content = "Sorry, I couldn’t process that request. Could you try rephrasing it?"
        await msg.update()

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        msg.content = "An error occurred. Please try again later."
        await msg.update()

    # Save history and finish
    history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("chat_history", history)
    logging.info(f"Total processing time: {time.time() - start_time:.2f}s")

