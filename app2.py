import os
import gradio as gr
from dotenv import load_dotenv  # ✅ you missed this import
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType

# Load .env file
load_dotenv()

# Load API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ Groq API key not found. Please set GROQ_API_KEY in your .env file.")

# Wikipedia Tool
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_api)

# Arxiv Tool
arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api)

# DuckDuckGo Tool
search = DuckDuckGoSearchRun()

# Initialize LLM
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=groq_api_key,
    streaming=False
)

# Initialize agent
tools = [search, wiki, arxiv]
search_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# Chatbot function
def chat(history, message):
    history = history or []
    history.append(("user", message))

    try:
        response = search_agent.run(message)
        history.append(("assistant", response))
    except Exception as e:
        response = f"⚠️ Error: {str(e)}"
        history.append(("assistant", response))

    return history, history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='color: green;'>DESMOND LANGCHAIN CHAT WITH SEARCH</h1>")
    gr.Markdown("🤖 Ready to help! No API key required from users.")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask me something...")

    state = gr.State([])

    def user_input(message, history):
        return "", history

    msg.submit(chat, [state, msg], [chatbot, state]).then(
        user_input, [msg, state], [msg, state]
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
