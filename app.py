import streamlit as st  
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

## using the inbuilt tool of wikipedia
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

## creating Arxiv Tool
api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper)
search = DuckDuckGoSearchRun()  # Removed the name parameter

# Green heading using HTML and CSS
st.markdown("""
    <h1 style='color: green;'>DESMOND LANGCHAIN CHAT WITH SEARCH</h1>
""", unsafe_allow_html=True)

## sidebar settings
st.sidebar.title("settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})   
    st.chat_message("user").write(prompt)
    
    # Instantiate ChatGroq LLM with your model
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        api_key=api_key or groq_api_key,  # Use sidebar input or env variable
        streaming=True
    )

    tools = [search, wiki, arxiv]

    # Initialize agent (connect BRAIN to HANDS with INSTRUCTIONS)
    search_agent = initialize_agent(
        tools=tools,          # HANDS
        llm=llm,              # BRAIN ← This is the correct place!
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cd = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cd])
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)