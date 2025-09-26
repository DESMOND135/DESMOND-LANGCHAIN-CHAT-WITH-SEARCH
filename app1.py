import streamlit as st  
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Get API key from Streamlit secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ Groq API key not found. Please configure it in Streamlit secrets.")
    st.stop()

## using the inbuilt tool of wikipedia
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

## creating Arxiv Tool
api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper)
search = DuckDuckGoSearchRun()

# Green heading using HTML and CSS
st.markdown("""
    <h1 style='color: green;'>DESMOND LANGCHAIN CHAT WITH SEARCH</h1>
    <p>Ready to help! No API key required from users.</p>
""", unsafe_allow_html=True)

# Remove the sidebar API key input entirely
# st.sidebar.title("settings")  # You can keep this for other settings if needed

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
        api_key=groq_api_key,  # Use the secret key directly
        streaming=True
    )

    tools = [search, wiki, arxiv]

    # Initialize agent (connect BRAIN to HANDS with INSTRUCTIONS)
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cd = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run(st.session_state.messages, callbacks=[st_cd])
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("If this persists, please contact the app administrator.")