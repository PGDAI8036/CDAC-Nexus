# Importing necessary libraries
import pickle
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
import base64

def response_generator(stream):
    """
    Handles the streaming response from the chat engine and yields chunks of data,
    allowing the assistant's respose to be displayed progressively.
    """
    try:
        for chunk in stream.response_gen:
            yield chunk
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")

# Function to load the pre-built index using pickle
@st.cache_resource(show_spinner=False)
def load_index():
    try:
        with open("index.pkl", "rb") as f:
            index = pickle.load(f)
        return index
    except Exception as e:
        st.error(f"An error occurred while loading the index: {e}")

# Function to load an image and encode it in base64 for displaying on the Streamlit interface
def load_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"An error occurred while loading the image: {e}")
        return None

def main() -> None:
    
    # Configuring Streamlit page settings
    st.set_page_config(page_title="CDAC Nexus", page_icon="cdac_logo.png", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("CDAC Nexus: Your Digital Assistant 💬")

    # Load and display the bot's image on the interface
    img = load_image("assets/Bot.png")
    image_html = f"""
        <style>
            .image-container {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 150px;
                z-index: 100;
            }}
        </style>
        <div class="image-container">
            <img src="data:image/png;base64,{img}" width="150" />
        </div>
    """
    st.markdown(image_html, unsafe_allow_html=True)

    # Initialize the chat session and load the index if not already loaded
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "index" not in st.session_state:
        st.session_state.index = load_index()
        st.session_state.activate_chat = True

        # Add the initial assistant message
        st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm CDAC Nexus. How can I help you?"}]
    
    # If the chat is activated, handle the user interaction
    if st.session_state.activate_chat:
        
        # Initialize message history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize the chat engine
        if "chat_engine" not in st.session_state:
            try:
                model_name = "llama3.1:8b-instruct-q4_K_M"
                llm = Ollama(model=model_name, request_timeout=300.0)

                # Define the system prompt to guide the chatbot's responses
                system_prompt = (
                    """
                    Use the following pieces of context to answer the question in one to 2 sentences.
                    If you don't know the answer, just say that you don't know in a polite manner, don't try to make up an answer.
                    Use three sentences maximum and keep the answer as concise as possible.
                    When user inclucdes words like in-depth/details in his/her prompt, answer in 3 to 4 sentences related to main keyword.
                    """
                )
                
                memory = ChatMemoryBuffer(token_limit=2000)

                # Initialize the chat engine with streaming capability
                st.session_state.chat_engine = st.session_state.index.as_chat_engine(llm=llm, chat_mode="context", memory=memory, streaming=True, system_prompt=system_prompt)
            except Exception as e:
                st.error(f"An error occurred while initializing the chat engine: {e}")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="💻" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        # Capture and process user input
        if prompt := st.chat_input("How can I help you?"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display the user's message in the chat
            with st.chat_message("user", avatar="💻"):
                st.markdown(prompt)

            # Generate and display the assistant's response
            try:
                if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        stream = st.session_state.chat_engine.stream_chat(prompt)
                        response = st.write_stream(response_generator(stream))
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")

    else:
        # Inform user that you are good to go
        st.markdown("<span style='font-size:15px;'><b>The assistant is ready! Feel free to ask!</b></span>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
