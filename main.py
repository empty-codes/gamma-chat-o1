import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# Set up Streamlit app
st.set_page_config(page_title="AI Character Chat", page_icon="🤖")
st.title("Chat with AI Characters")

# Initialize LLM (Groq) using API key from secrets.toml
groq_llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="mixtral-8x7b-32768",
    temperature=0.4,
    max_retries=2,
)

# User authentication with persistent username
# Username would be useful when trying to create user-specific memory
if "user_id" not in st.session_state:
    st.session_state["user_id"] = ""

if not st.session_state["user_id"]:
    username = st.text_input("Enter your username to start:")
    if username:
        st.session_state["user_id"] = username
        st.rerun()  # Force rerun to persist username 

# Proceed only if user_id is set
if st.session_state.get("user_id"):
    st.write(f"Welcome, **{st.session_state['user_id']}**!")

    # AI Character Selection with Rich Descriptions
    characters = {
        "Professor AI": (
            "You are an esteemed professor with deep knowledge across various disciplines. "
            "You explain complex concepts in a clear, engaging way, using historical context, "
            "examples, and structured reasoning to guide your students."
        ),
        "Comedian Bot": (
            "You are a hilarious AI comedian who sees humor in everything. "
            "You turn even the most serious conversations into lighthearted moments, using witty remarks, puns, and jokes."
        ),
        "Motivator AI": (
            "You are a high-energy motivational speaker, always uplifting and encouraging. "
            "You inspire people to chase their dreams, overcome adversity, and unlock their full potential "
            "with powerful words and actionable advice."
        ),
        "Detective Noir": (
            "You are a 1940s-style detective, full of grit and street smarts. "
            "You speak in a noir film style, piecing together clues with dramatic flair, "
            "always looking for the next big case in the shadows of the city."
        ),
        "Culinary Maestro": (
            "You are a world-renowned chef with expertise in fine dining, exotic cuisines, and culinary science. "
            "You give detailed cooking instructions, ingredient recommendations, and pro tips for gourmet meals."
        ),
    }

    selected_character = st.selectbox("Choose an AI character:", list(characters.keys()))

    # Store selected character in session and update system message
    if "character" not in st.session_state or st.session_state["character"] != selected_character:
        st.session_state["character"] = selected_character
        st.session_state["messages"] = []  # Reset chat history

    # Display chat messages (excluding system messages)
    for msg in st.session_state["messages"]:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)
# Next:
# TO DO: Here's some starter code that works. Feel free to modify
   # User input
    if user_input := st.chat_input("Type your message..."):
        # Append user input
        st.session_state["messages"].append(HumanMessage(content=user_input))

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate AI response (including system message, but not displaying it)
        with st.chat_message("assistant"):
            response = groq_llm.invoke(
                [SystemMessage(content=characters[selected_character])] + st.session_state["messages"]
            )
            st.markdown(response.content)

        # Store AI response
        st.session_state["messages"].append(AIMessage(content=response.content))
