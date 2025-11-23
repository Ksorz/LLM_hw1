import streamlit as st
import requests
import json
import os

# Configuration
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="LLM Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LLM Chat Interface")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Check connection
    try:
        response = requests.get(f"{API_URL}/metadata")
        if response.status_code == 200:
            meta = response.json()
            st.success("Connected to API ✅")
            st.json(meta)
        else:
            st.error(f"API Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API ❌")
        st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            response = requests.post(
                f"{API_URL}/forward",
                json={"text": prompt},
                timeout=30
            )
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                message_placeholder.markdown(prediction)
                st.session_state.messages.append({"role": "assistant", "content": prediction})
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                message_placeholder.error(error_msg)
        except Exception as e:
            message_placeholder.error(f"Connection error: {str(e)}")
