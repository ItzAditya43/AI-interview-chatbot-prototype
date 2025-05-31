"""
Main Streamlit application for the AI Interview Chatbot.
"""

import json
import time
import requests
import streamlit as st
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from datetime import datetime
from chatbot.conversation_manager import ConversationManager, ConversationState
from utils.validators import validate_email, validate_name
from config.settings import OLLAMA_API_URL, DEFAULT_MODEL, MAX_TECHNICAL_QUESTIONS
import pandas as pd

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Interview Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e6f3ff;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-online {
        background-color: #28a745;
    }
    .status-offline {
        background-color: #dc3545;
    }
    .progress-container {
        margin: 1rem 0;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
    .chat-container {
        height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def check_ollama_status() -> bool:
    """Check if Ollama service is running by hitting the tags endpoint."""
    try:
        # Use the API URL from settings
        response = requests.get(f'{OLLAMA_API_URL}/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

def initialize_session_state():
    """Initialize session state variables and set up the initial greeting.

    Forces recreation of ConversationManager and adds the initial greeting
    to history only on the very first run.
    """
    # Use a dedicated flag to ensure initial setup runs only once per session
    if '_initial_setup_complete' not in st.session_state:
        st.session_state._initial_setup_complete = False

    # Force re-creation of ConversationManager to ensure updated model_name and API URL are used
    # Use DEFAULT_MODEL from settings
    if 'conversation_manager' not in st.session_state:
        from chatbot.conversation_manager import ConversationManager, ConversationState
        from config.settings import DEFAULT_MODEL, MAX_TECHNICAL_QUESTIONS
        st.session_state.conversation_manager = ConversationManager(model_name=DEFAULT_MODEL, max_questions=MAX_TECHNICAL_QUESTIONS)

        # Initialize chat history and add the initial greeting message here
        st.session_state.chat_history = []

        # Generate and add initial greeting to history only if initial setup is not complete
        if not st.session_state._initial_setup_complete:
            initial_message = st.session_state.conversation_manager.generate_response(ConversationState.GREETING, st.session_state.conversation_manager.user_data)
            st.session_state.chat_history.append({"role": "assistant", "content": initial_message})
            # Also add to the manager's internal history for saving
            st.session_state.conversation_manager.add_message("assistant", initial_message)
            # Update the manager's user_data history and save the session after adding the initial message
            st.session_state.conversation_manager.user_data.conversation_history = st.session_state.conversation_manager.get_conversation_history()
            st.session_state.conversation_manager.save_current_session()

            # Mark initial setup as complete
            st.session_state._initial_setup_complete = True

        # Set initial state after greeting is handled
        st.session_state.current_state = ConversationState.GREETING # Move to next state after greeting
        st.session_state.conversation_active = True
        st.session_state.candidate_data = {}
        st.session_state.current_questions = []

    # Ensure other session state variables are initialized if they somehow got missed
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = True
    if 'current_state' not in st.session_state:
        from chatbot.conversation_manager import ConversationState
        st.session_state.current_state = ConversationState.GREETING
    if 'candidate_data' not in st.session_state:
        st.session_state.candidate_data = {}
    if 'current_questions' not in st.session_state:
        st.session_state.current_questions = []

def export_conversation_data() -> str:
    """Export conversation data as JSON."""
    # Export the full session data from the conversation manager
    export_data = st.session_state.conversation_manager.user_data.to_dict()
    export_data["session_id"] = st.session_state.conversation_manager.session_id
    export_data["current_state"] = st.session_state.conversation_manager.current_state.value
    # The timestamp is already in user_data.to_dict()
    
    return json.dumps(export_data, indent=2)

def display_progress():
    """Display conversation progress indicator using Streamlit components."""
    states = list(ConversationState)
    try:
        # Use current_state from the conversation manager
        current_index = states.index(st.session_state.conversation_manager.current_state)
    except ValueError:
        current_index = -1 # Handle cases where state is not in the list

    st.sidebar.markdown("### Interview Progress")

    for i, state in enumerate(states):
        state_name = state.value.replace('_', ' ').title()
        if i < current_index:
            st.sidebar.markdown(f"✅ {state_name}")
        elif i == current_index:
            st.sidebar.markdown(f"➡️ **{state_name}**")
        else:
            st.sidebar.markdown(f"⏳ {state_name}")

def display_candidate_summary():
    """Display candidate information summary."""
    # Use user_data from the conversation manager
    candidate_data = st.session_state.conversation_manager.user_data.to_dict()
    if candidate_data:
        st.markdown("### Candidate Information")
        # Exclude empty fields and history/questions from summary display
        summary_data = {k: v for k, v in candidate_data.items() if v and k not in ['conversation_history', 'technical_questions', 'question_count', 'conversation_complete', 'timestamp']}
        for key, value in summary_data.items():
             st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

def load_session(session_id: str):
    """
    Load a specific interview session.
    """
    # Clear current session state to load new data cleanly
    st.session_state.clear()

    # Reinitialize ConversationManager with the loaded session ID
    # The __init__ method will handle loading data if session_id is provided
    from chatbot.conversation_manager import ConversationManager, ConversationState # Re-import locally if needed after clear()
    from config.settings import DEFAULT_MODEL, MAX_TECHNICAL_QUESTIONS # Re-import locally

    st.session_state.conversation_manager = ConversationManager(
        model_name=DEFAULT_MODEL, # Use default model name from settings
        session_id=session_id, # Pass the session ID to load
        max_questions=MAX_TECHNICAL_QUESTIONS # Pass max_questions
    )

    # Update other session state variables based on the loaded manager state
    st.session_state.chat_history = st.session_state.conversation_manager.get_conversation_history()
    st.session_state.current_state = st.session_state.conversation_manager.current_state
    st.session_state.conversation_active = not st.session_state.conversation_manager.user_data.conversation_complete # Set active based on loaded data
    st.session_state.candidate_data = st.session_state.conversation_manager.user_data.to_dict() # Sync candidate data
    st.session_state.current_questions = st.session_state.conversation_manager.user_data.technical_questions # Sync questions

    # Set the initial setup complete flag to prevent the initial greeting from being added again
    st.session_state._initial_setup_complete = True

def main():
    """Main application function."""
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AI Interview Bot")
        
        # Ollama Status
        ollama_status = check_ollama_status()
        status_color = "status-online" if ollama_status else "status-offline"
        status_text = "Online" if ollama_status else "Offline"
        st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <span class="status-indicator {status_color}"></span>
                Ollama Status: {status_text}
            </div>
        """, unsafe_allow_html=True)
        
        if not ollama_status:
            # Use OLLAMA_API_URL and MODEL_NAME from settings in the error message
            st.error(f"Ollama service is not running or accessible at {OLLAMA_API_URL}. Please ensure Ollama is installed, a model is pulled (e.g., `ollama pull {DEFAULT_MODEL}`), and the service is running (`ollama serve`).")
            st.stop() # Stop execution if Ollama is not available
        
        # Controls
        st.markdown("### Controls")
        if st.button("Restart Conversation"):
            st.session_state.conversation_manager.clear_history()
            # Clearing the manager's history also resets its user_data, including conversation_history
            # So we need to sync the Streamlit session state chat_history with the cleared manager history
            st.session_state.chat_history = st.session_state.conversation_manager.get_conversation_history()
            # Also reset other related session state variables for clarity, although clear_history in manager resets user_data
            st.session_state.current_state = st.session_state.conversation_manager.current_state
            st.session_state.candidate_data = st.session_state.conversation_manager.user_data.to_dict()
            st.session_state.current_questions = st.session_state.conversation_manager.user_data.technical_questions
            st.session_state.conversation_active = True # Ensure conversation is active after restart
            st.rerun()
        
        # Export Data
        export_data = export_conversation_data()
        st.download_button(
            label="Export Conversation (JSON)", # Changed label to be more specific
            data=export_data,
            file_name=f"interview_{st.session_state.conversation_manager.session_id}.json", # Use session ID in filename
            mime="application/json"
        )
        
        # Data Cleanup Control (Optional)
        st.markdown("### Data Management")
        days_to_keep = st.number_input("Cleanup sessions older than (days):", min_value=1, value=30)
        if st.button("Run Cleanup"):
             st.session_state.conversation_manager.data_handler.cleanup_old_sessions(days_to_keep)
             st.sidebar.success(f"Cleaned up sessions older than {days_to_keep} days.") # Provide user feedback

        # Session Management (Load Previous)
        st.markdown("### Load Session")
        # Get list of available sessions
        available_sessions = st.session_state.conversation_manager.data_handler.list_all_sessions()

        if available_sessions:
            selected_session_id = st.selectbox("Select a session to load:", available_sessions)
            if st.button("Load Session"):
                # Call a function to handle loading the session
                load_session(selected_session_id)
                st.sidebar.success(f"Session {selected_session_id} loaded.")
                st.rerun() # Rerun to update the chat display with loaded history
        else:
            st.sidebar.info("No saved sessions found.")

        # Display progress
        display_progress()
        
        # Display candidate summary
        display_candidate_summary()

    # Main chat interface
    st.title("AI Technical Interview")
    
    # Display the welcome message only at the beginning of a new conversation
    # Removed the previous conditional display as the initial message is now added in initialize_session_state
    # if st.session_state.current_state == ConversationState.GREETING and not st.session_state.initial_greeting_displayed:
    #      initial_message = st.session_state.conversation_manager.generate_response(st.session_state.current_state, st.session_state.conversation_manager.user_data)
    #      with st.chat_message("assistant"):
    #           st.write(initial_message)
    #      # Add the initial message to history so it's not repeated
    #      st.session_state.chat_history.append({"role": "assistant", "content": initial_message})
    #      # Also add to the manager's internal history for saving
    #      st.session_state.conversation_manager.add_message("assistant", initial_message)
    #      st.session_state.conversation_manager.user_data.conversation_history = st.session_state.conversation_manager.get_conversation_history() # Sync history in user_data
    #      st.session_state.conversation_manager.save_current_session() # Save the initial state and message
    #      st.session_state.initial_greeting_displayed = True # Set the flag after displaying

    # Chat container
    chat_container = st.container()
    with chat_container:
        # Display chat history from session state (which is synced with manager's history)
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # User input
    if st.session_state.conversation_active:
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to chat immediately for responsiveness
            with chat_container: # Use the chat_container to append messages
                with st.chat_message("user"):
                    st.write(user_input)

            # Process user input and get response from ConversationManager
            response = None
            with st.spinner("AI is thinking..."):
                 # Pass the user input to the conversation manager
                 # process_user_input now handles state updates, data updates, and internal response generation/history addition
                 response = st.session_state.conversation_manager.process_user_input(user_input)

            # Update session state variables that are used for display and logic in app.py
            # These are updated inside process_user_input and we sync them here for app.py's use
            st.session_state.current_state = st.session_state.conversation_manager.current_state
            st.session_state.candidate_data = st.session_state.conversation_manager.user_data.to_dict()
            st.session_state.current_questions = st.session_state.conversation_manager.user_data.technical_questions # Sync questions
            st.session_state.conversation_active = not st.session_state.conversation_manager.user_data.conversation_complete # Sync active status

            # The assistant response was already added to the manager's internal history by process_user_input.
            # We need to sync the app's display history with the manager's history.
            st.session_state.chat_history = st.session_state.conversation_manager.get_conversation_history()

            # Rerun to update the UI (chat history, sidebar, etc.)
            st.rerun()
    else:
        # If conversation is not active (i.e., concluded), display a message below the chat input
        st.info("The interview has concluded. You can restart the conversation or export the data.")

    # Display analytics section
    # TODO: Make analytics more robust to partial data or sessions that didn't reach technical questions
    # st.subheader("Analytics Dashboard")
    # tabs = st.tabs(["Session Overview", "Tech Stack Analysis", "Question Performance", "Export Data"])
    # with tabs[0]:
    #     display_analytics_section(st.session_state.user_data)
    # with tabs[1]:
    #     st.write("Tech Stack Analysis (Coming Soon)") # Placeholder
    #     # TODO: Implement tech stack analysis
    # with tabs[2]:
    #     st.write("Question Performance (Coming Soon)") # Placeholder
    #     # TODO: Implement question performance analysis
    # with tabs[3]:
    #     st.write("Export Data (Coming Soon)") # Placeholder
    #     # TODO: Implement data export functionality

if __name__ == "__main__":
    main() 