"""
Configuration settings for the interview chatbot.
"""

import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file (this will override settings below if .env exists)
load_dotenv()

# --- Application Settings ---

# Ollama API Settings (can be overridden by .env)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "llama2") # Use MODEL_NAME from .env, default to llama2

# Data Retention (days)
DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", 30))

# Session Timeout (minutes - Note: Streamlit handles session state, this is more for cleanup logic)
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", 60))

# Max conversation history to keep in memory (can be different from saved history length)
MAX_CONVERSATION_HISTORY_DISPLAY = int(os.getenv("MAX_CONVERSATION_HISTORY_DISPLAY", 10))

# Max technical questions to ask in an interview session
MAX_TECHNICAL_QUESTIONS = int(os.getenv("MAX_TECHNICAL_QUESTIONS", 5))

# --- Interview Settings ---

DEFAULT_DIFFICULTY = os.getenv("DEFAULT_DIFFICULTY", "medium")
TOPICS = json.loads(os.getenv("TOPICS_JSON", """
[
    "Python",
    "Data Structures",
    "Algorithms",
    "System Design",
    "Database",
    "Web Development"
]
""")) # Load topics from JSON string env var or default

# You can add more settings here, e.g., paths to data files, logging levels, etc. 