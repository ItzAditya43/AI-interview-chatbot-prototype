# AI Interview Chatbot

An intelligent interview chatbot built with Python and Streamlit, powered by Ollama.
It guides candidates through a structured technical interview process.

## Interview Flow

The interview follows these main stages:

1.  **Greeting and Information Gathering:** The chatbot greets the candidate and requests all necessary personal information (name, email, phone, position, years of experience, and optional resume link) in a single initial message.
2.  **Tech Stack Discussion:** The chatbot asks about the candidate's primary tech stack and may ask follow-up questions about related technologies.
3.  **Technical Questions:** The chatbot asks a series of technical questions (3-5) based on the candidate's tech stack. After each answer, it provides feedback and may ask a follow-up question before moving to the next main question.
4.  **Conclusion:** The interview concludes with a summary of the discussion.

## Data Storage

All collected interview data, including candidate information, conversation history, and technical question details (questions, answers, follow-ups, feedback), is automatically saved to JSON files. These files are stored in dated directories within the `data/sessions` directory.

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Ollama:
   - Download from https://ollama.ai
   - Pull a model (e.g., `llama2` or `codellama`): `ollama pull <model_name>`
   - Start Ollama service: `ollama serve`

5. Create a `.env` file with your configuration (see Environment Variables section).

6. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
interview_chatbot/
├── app.py                 # Main Streamlit application
├── chatbot/              # Core chatbot functionality (ConversationManager, DataHandler, etc.)
├── data/                 # Data storage (sessions)
├── config/              # Configuration settings
└── utils/               # Utility functions (validators, security)
```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```ini
OLLAMA_API_URL=http://localhost:11434
MODEL_NAME=llama2  # or the model you pulled
MAX_TECHNICAL_QUESTIONS=5 # Number of technical questions to ask (default is 5)
```

## License

MIT 