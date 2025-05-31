"""
Manages the conversation flow and context for the interview chatbot.
"""

import enum
import json
import logging
import requests
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from .data_handler import DataHandler # Import DataHandler
from utils.security import generate_session_id # Import session ID generator
from .question_generator import QuestionGenerator # Import QuestionGenerator
from config.settings import MAX_TECHNICAL_QUESTIONS, DEFAULT_DIFFICULTY # Import MAX_TECHNICAL_QUESTIONS and DEFAULT_DIFFICULTY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConversationState(enum.Enum):
    """Enumeration of possible conversation states."""
    GREETING = "greeting"  # Initial greeting and info gathering
    TECH_STACK = "tech_stack"  # Tech stack discussion
    TECHNICAL_QUESTIONS = "technical_questions"  # Technical Q&A
    CONCLUSION = "conclusion"  # Interview conclusion

@dataclass
class TechnicalQuestion:
    """Data class for a technical question asked during the interview."""
    question: str
    topic: str
    difficulty: str
    answer: Optional[str] = None
    feedback: Optional[str] = None
    expected_keywords: List[str] = None
    generated_follow_up_question: Optional[str] = None
    follow_up_asked: bool = False
    follow_up_answer: Optional[str] = None
    follow_up_feedback: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __post_init__(self):
        if self.expected_keywords is None:
            self.expected_keywords = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "answer": self.answer,
            "feedback": self.feedback,
            "expected_keywords": self.expected_keywords,
            "generated_follow_up_question": self.generated_follow_up_question,
            "follow_up_asked": self.follow_up_asked,
            "follow_up_answer": self.follow_up_answer,
            "follow_up_feedback": self.follow_up_feedback,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TechnicalQuestion":
        return cls(
            question=data.get("question", ""),
            topic=data.get("topic", ""),
            difficulty=data.get("difficulty", ""),
            answer=data.get("answer"),
            feedback=data.get("feedback"),
            expected_keywords=data.get("expected_keywords", []),
            generated_follow_up_question=data.get("generated_follow_up_question"),
            follow_up_asked=data.get("follow_up_asked", False),
            follow_up_answer=data.get("follow_up_answer"),
            follow_up_feedback=data.get("follow_up_feedback"),
            start_time=datetime.fromisoformat(data.get("start_time")) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data.get("end_time")) if data.get("end_time") else None,
        )

@dataclass
class UserData:
    """Data class to store user information and interview progress."""
    full_name: str = ""
    email: str = ""
    phone: str = ""
    position: str = ""
    years_experience: int = 0
    desired_positions: List[str] = None
    current_location: str = ""
    tech_stack: List[str] = None
    resume_link: Optional[str] = None  # Add resume link field
    conversation_history: List[Dict[str, str]] = None
    technical_questions: List[TechnicalQuestion] = None
    question_count: int = 0
    conversation_complete: bool = False
    session_start_time: Optional[datetime] = None
    session_end_time: Optional[datetime] = None

    def __post_init__(self):
        if self.desired_positions is None:
            self.desired_positions = []
        if self.tech_stack is None:
            self.tech_stack = []
        if self.conversation_history is None:
            self.conversation_history = []
        if self.technical_questions is None:
            self.technical_questions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert user data to dictionary format, converting dataclasses to dicts."""
        return {
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "position": self.position,
            "years_experience": self.years_experience,
            "desired_positions": self.desired_positions,
            "current_location": self.current_location,
            "tech_stack": self.tech_stack,
            "resume_link": self.resume_link,  # Include resume link in dict
            "conversation_history": self.conversation_history,
            "technical_questions": [q.to_dict() for q in self.technical_questions],
            "question_count": self.question_count,
            "conversation_complete": self.conversation_complete,
            "timestamp": datetime.now().isoformat(),
            "session_start_time": self.session_start_time.isoformat() if self.session_start_time else None,
            "session_end_time": self.session_end_time.isoformat() if self.session_end_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserData":
        """Create UserData instance from dictionary, converting dicts back to dataclasses."""
        loaded_questions = [
            TechnicalQuestion.from_dict(q_data)
            for q_data in data.get("technical_questions", [])
        ]
        return cls(
            full_name=data.get("full_name", ""),
            email=data.get("email", ""),
            phone=data.get("phone", ""),
            position=data.get("position", ""),
            years_experience=data.get("years_experience", 0),
            desired_positions=data.get("desired_positions", []),
            current_location=data.get("current_location", ""),
            tech_stack=data.get("tech_stack", []),
            resume_link=data.get("resume_link"),  # Load resume link from dict
            conversation_history=data.get("conversation_history", []),
            technical_questions=loaded_questions,
            question_count=data.get("question_count", 0),
            conversation_complete=data.get("conversation_complete", False),
            session_start_time=datetime.fromisoformat(data.get("session_start_time")) if data.get("session_start_time") else None,
            session_end_time=datetime.fromisoformat(data.get("session_end_time")) if data.get("session_end_time") else None,
        )

class ConversationManager:
    """Manages the interview conversation flow and state."""

    EXIT_KEYWORDS = ["exit", "quit", "bye", "end", "stop"]

    # Update system prompt for LLM interviewer personality
    INTERVIEWER_SYSTEM_PROMPT = """You are a professional technical interviewer conducting a job interview. Your role is to:
    1. Maintain a professional, friendly, and engaging demeanor
    2. Ask clear, specific technical questions based on the candidate's stated skills
    3. Keep responses concise and focused
    4. Provide constructive feedback after each answer
    5. Ask follow-up questions when answers need clarification
    6. Follow the exact Q&A format: I will ask, the candidate will answer, I may follow up, the candidate may improve their answer, and I will provide feedback.

    Crucially, you must *only* generate your own responses. Do *not* include any text or formatting that represents the candidate's (user's) turn in the conversation.

    Remember: You are conducting a real interview. Stay in character and maintain professional boundaries.
    Do not include descriptions of your own actions, emotions, or non-verbal cues in parentheses or any other format.
    """

    def __init__(
        self, model_name: str = "llama3.2:3b", session_id: Optional[str] = None, max_questions: int = 5
    ):
        self.data_handler = DataHandler() # Initialize DataHandler
        self.question_generator = QuestionGenerator() # Initialize QuestionGenerator
        self.question_generator.load_questions() # Load questions when manager is initialized
        self.session_id = session_id if session_id else generate_session_id() # Use provided ID or generate new
        self.conversation_history: List[Dict[str, str]] = [] # This will mirror user_data.conversation_history for display
        self.current_state = ConversationState.GREETING
        self.user_data = UserData() # Use UserData dataclass
        self.model_name = model_name
        self.max_questions = max_questions # Store max_questions as an instance variable
        self._ollama_cache: Dict[str, str] = {} # Initialize in-memory cache for Ollama responses

        # Attempt to load session data if session_id was provided
        if session_id:
            loaded_data = self.data_handler.load_session_data(self.session_id)
            if loaded_data:
                self._load_state_from_data(loaded_data) # Load state from saved data
        else:
            # If a new session, set the session start time
            self.user_data.session_start_time = datetime.now()
            logger.info(f"New session created: {self.session_id}")

    def _load_state_from_data(self, data: Dict[str, Any]):
        """Load conversation state and user data from a saved data dictionary."""
        try:
            self.user_data = UserData.from_dict(data.get("user_data", {})) # Load user_data
            self.conversation_history = self.user_data.conversation_history # Sync history for display
            # Attempt to infer state from loaded data if not explicitly saved (optional but helpful)
            if self.user_data.conversation_complete:
                self.current_state = ConversationState.CONCLUSION
            elif self.user_data.tech_stack:
                if self.user_data.question_count < self.max_questions:
                    self.current_state = ConversationState.TECHNICAL_QUESTIONS
                else:
                    # Should be conclusion if questions done
                    self.current_state = ConversationState.CONCLUSION
            elif self.user_data.full_name:
                # Assume info gathered, ready for tech stack
                self.current_state = ConversationState.TECH_STACK
            else:
                self.current_state = ConversationState.GREETING # Assume still gathering info
            logger.info(
                f"Loaded session {self.session_id}. Current state: {self.current_state.value}"
            )
        except Exception as e:
            logger.error(f"Error loading state for session {self.session_id}: {e}")
            # Reset to initial state if loading fails
            self.clear_history()

    def save_current_session(self):
        """Save the current state of the conversation and user data."""
        session_data = {
            "session_id": self.session_id,
            "current_state": self.current_state.value, # Save current state explicitly
            "user_data": self.user_data.to_dict(), # Save user data including history etc.
            "last_saved": datetime.now().isoformat(),
        }
        self.data_handler.save_session_data(self.session_id, session_data)

    def call_ollama_api(self, prompt: str, stream: bool = False) -> str:
        """
        Call Ollama API with proper error handling and timeout.

        Args:
            prompt: The input prompt for the model
            stream: Whether to stream the response

        Returns:
            str: Model's response or error message
        """
        # Check cache first
        if prompt in self._ollama_cache:
            logger.info(f"Cache hit for prompt: {prompt[:50]}...")
            return self._ollama_cache[prompt]

        api_url = 'http://localhost:11434/api/generate'
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': stream,
        }
        logger.info(f"Calling Ollama API: {api_url} with payload: {payload}")
        try:
            response = requests.post(
                api_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            response_json = response.json()
            logger.info(f"Ollama API response status: {response.status_code}")
            ollama_response = response_json['response']

            # Store response in cache before returning
            self._ollama_cache[prompt] = ollama_response

            return ollama_response
        except requests.exceptions.Timeout:
            logger.error("Ollama API request timed out")
            return "I'm taking longer than usual to respond. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Ollama API request failed: {e}. "
                f"Status Code: {e.response.status_code if e.response else 'N/A'}. "
                f"Response Text: {e.response.text if e.response else 'N/A'}"
            )
            # More specific error message for 404 related to model
            if e.response and e.response.status_code == 404:
                return (
                    f"Sorry, I can't find the model '{self.model_name}' in Ollama. "
                    "Please ensure it is pulled and available "
                    f"(`ollama pull {self.model_name}`)."
                )
            return f"Sorry, I'm having trouble processing that. Ollama API error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API call: {str(e)}")
            return "An unexpected error occurred during the API call. Please try again."

    def validate_input(self, input_value: str, field_type: str) -> Tuple[bool, str]:
        """
        Validate user input based on field type.

        Args:
            input_value: The input to validate
            field_type: The type of field being validated

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not input_value.strip():
            return False, "This field cannot be empty."

        # Use validators from utils.validators
        if field_type == "name":
            from utils.validators import validate_name
            return validate_name(input_value), "Please enter a valid name."
        elif field_type == "email":
            from utils.validators import validate_email
            return validate_email(input_value), "Please enter a valid email address."
        elif field_type == "phone":
            from utils.validators import validate_phone
            return validate_phone(input_value), "Please enter a valid phone number."
        elif field_type == "experience":
            from utils.validators import validate_years_experience
            return validate_years_experience(input_value), "Please enter a valid number of years for experience."
        else:
            return True, ""

    def should_exit_conversation(self, user_input: str) -> bool:
        """
        Check if the user input contains an exit keyword.

        Args:
            user_input: The user's input string.

        Returns:
            bool: True if an exit keyword is found, False otherwise.
        """
        return user_input.lower().strip() in self.EXIT_KEYWORDS

    def get_next_state(self, current_state: ConversationState, user_input: str) -> ConversationState:
        """
        Determine the next conversation state based on current state and user input.
        Note: This function primarily determines *potential* transitions. 
        The actual state update happens in process_user_input based on input validity and collected data.
        
        Args:
            current_state: Current conversation state
            user_input: User's input text (may be used for exit check, though that's primary in process_user_input)
            
        Returns:
            ConversationState: Next state in the conversation
        """
        # Exit keyword check is handled at the start of process_user_input
        
        if current_state == ConversationState.GREETING:
            return ConversationState.INFO_GATHERING
        
        elif current_state == ConversationState.INFO_GATHERING:
             # Transition to TECH_STACK if all required info is collected
             if all([self.user_data.full_name, self.user_data.email, self.user_data.phone, self.user_data.years_experience]):
                  return ConversationState.TECH_STACK
             else:
                  # Stay in INFO_GATHERING if information is still missing
                  return ConversationState.INFO_GATHERING
        
        elif current_state == ConversationState.TECH_STACK:
             # Transition to TECHNICAL_QUESTIONS if tech stack is provided
             if self.user_data.tech_stack:
                  return ConversationState.TECHNICAL_QUESTIONS
             else:
                  # Stay in TECH_STACK if tech stack is missing
                  return ConversationState.TECH_STACK
        
        elif current_state == ConversationState.TECHNICAL_QUESTIONS:
             # Transition to CONCLUSION if max questions are reached
             if self.user_data.question_count >= self.max_questions:
                  # Note: question_count is incremented in process_user_input before calling get_next_state
                  return ConversationState.CONCLUSION
             else:
                  # Stay in TECHNICAL_QUESTIONS if more questions to ask
                  return ConversationState.TECHNICAL_QUESTIONS
        
        elif current_state == ConversationState.CONCLUSION:
            # Stay in CONCLUSION state
            return ConversationState.CONCLUSION
            
        # Fallback
        return current_state

    def process_user_input(self, user_input: str) -> str:
        """
        Processes user input and manages the interview flow.
        """
        logger.info(f"Processing user input: '{user_input[:50]}...' in state: {self.current_state.value}")

        # Add user message to history immediately
        self.add_message("user", user_input)

        # Check for exit keywords first
        if self.should_exit_conversation(user_input):
            logger.info("Exit keyword detected. Transitioning to CONCLUSION.")
            self.current_state = ConversationState.CONCLUSION
            self.user_data.conversation_complete = True
            response = self.generate_response(self.current_state, self.user_data)
            self.save_current_session()
            self.add_message("assistant", response)
            return response

        response = ""

        # Update state transitions
        if self.current_state == ConversationState.GREETING:
            # Use LLM for information extraction
            extraction_prompt = f"""Extract *only* the following information from the user's message below and return it *strictly* as a JSON object. If a piece of information is not present, use a null value for that key. Do NOT include any other text, explanations, or formatting besides the JSON object.

Information to extract:
- Full Name
- Email Address
- Phone Number
- Position applying for
- Years of professional experience (as a number)
- Google Drive Resume Link

User's message: {user_input}

Return format Example:
{{
  "Full Name": "John Doe",
  "Email Address": "john.doe@example.com",
  "Phone Number": "123-456-7890",
  "Position Applying For": "Software Engineer",
  "Years of Professional Experience": 5,
  "Google Drive Resume Link": "https://drive.google.com/..."
}}"""

            try:
                logger.info(f"LLM Extraction Prompt:\n{extraction_prompt}")
                extraction_response = self.call_ollama_api(extraction_prompt)
                # Attempt to parse the JSON response. Be lenient with whitespace/newlines around the JSON.
                # Sometimes LLMs might add a bit of whitespace or newlines before/after the JSON.
                extraction_response = extraction_response.strip()
                logger.info(f"Raw LLM Extraction Response:\n{extraction_response}")
                parsed_data = json.loads(extraction_response)
                logger.info(f"Parsed Data from LLM:\n{parsed_data}")

                # Update user_data with extracted information (handle potential key variations from LLM)
                # Use .get() with multiple possible keys and case-insensitivity check if needed
                self.user_data.full_name = parsed_data.get("Full Name", parsed_data.get("full_name", parsed_data.get("FullName"))) or ""
                self.user_data.email = parsed_data.get("Email Address", parsed_data.get("email address", parsed_data.get("email"))) or ""
                self.user_data.phone = parsed_data.get("Phone Number", parsed_data.get("phone number", parsed_data.get("phone"))) or ""
                self.user_data.position = parsed_data.get("Position Applying For", parsed_data.get("position applying for", parsed_data.get("position"))) or ""
                # Handle years of experience which should be an integer
                years_exp = parsed_data.get("Years of Professional Experience", parsed_data.get("years of professional experience", parsed_data.get("years_experience")))
                self.user_data.years_experience = int(years_exp) if isinstance(years_exp, (int, str)) and str(years_exp).isdigit() else 0
                self.user_data.resume_link = parsed_data.get("Google Drive Resume Link", parsed_data.get("google drive resume link", parsed_data.get("resume link")))

                # Check if all *required* fields are now present
                logger.info(f"User Data after Update:\n{self.user_data}")
                required_fields = [
                    ("full name", self.user_data.full_name),
                    ("email address", self.user_data.email),
                    ("phone number", self.user_data.phone),
                    ("position you are applying for", self.user_data.position),
                    ("years of professional experience", self.user_data.years_experience)
                ]
                missing_fields = [name for name, value in required_fields if not value and value != 0]
                logger.info(f"Missing Fields: {missing_fields}")

                if not missing_fields:
                    # All required info collected, move to tech stack
                    self.current_state = ConversationState.TECH_STACK
                    logger.info(f"All required info collected. Transitioning to {self.current_state.value}.")
                    response = self.generate_response(self.current_state, self.user_data) # Generate tech stack prompt
                else:
                    # Information is still missing, prompt the user for the missing pieces
                    missing_str = ", ".join(missing_fields)
                    # Generate a more specific prompt asking only for the missing fields
                    response = f"I still need the following information: {missing_str}. Please provide these details."
                    if not self.user_data.resume_link: # Remind about resume link only if not provided yet
                        response += "\n\n(Optional: You can also include a Google Drive link to your resume.)"
                    
                    logger.info(f"Missing required info: {missing_fields}. Prompting user again.")
                    # Stay in GREETING state

            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM extraction response: {extraction_response}")
                response = "I had trouble processing your information. Please ensure you provide your full name, email, phone number, position, and years of experience. You can also include a resume link."
            except Exception as e:
                logger.error(f"An error occurred during LLM information extraction: {e}")
                response = "An error occurred while trying to extract your information. Please try again."

        elif self.current_state == ConversationState.TECH_STACK:
            if not self.user_data.tech_stack:
                # Process tech stack input
                tech_stack = [tech.strip() for tech in user_input.split(',')]
                if tech_stack:
                    self.user_data.tech_stack = tech_stack
                    self.current_state = ConversationState.TECHNICAL_QUESTIONS
                    # Assign the generated response (the first question) to the response variable
                    response = self.generate_response(self.current_state, self.user_data)
                    
                    # Fallback if generate_response didn't produce a question
                    if not response and self.current_state == ConversationState.TECHNICAL_QUESTIONS:
                        response = "Moving to technical questions... (Note: Could not generate the first question at this time.)"
                else:
                    return "Please list your technical skills, separated by commas."
            else:
                # Handle follow-up questions about additional tech stack
                additional_tech = [tech.strip() for tech in user_input.split(',')]
                if additional_tech:
                    self.user_data.tech_stack.extend(additional_tech)
                    self.current_state = ConversationState.TECHNICAL_QUESTIONS
                    # Assign the generated response (the next question or feedback) to the response variable
                    response = self.generate_response(self.current_state, self.user_data)
                else:
                    return "Please list any additional technologies you have experience with, separated by commas."

        elif self.current_state == ConversationState.TECHNICAL_QUESTIONS:
            # Handle Q&A format with follow-ups
            if not self.user_data.technical_questions:
                # This case should ideally be handled by the transition from TECH_STACK,
                # but as a fallback, ensure the first question is generated if not already.
                logger.warning("Entered TECHNICAL_QUESTIONS state with no questions. Generating first question as fallback.")
                generated_response = self.generate_response(self.current_state, self.user_data)
                if generated_response:
                     response = generated_response
            else:
                current_question = self.user_data.technical_questions[-1]
                if not current_question.answer:
                    # Process main answer
                    current_question.answer = user_input
                    # First, generate feedback for the initial answer
                    feedback = self.generate_feedback(current_question)

                    # Then, generate follow-up if needed based on the answer
                    follow_up = self.generate_follow_up(current_question)

                    if follow_up:
                        # If follow-up is generated, present feedback first, then the follow-up question
                        response = f"{feedback}\n\n{follow_up}"
                    else:
                        # If no follow-up, increment question count and proceed to next question or conclusion
                        self.user_data.question_count += 1
                        if self.user_data.question_count >= self.max_questions:
                            # Transition to conclusion if max questions reached
                            self.current_state = ConversationState.CONCLUSION
                            conclusion_response = self.generate_response(self.current_state, self.user_data)
                            response = f"{feedback}\n\n{conclusion_response}"
                        else:
                            # Generate the next question and present feedback + next question
                            next_question = self.generate_response(self.current_state, self.user_data)
                            response = f"{feedback}\n\n{next_question}"
                elif not current_question.follow_up_asked:
                    # Process follow-up answer
                    current_question.follow_up_answer = user_input
                    current_question.follow_up_asked = True
                    # Provide feedback on both initial and follow-up answers
                    feedback = self.generate_feedback(current_question)

                    self.user_data.question_count += 1 # Increment question count after the follow-up answer

                    if self.user_data.question_count >= self.max_questions:
                        # Transition to conclusion if max questions reached
                        self.current_state = ConversationState.CONCLUSION
                        conclusion_response = self.generate_response(self.current_state, self.user_data)
                        response = f"{feedback}\n\n{conclusion_response}"
                    else:
                        # Generate the next question and prepend feedback
                        next_question = self.generate_response(self.current_state, self.user_data) # This should generate the next question
                        response = f"{feedback}\n\n{next_question}"

        elif self.current_state == ConversationState.CONCLUSION:
            # Generate conclusion with summary
            if not self.user_data.session_end_time:
                self.user_data.session_end_time = datetime.now()
            return self.generate_response(self.current_state, self.user_data)

        # Add assistant response to history and save session
        if response:
            self.add_message("assistant", response)
            self.user_data.conversation_history = self.conversation_history
            self.save_current_session()

        return response

    def add_message(self, role: str, content: Any):
        """Add a message to the conversation history."""
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        self.conversation_history.append({"role": role, "content": str(content)})

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Retrieve the conversation history."""
        return self.conversation_history

    def clear_history(self):
        """
        Clear the conversation history and reset state.
        Also clears the Ollama API cache.
        """
        logger.info(f"Clearing session history for {self.session_id}.")
        self.conversation_history = []
        self.current_state = ConversationState.GREETING
        self.user_data = UserData()
        self.user_data.conversation_history = []
        self.user_data.technical_questions = []
        self.user_data.question_count = 0
        self.user_data.conversation_complete = False
        self._ollama_cache = {} # Clear the cache on history clear
        self.user_data.session_start_time = datetime.now() # Set new session start time

    def generate_conversation_summary(self, user_data: UserData) -> str:
        """
        Generates a summary of the interview conversation using Ollama.

        Args:
            user_data: The current user data and interview progress.

        Returns:
            str: The generated conversation summary.
        """
        logger.info("Starting conversation summary generation.")
        summary_prompt = f"""
Please summarize the following technical interview session based on the provided data.

Candidate Information:
- Full Name: {user_data.full_name if user_data.full_name else 'N/A'}
- Email: {user_data.email if user_data.email else 'N/A'}
- Phone: {user_data.phone if user_data.phone else 'N/A'}
- Years of Experience: {user_data.years_experience if user_data.years_experience > 0 else 'N/A'}
- Tech Stack: {", ".join(user_data.tech_stack) if user_data.tech_stack else 'N/A'}

Technical Questions and Answers:
"""

        if user_data.technical_questions:
            logger.info(f"Including {len(user_data.technical_questions)} technical questions in summary prompt.")
            for i, question in enumerate(user_data.technical_questions):
                summary_prompt += f"\nQuestion {i+1}: {question.question}\n"
                summary_prompt += f"Answer: {question.answer if question.answer else 'N/A'}\n"
                if question.feedback:
                    summary_prompt += f"Feedback: {question.feedback}\n"
                if question.generated_follow_up_question:
                     summary_prompt += f"Follow-up Question: {question.generated_follow_up_question}\n"
                     summary_prompt += f"Follow-up Answer: {question.follow_up_answer if question.follow_up_answer else 'N/A'}\n"
                     if question.follow_up_feedback:
                          summary_prompt += f"Follow-up Feedback: {question.follow_up_feedback}\n"
        else:
            logger.info("No technical questions found to include in summary.")
            summary_prompt += "No technical questions were asked or answered.\n"


        summary_prompt += f"\nOverall Conversation History (for additional context):\n"
        if user_data.conversation_history:
            logger.info("Including recent conversation history in summary prompt.")
            # Include recent conversation history for context, but not necessarily the full transcript
            # Limiting to a reasonable number of recent turns
            history_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in user_data.conversation_history[-20:] # Last 20 messages for context
            ])
            summary_prompt += history_context
        else:
            logger.info("No conversation history found.")
            summary_prompt += "No conversation history available.\n"

        # Add context about the interview not reaching the technical question stage if that was the case
        if user_data.tech_stack and not user_data.technical_questions:
            logger.info("Adding note to summary: did not reach technical questions after tech stack.")
            summary_prompt += "\nNote: The interview did not proceed to the technical question phase after the tech stack was provided.\n"
        elif user_data.conversation_complete and not user_data.technical_questions:
             logger.info("Adding note to summary: concluded before technical questions.")
             summary_prompt += "\nNote: The interview concluded before technical questions were asked.\n"


        summary_prompt += f"\nPlease provide a concise summary of the candidate's performance, highlighting strengths and areas for improvement based on their answers to the technical questions and follow-ups. If technical questions were not reached, summarize the initial interaction and collected information.\n"

        logger.info("Generating conversation summary using Ollama...")
        try:
            summary = self.call_ollama_api(summary_prompt)
            logger.info("Conversation summary generated.")
            return summary
        except Exception as e:
            logger.error(f"Error generating conversation summary via Ollama: {e}")
            return "Could not generate conversation summary."

    def generate_response(self, state: ConversationState, user_data: UserData) -> str:
        """
        Generates appropriate response based on conversation state.
        """
        logger.info(f"Generating response for state: {state.value}.")
        
        # Updated prompts for each state
        prompts = {
            ConversationState.GREETING: """
                Good morning/afternoon! Thank you for taking the time to speak with me today. My name is Rachel Lee, and I'm a Technical Interviewer here at [Company Name].

                To help us get started efficiently, please provide all of the following information in your response. Please list each item clearly:
                1. Your full name
                2. Your email address
                3. Your phone number
                4. The position you are applying for
                5. Your years of professional experience in that position
                6. A Google Drive link to your resume (optional)
            """,

            ConversationState.TECH_STACK: """
                Thank you for providing your information. Now, I'd like to understand your technical background better.

                Please list your primary programming languages, frameworks, and tools that you are proficient in (separated by commas).

                I may ask follow-up questions about other relevant technologies after you provide your initial list.
            """,

            ConversationState.TECHNICAL_QUESTIONS: """
                Based on the candidate's tech stack ({tech_stack}), I will now ask you technical questions. I will ask one question at a time, and you will provide your answer. I may ask follow-up questions if needed and provide feedback before we move to the next question.

                Here is the first question:
            """,

            ConversationState.CONCLUSION: """
                Thank you for completing the technical interview. Based on our discussion, here is a summary:
                - Position applied for: {position}
                - Years of experience: {years_experience}
                - Technical skills discussed: {tech_stack}
                - Overall performance: [Summary of strengths and areas for growth generated by the AI based on the Q&A]

                Do you have any questions about the next steps?
            """
        }

        # For states with fixed prompts
        if state in [ConversationState.GREETING, ConversationState.TECH_STACK, ConversationState.CONCLUSION]:
            logger.info(f"Using fixed prompt for state: {state.value}.")

            # Format prompt with user data
            prompt = prompts[state].format(
                full_name=user_data.full_name if user_data.full_name else "[Not Provided]",
                email=user_data.email if user_data.email else "[Not Provided]",
                phone=user_data.phone if user_data.phone else "[Not Provided]",
                years_experience=user_data.years_experience if user_data.years_experience else "[Not Provided]",
                tech_stack=", ".join(user_data.tech_stack) if user_data.tech_stack else "[Not Provided]",
                position=user_data.position if user_data.position else "[Not Provided]",
            )

            # Add system prompt and conversation history context
            full_prompt = f"""{self.INTERVIEWER_SYSTEM_PROMPT}

"""
            if user_data.conversation_history:
                context = "\n".join([
                    msg['content']  # Only include the content of the message
                    for msg in user_data.conversation_history[-3:]  # Last 3 messages for context
                ])
                full_prompt += f"""Previous conversation:
{context}

"""
            full_prompt += prompt

            # Get response from Ollama
            response = self.call_ollama_api(full_prompt)
            return response

        # Handle TECHNICAL_QUESTIONS state
        elif state == ConversationState.TECHNICAL_QUESTIONS:
            if not user_data.technical_questions and user_data.question_count == 0:
                # Generate first technical question
                if not user_data.tech_stack:
                    return "Before we proceed with technical questions, please provide your technical skills and project experience."

                # Construct prompt for generating personalized technical questions
                tech_question_prompt = f"""{self.INTERVIEWER_SYSTEM_PROMPT}

                Based on the candidate's tech stack: {', '.join(user_data.tech_stack)}

                Generate the *first* specific technical question for an interview.

                Requirements for the question:
                1. The question should be directly related to their stated skills.
                2. It should be a mix of theoretical and practical aspects if possible.
                3. Start with an intermediate difficulty level.
                4. Focus on a real-world scenario if applicable.
                5. Ensure it is a single, clear, and concise question.
                6. Do NOT include any introductory or conversational text, just the question itself.
                7. Do NOT include anything that looks like a candidate response.
                """

                # Get the first question from LLM
                logger.info(f"Prompt for first technical question:\n{tech_question_prompt}")
                first_question = self.call_ollama_api(tech_question_prompt)
                logger.info(f"Raw response from Ollama for first question: {first_question}")

                # Ensure the generated question is not empty before creating the object
                if not first_question or first_question.strip() == "":
                    logger.error("LLM returned empty response for the first technical question.")
                    # Fallback response if question generation fails
                    logger.info("Returning fallback message for first question.")
                    return "I am sorry, I encountered an issue generating the first technical question at this time. We can proceed with the interview or you can try again."

                # Create TechnicalQuestion object
                question_obj = TechnicalQuestion(
                    question=first_question.strip(), # Use strip() to remove leading/trailing whitespace
                    topic="Technical Assessment", # Default topic, could be improved later
                    difficulty="Intermediate", # Default difficulty
                    start_time=datetime.now()
                )

                self.user_data.technical_questions.append(question_obj)
                logger.info(f"First question object created: {question_obj}")
                logger.info(f"Returning first question: {first_question.strip()}")
                return first_question.strip() # Return the stripped question

            # Handle follow-up questions if any
            elif user_data.technical_questions and user_data.technical_questions[-1].generated_follow_up_question:
                return user_data.technical_questions[-1].generated_follow_up_question

            return ""  # Let process_user_input handle the next question

        # Handle INFO_GATHERING state
        elif state == ConversationState.INFO_GATHERING:
            if not user_data.full_name:
                return "Please share your full name."
            elif not user_data.email:
                return "Could you please provide your email address?"
            elif not user_data.phone:
                return "What's your contact number?"
            elif not user_data.years_experience:
                return "How many years of professional experience do you have?"
            return "I have all the necessary information. Let's proceed with your technical background."

        else:
            logger.error(f"generate_response called with unknown state: {state.value}")
            return "An error occurred while generating the response."

    def generate_follow_up(self, question: TechnicalQuestion) -> Optional[str]:
        """
        Generates a follow-up question based on the candidate's answer to a previous question.

        Args:
            question: The TechnicalQuestion object containing the original question and candidate's answer.

        Returns:
            Optional[str]: The generated follow-up question, or None if no follow-up is needed.
        """
        logger.info(f"Attempting to generate follow-up for question: {question.question[:50]}...")

        follow_up_prompt = f"""{self.INTERVIEWER_SYSTEM_PROMPT}

        The candidate was asked the following question:
        Question: {question.question}

        The candidate provided the following answer:
        Answer: {question.answer}

        Based on the original question and the candidate's answer, determine if a follow-up question is necessary to assess their understanding further. If the answer is complete and demonstrates sufficient understanding, indicate that no follow-up is needed. If the answer is incomplete, vague, or could be expanded upon, generate ONE concise follow-up question to probe deeper.

        If a follow-up is needed, provide ONLY the follow-up question text. If no follow-up is needed, respond with the single word 'NO_FOLLOW_UP'.
        """

        try:
            follow_up_response = self.call_ollama_api(follow_up_prompt)
            logger.info(f"Raw response from Ollama for follow-up: {follow_up_response}")

            if follow_up_response.strip().upper() == 'NO_FOLLOW_UP':
                logger.info("LLM indicated no follow-up needed.")
                question.generated_follow_up_question = None
                return None
            else:
                # Assuming the response is the follow-up question
                generated_question = follow_up_response.strip()
                logger.info(f"Generated follow-up question: {generated_question}")
                question.generated_follow_up_question = generated_question
                # Add the follow-up question to the conversation history immediately
                # Note: This is added to history but will be explicitly returned by process_user_input
                # self.add_message("assistant", generated_question) # Avoid double adding if process_user_input handles return
                return generated_question
        except Exception as e:
            logger.error(f"Error generating follow-up question via Ollama: {e}")
            question.generated_follow_up_question = None # Ensure it's reset on error
            return "I am sorry, I encountered an issue generating a follow-up question at this time."

    def generate_feedback(self, question: TechnicalQuestion) -> str:
        """
        Generates feedback for the candidate's answer(s) to a technical question.

        Args:
            question: The TechnicalQuestion object with answers and potential follow-up.

        Returns:
            str: The generated feedback.
        """
        logger.info(f"Attempting to generate feedback for question: {question.question[:50]}...")

        # Build follow-up information string conditionally
        follow_up_info = ""
        if question.generated_follow_up_question:
            follow_up_info += f"Follow-up Question: {question.generated_follow_up_question}\n"
            if question.follow_up_asked:
                follow_up_info += f"Candidate's Follow-up Answer: {question.follow_up_answer if question.follow_up_answer else '[No answer provided]'}\n"

        feedback_prompt = f"""{self.INTERVIEWER_SYSTEM_PROMPT}

        Provide constructive feedback on the candidate's answer(s) to the following technical interview question. Consider the correctness, clarity, completeness, and efficiency of their approach.

        Question: {question.question}
        Candidate's Answer: {question.answer if question.answer else '[No answer provided]'}

        {follow_up_info}

        Provide feedback in a concise paragraph. Highlight strengths and suggest areas for improvement if any. Do NOT include introductory phrases like 'Feedback:' or conversational filler. Just provide the feedback text.
        """

        try:
            feedback_response = self.call_ollama_api(feedback_prompt)
            logger.info(f"Raw response from Ollama for feedback: {feedback_response}")
            # Store feedback in the question object
            if question.follow_up_asked:
                 question.follow_up_feedback = feedback_response.strip()
            else:
                 question.feedback = feedback_response.strip()
            return feedback_response.strip()
        except Exception as e:
            logger.error(f"Error generating feedback via Ollama: {e}")
            # Store error message as feedback
            error_feedback = "Could not generate feedback at this time."
            if question.follow_up_asked:
                 question.follow_up_feedback = error_feedback
            else:
                 question.feedback = error_feedback
            return error_feedback

chatbot = ConversationManager(model_name="llama3.2:3b") # Ensure correct model name is used
# Example usage (remove or comment out in final app.py)
# user_input = "John Doe"
# response = chatbot.process_user_input(user_input)
# print(response) 