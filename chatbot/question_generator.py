"""
Generates technical interview questions based on context and difficulty.
"""

import json
import random
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """Handles loading and generating technical interview questions."""

    def __init__(self, question_bank_path: str = "data/question_bank.json"):
        self.question_bank_path = question_bank_path
        self.question_bank: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.loaded = False

    def load_questions(self) -> bool:
        """Load questions from the question bank JSON file."""
        try:
            with open(self.question_bank_path, 'r') as f:
                self.question_bank = json.load(f)
            self.loaded = True
            logger.info(f"Successfully loaded question bank from {self.question_bank_path}")
            return True
        except FileNotFoundError:
            logger.error(f"Question bank file not found at {self.question_bank_path}")
            self.loaded = False
            return False
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from question bank file at {self.question_bank_path}")
            self.loaded = False
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading question bank: {e}")
            self.loaded = False
            return False

    def generate_question(self, tech_stack: List[str], difficulty: str, asked_questions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generate a technical question based on tech stack and difficulty.
        Avoids asking questions already in asked_questions.
        
        Args:
            tech_stack: List of technologies the candidate is familiar with.
            difficulty: The desired difficulty level ('easy', 'medium', 'hard').
            asked_questions: List of questions already asked in the session.
            
        Returns:
            Dict[str, Any]: A dictionary containing the question and expected keywords, or None if no question found.
        """
        if not self.loaded:
            logger.warning("Question bank not loaded.")
            return None
            
        # Filter topics to those present in the question bank and the candidate's tech stack
        available_topics = [topic.lower() for topic in tech_stack if topic.lower() in self.question_bank]

        if not available_topics:
            logger.warning("No matching topics found in question bank for the provided tech stack.")
            return None
            
        # Attempt to find a question across relevant topics
        random.shuffle(available_topics) # Randomize topic selection
        
        for topic in available_topics:
             if difficulty.lower() in self.question_bank[topic.lower()]:
                  available_questions_for_topic = self.question_bank[topic.lower()][difficulty.lower()]
                  
                  # Filter out questions that have already been asked
                  unasked_questions = [q for q in available_questions_for_topic if q not in asked_questions]
                  
                  if unasked_questions:
                       # Select a random question from the unasked ones
                       selected_question = random.choice(unasked_questions)
                       # Add the topic and difficulty to the returned question for context
                       selected_question['topic'] = topic
                       selected_question['difficulty'] = difficulty.lower()
                       logger.info(f"Generated question for topic '{topic}', difficulty '{difficulty}'.")
                       return selected_question
                       
        # If no unasked question found in any relevant topic/difficulty
        logger.warning("No new question found for the provided tech stack and difficulty.")
        return None

    def get_available_topics(self) -> List[str]:
        """Get a list of topics available in the question bank."""
        return list(self.question_bank.keys())

    def get_available_difficulties(self, topic: str) -> List[str]:
        """Get a list of difficulties available for a given topic."""
        topic_lower = topic.lower()
        if topic_lower in self.question_bank:
            return list(self.question_bank[topic_lower].keys())
        return [] 