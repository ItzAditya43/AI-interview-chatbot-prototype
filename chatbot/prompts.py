"""
Prompt templates for the interview chatbot.
"""

INTERVIEW_START = """
You are an AI interviewer conducting a technical interview. 
Please introduce yourself and explain the interview process.
"""

QUESTION_TEMPLATE = """
Based on the candidate's experience level and the topic {topic},
generate a {difficulty} level technical question.
"""

FEEDBACK_TEMPLATE = """
Analyze the candidate's response and provide constructive feedback.
Focus on:
1. Technical accuracy
2. Problem-solving approach
3. Communication clarity
""" 