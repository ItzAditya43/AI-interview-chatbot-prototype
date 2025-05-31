"""
Input validation utilities for the interview chatbot.
"""

import re

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_name(name: str) -> bool:
    """Validate candidate name (at least 2 non-whitespace characters)."""
    return bool(name and len(name.strip()) >= 2)

def validate_difficulty(difficulty: str) -> bool:
    """Validate question difficulty level."""
    return difficulty.lower() in ['easy', 'medium', 'hard']

def validate_phone(phone: str) -> bool:
    """Validate phone number format (basic check for digits and optional +1)."""
    # Allows optional +1, spaces, hyphens, and parentheses, followed by 7-15 digits
    pattern = r'^\+?1?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4,9}$'
    # A simpler pattern for now as requested: optional +1 followed by 9-15 digits
    pattern = r'^\+?1?\d{9,15}$'
    return bool(re.match(pattern, phone))

def validate_years_experience(years: str) -> bool:
    """Validate years of experience (numeric between 0 and 50, accepting natural language like '1 year')."""
    try:
        # Attempt to extract a number from the string first
        number_match = re.search(r'\d+(\.\d+)?', years)
        if number_match:
            years_float = float(number_match.group(0))
            return 0 <= years_float <= 50
        # If no number is found, try converting the whole string if it's purely numeric
        else:
            years_float = float(years)
            return 0 <= years_float <= 50
    except ValueError:
        return False 