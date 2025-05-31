"""
Security utilities for data privacy and protection.
"""

import hashlib
import uuid
from typing import Dict, Any

def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())

def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for storage (simple SHA-256)."""
    return hashlib.sha256(data.encode()).hexdigest()

def sanitize_user_input(input_data: str) -> str:
    """Basic input sanitization: remove leading/trailing whitespace and prevent simple script injection."""
    # This is a basic example. For production, consider more robust sanitization libraries.
    sanitized = input_data.strip()
    # Simple attempt to prevent script tags - not foolproof
    sanitized = sanitized.replace('<script>', '').replace('</script>', '')
    return sanitized 