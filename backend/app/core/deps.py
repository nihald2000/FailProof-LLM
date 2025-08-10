"""
Core dependencies for FastAPI applications.
"""

from typing import Optional, Dict, Any


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Placeholder for user authentication dependency.
    In a real implementation, this would validate JWT tokens and return user info.
    """
    return {"id": "default_user", "username": "test_user"}
