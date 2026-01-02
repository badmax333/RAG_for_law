"""
Configuration file for API keys.
Create a .env file or set environment variables for production use.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Mistral AI API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

if not MISTRAL_API_KEY:
    # Fallback: try to read from config file directly (for development)
    # You can set it here temporarily, but it's better to use .env file
    MISTRAL_API_KEY = "your_mistral_api_key_here"
