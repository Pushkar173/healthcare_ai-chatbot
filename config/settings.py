"""
Configuration settings for Healthcare Chatbot
"""

# Database settings
DB_PATH = "data/chatbot.db"
DB_INIT = True

# Intent types
INTENTS = [
    "symptom_inquiry",
    "medication_info",
    "lifestyle_advice",
    "general_health",
    "emergency_info",
    "nutrition_advice",
    "exercise_advice"
]

# Medical entity types
ENTITY_TYPES = [
    "SYMPTOM",
    "BODY_PART",
    "MEDICATION",
    "CONDITION",
    "TREATMENT"
]

# Model paths
INTENT_MODEL_PATH = "models/intent_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
ENTITY_PATTERNS_PATH = "models/entity_patterns.pkl"

# Data paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
QA_DATASET_PATH = "data/processed/qa_dataset.csv"

# Model training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_DF = 2
MAX_DF = 0.8

# UI settings
BOT_NAME = "HealthCare Assistant"
APP_TITLE = "Healthcare AI Chatbot"
MAX_HISTORY_DISPLAY = 20

# Safety & disclaimers
DISCLAIMER = "⚠️ **Disclaimer**: This chatbot provides general health information only and is NOT a substitute for professional medical advice. Always consult a healthcare provider for diagnosis and treatment."
