"""
Healthcare Chatbot UI using Streamlit
"""
import streamlit as st
import uuid
from datetime import datetime
from config.settings import (
    BOT_NAME, APP_TITLE, DISCLAIMER, MAX_HISTORY_DISPLAY
)
from utils.database import ChatbotDB
from models.intent_classifier import IntentClassifier
from models.entity_extractor import EntityExtractor
from models.response_engine import ResponseEngine


# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .bot-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #1976d2;
    }
    .user-message {
        background-color: #f3e5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: right;
        border-right: 4px solid #7b1fa2;
    }
    .disclaimer {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #f57c00;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all ML models (cached for performance)"""
    intent_classifier = IntentClassifier()
    entity_extractor = EntityExtractor()
    response_engine = ResponseEngine()
    return intent_classifier, entity_extractor, response_engine


@st.cache_resource
def get_database():
    """Get database connection (cached for session)"""
    return ChatbotDB()


def init_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'conversation_id' not in st.session_state:
        db = get_database()
        conv_id = db.create_conversation(st.session_state.session_id)
        st.session_state.conversation_id = conv_id
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None


def display_disclaimer():
    """Display medical disclaimer"""
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


def display_chat_history():
    """Display conversation history"""
    if not st.session_state.messages:
        st.info("Start a conversation by typing your health question below!")
        return
    
    for message in st.session_state.messages[-MAX_HISTORY_DISPLAY:]:
        if message['role'] == 'user':
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            bot_text = f'<div class="bot-message"><strong>{BOT_NAME}:</strong> {message["content"]}'
            if message.get('confidence'):
                bot_text += f'<br><small>Confidence: {message["confidence"]:.1%}</small>'
            bot_text += '</div>'
            st.markdown(bot_text, unsafe_allow_html=True)


def process_user_input(user_input):
    """Process user input and generate response"""
    # Load models
    intent_classifier, entity_extractor, response_engine = load_models()
    db = get_database()
    
    # Add user message to UI and database
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input
    })
    
    db.add_message(
        st.session_state.conversation_id,
        'user',
        user_input
    )
    
    # Classify intent
    intent_result = intent_classifier.predict(user_input)
    intent = intent_result['intent']
    intent_confidence = intent_result['confidence']
    
    # Extract entities
    entities = entity_extractor.extract_simplified(user_input)
    
    # Generate response
    response_result = response_engine.get_response(user_input)
    bot_response = response_result['response']
    response_confidence = response_result['confidence']
    
    # Add bot message to UI and database
    bot_message = {
        'role': 'bot',
        'content': bot_response,
        'confidence': response_confidence,
        'intent': intent
    }
    st.session_state.messages.append(bot_message)
    
    db.add_message(
        st.session_state.conversation_id,
        'bot',
        bot_response,
        intent=intent,
        entities=str(entities),
        confidence=response_confidence
    )


def sidebar_controls():
    """Sidebar controls and information"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Controls")
    
    # User name input
    user_name = st.sidebar.text_input("Enter your name (optional):")
    if user_name:
        st.session_state.user_name = user_name
    
    # Clear history button
    if st.sidebar.button("Clear Conversation", key="clear_button"):
        db = get_database()
        db.clear_conversation(st.session_state.conversation_id)
        st.session_state.messages = []
        st.success("Conversation cleared!")
        st.rerun()
    
    # Session info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Information")
    st.sidebar.text(f"Session ID: {st.session_state.session_id[:8]}...")
    st.sidebar.text(f"Messages: {len(st.session_state.messages)}")
    
    # Help section
    st.sidebar.markdown("---")
    st.sidebar.subheader("How to Use")
    st.sidebar.markdown("""
    1. **Ask Questions**: Type your health-related questions
    2. **Get Responses**: The chatbot will provide relevant information
    3. **Clear History**: Use the button above to start fresh
    4. **Important**: Always consult healthcare professionals for diagnosis
    """)


def main():
    """Main Streamlit application"""
    # Initialize session state
    init_session_state()
    
    # Header
    st.title(f"🏥 {APP_TITLE}")
    st.markdown("An AI-powered assistant for general health information and guidance")
    
    # Display disclaimer
    display_disclaimer()
    
    st.markdown("---")
    
    # Sidebar
    sidebar_controls()
    
    # Main chat interface
    col1, col2 = st.columns([1, 0.15])
    
    with col1:
        st.subheader("Chat")
    
    # Display chat history
    display_chat_history()
    
    st.markdown("---")
    
    # User input
    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        user_input = st.text_input(
            "Your Question:",
            placeholder="e.g., What are symptoms of a cold?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", key="send_button")
    
    # Process input
    if send_button and user_input:
        with st.spinner("Processing..."):
            process_user_input(user_input)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        <p>Healthcare Chatbot v1.0 | Powered by ML & Streamlit</p>
        <p>For medical emergencies, please call 911 or visit your nearest hospital</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
