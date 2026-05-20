# Healthcare AI Chatbot

An AI-powered healthcare chatbot built with Python, Machine Learning, and Streamlit that provides general health information and guidance.

## Features

- **Multi-turn Conversation**: Natural dialogue with conversation history persistence
- **Intent Classification**: Understand user queries and classify them into health categories
- **Entity Extraction**: Identify medical entities like symptoms, medications, and body parts
- **Response Retrieval**: Semantic search to find relevant Q&A pairs from knowledge base
- **Conversation Persistence**: Store and retrieve conversations using SQLite
- **Medical Disclaimer**: Safety disclaimers to emphasize professional consultation
- **User-friendly UI**: Clean Streamlit interface with chat-like messaging

## Project Structure

```
chatbot/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Project dependencies
├── config/
│   ├── __init__.py
│   └── settings.py                # Configuration constants
├── data/
│   ├── raw/                       # Raw dataset files
│   └── processed/                 # Processed training data (qa_dataset.csv)
├── models/
│   ├── __init__.py
│   ├── intent_classifier.py       # Intent classification model
│   ├── entity_extractor.py        # Named entity extraction
│   └── response_engine.py         # Query-response retrieval engine
└── utils/
    ├── __init__.py
    ├── database.py                # SQLite database operations
    ├── preprocessor.py            # Text preprocessing utilities
    └── data_loader.py             # Dataset loading and processing
```

## Installation

1. **Clone/Navigate to project directory**
   ```bash
   cd chatbot
   ```

2. **Create and activate virtual environment** (if not already done)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### First Run

On the first run, the system will:
1. Create a sample healthcare Q&A dataset (12 sample Q&A pairs)
2. Train the intent classifier model
3. Initialize the SQLite database
4. Generate entity extraction patterns

Subsequent runs will load the cached models for faster startup.

## How It Works

### 1. Intent Classification
- Text is preprocessed (cleaned, tokenized, stop words removed)
- TF-IDF vectorizer transforms text into numerical features
- Naive Bayes classifier predicts the intent (e.g., "symptom_inquiry", "medication_info")

### 2. Entity Extraction
- Regex patterns and rule-based extraction identify medical entities
- Extracts symptoms, medications, body parts, conditions, and treatments
- Entities are stored with confidence information

### 3. Response Retrieval
- User input is vectorized using TF-IDF
- Cosine similarity finds the most similar questions in the knowledge base
- Returns the corresponding response with confidence score
- Falls back to generic response if confidence is too low

### 4. Conversation Persistence
- All messages are stored in SQLite database
- Each session has a unique ID
- Messages include metadata: intent, entities, confidence, timestamp

## Intent Types

The system recognizes the following intents:
- `symptom_inquiry` - Questions about symptoms and illnesses
- `medication_info` - Questions about medications and dosages
- `lifestyle_advice` - Questions about lifestyle and wellness
- `general_health` - General health questions
- `emergency_info` - Critical/emergency situations
- `nutrition_advice` - Questions about diet and nutrition
- `exercise_advice` - Questions about physical activity

## Entity Types

- `SYMPTOM` - Health symptoms (headache, fever, etc.)
- `BODY_PART` - Body parts (head, chest, heart, etc.)
- `MEDICATION` - Medicines and drugs (aspirin, ibuprofen, etc.)
- `CONDITION` - Medical conditions (diabetes, asthma, etc.)
- `TREATMENT` - Treatments and interventions (exercise, therapy, etc.)

## Configuration

Edit `config/settings.py` to customize:
- Database path and settings
- Intent and entity types
- Model training parameters
- UI settings and disclaimers
- Model file paths

## Extending the Knowledge Base

To add more Q&A pairs:

1. **Option 1**: Edit the `create_sample_qa_dataset()` function in `utils/data_loader.py`

2. **Option 2**: Add CSV file to `data/processed/qa_dataset.csv` with columns:
   - `question` - Health question
   - `intent` - Intent type
   - `entities` - List of entities in the question
   - `response` - Response text

3. **Option 3**: Load from external sources (MedDialog, HealthQA datasets, etc.)

After adding data, retrain the model by deleting `models/intent_classifier.pkl` and `models/tfidf_vectorizer.pkl`, then restart the app.

## Improving Entity Extraction

Add custom entity patterns:

```python
from models.entity_extractor import EntityExtractor

extractor = EntityExtractor()
extractor.add_pattern('SYMPTOM', r'\b(new_symptom_name)\b')
```

## Database Operations

Access conversation history:

```python
from utils.database import ChatbotDB

db = ChatbotDB()
conversations = db.get_all_conversations()
history = db.get_conversation_history(conversation_id)
```

## Important Notes

⚠️ **Medical Disclaimer**: This chatbot provides general health information only and is **NOT** a substitute for professional medical advice. Always consult with a licensed healthcare provider for:
- Diagnosis
- Medical treatment
- Personal health conditions
- Emergency situations

## Safety Features

- Generic fallback responses for unrecognized queries
- Confidence thresholds to avoid incorrect information
- Emergency protocol for severe symptoms
- Clear disclaimers throughout the interface
- Encourages professional consultation

## Limitations

- Retrieval-based approach (cannot generate novel responses)
- Limited by size of knowledge base
- Rule-based entity extraction (not deep learning NER)
- No real-time medical data integration
- No multi-language support

## Future Enhancements

- [ ] Integration with external medical APIs (for medication info, etc.)
- [ ] Advanced NER using spaCy or transformers
- [ ] Generative responses using fine-tuned LLMs
- [ ] Multi-language support
- [ ] User feedback and response rating
- [ ] Admin dashboard for managing Q&A pairs
- [ ] Advanced analytics and conversation insights
- [ ] Integration with real medical professionals

## Dependencies

- **streamlit** - Web UI framework
- **scikit-learn** - Machine learning models and metrics
- **nltk** - Natural language processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **requests** - HTTP requests
- **python-dotenv** - Environment variable management
- **spacy** - Advanced NLP (optional)

## License

This project is for educational purposes.

## Support

For issues or questions, please refer to:
- Streamlit Documentation: https://docs.streamlit.io
- Scikit-learn Guide: https://scikit-learn.org
- NLTK Documentation: https://www.nltk.org
