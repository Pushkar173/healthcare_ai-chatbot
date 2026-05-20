"""
Text preprocessing utilities for Healthcare Chatbot
"""
import re
import string

# Common English stop words (simplified without NLTK dependency)
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from',
    'has', 'he', 'her', 'hers', 'his', 'how', 'i', 'if', 'in', 'into', 'is',
    'it', 'its', 'me', 'my', 'of', 'on', 'or', 'she', 'that', 'the', 'them',
    'then', 'these', 'they', 'this', 'those', 'to', 'us', 'was', 'we', 'what',
    'which', 'who', 'will', 'with', 'you', 'your', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'have', 'having', 'had', 'been', 'being'
}

# Medical terms to preserve
MEDICAL_PRESERVE_WORDS = {'doctor', 'hospital', 'medicine', 'health', 'patient', 'symptom'}


class TextPreprocessor:
    def __init__(self):
        # Remove medical terms from stop words to preserve them
        self.stop_words = STOP_WORDS - MEDICAL_PRESERVE_WORDS
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words (simple whitespace split)"""
        # Split by whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stop words from tokens"""
        return [token for token in tokens if token not in self.stop_words and token not in string.punctuation]
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in texts]
