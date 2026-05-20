"""
Intent Classification Model for Healthcare Chatbot
"""

import os
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from config.settings import (
    INTENT_MODEL_PATH,
    VECTORIZER_PATH,
    MIN_DF,
    MAX_DF
)

from utils.data_loader import HealthcareDataLoader
from utils.preprocessor import TextPreprocessor


class IntentClassifier:

    def __init__(self):

        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
        self.intents = []

        self.load_or_create_model()

    def load_or_create_model(self):
        """Load existing model or train a new one"""

        try:

            if (
                os.path.exists(INTENT_MODEL_PATH)
                and os.path.exists(VECTORIZER_PATH)
            ):

                print("Existing model found. Loading...")
                self.load_model()

            else:

                print("No trained model found. Training new model...")
                self.train_model()

        except Exception as e:

            print(f"MODEL LOAD FAILED: {e}")
            print("Retraining fresh model...")

            self.train_model()

    def train_model(self):
        """Train intent classifier"""

        try:

            print("Training intent classifier...")

            # Load dataset
            df = HealthcareDataLoader.load_qa_dataset()

            # Safety check
            if df.empty:
                raise ValueError("Dataset is empty!")

            if 'question' not in df.columns or 'intent' not in df.columns:
                raise ValueError(
                    "Dataset must contain 'question' and 'intent' columns"
                )

            # Preprocess text
            X = df['question'].astype(str).apply(
                self.preprocessor.preprocess
            ).values

            y = df['intent'].astype(str).values

            # Save intent labels
            self.intents = sorted(df['intent'].unique().tolist())

            # Create vectorizer
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF,
                max_df=MAX_DF,
                lowercase=True,
                stop_words='english'
            )

            # Vectorize
            X_vec = self.vectorizer.fit_transform(X)

            # Train model
            self.model = MultinomialNB()
            self.model.fit(X_vec, y)

            # Create models directory
            os.makedirs(
                os.path.dirname(INTENT_MODEL_PATH),
                exist_ok=True
            )

            # Save model
            with open(INTENT_MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)

            # Save vectorizer
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(self.vectorizer, f)

            print("Intent classifier trained successfully!")

        except Exception as e:

            print(f"TRAINING ERROR: {e}")
            raise e

    def load_model(self):
        """Load trained model safely"""

        try:

            print(f"Loading model: {INTENT_MODEL_PATH}")

            # Load model
            with open(INTENT_MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)

            # Load vectorizer
            with open(VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)

            # Reload intents
            df = HealthcareDataLoader.load_qa_dataset()

            self.intents = sorted(
                df['intent'].unique().tolist()
            )

            print("Model loaded successfully!")

        except EOFError:

            print("Model file corrupted or empty.")
            self.train_model()

        except pickle.UnpicklingError:

            print("Invalid pickle file.")
            self.train_model()

        except Exception as e:

            print(f"LOAD MODEL ERROR: {e}")
            self.train_model()

    def predict(self, text):
        """Predict intent"""

        try:

            if not text.strip():

                return {
                    'intent': 'unknown',
                    'confidence': 0.0,
                    'all_probabilities': {}
                }

            # Preprocess
            text_processed = self.preprocessor.preprocess(text)

            # Vectorize
            X_vec = self.vectorizer.transform([text_processed])

            # Predict
            intent = self.model.predict(X_vec)[0]

            # Confidence
            probabilities = self.model.predict_proba(X_vec)[0]

            confidence = float(np.max(probabilities))

            return {
                'intent': intent,
                'confidence': confidence,
                'all_probabilities': dict(
                    zip(self.intents, probabilities)
                )
            }

        except Exception as e:

            print(f"PREDICTION ERROR: {e}")

            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'all_probabilities': {}
            }

    def predict_batch(self, texts):
        """Predict multiple texts"""

        return [self.predict(text) for text in texts]