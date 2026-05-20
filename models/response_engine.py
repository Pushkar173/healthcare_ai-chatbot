"""
Response Generation Engine for Healthcare Chatbot
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_loader import HealthcareDataLoader
from utils.preprocessor import TextPreprocessor


class ResponseEngine:

    def __init__(self):

        self.df_qa = HealthcareDataLoader.load_qa_dataset()

        self.preprocessor = TextPreprocessor()

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            min_df=1,
            max_df=0.95
        )

        # preprocess questions
        processed_questions = self.df_qa['question'].apply(
            self.preprocessor.preprocess
        ).values

        # vectorize
        self.question_vectors = self.vectorizer.fit_transform(
            processed_questions
        )

    def get_response(
        self,
        user_input,
        top_k=3,
        confidence_threshold=0.15
    ):

        try:

            processed_input = self.preprocessor.preprocess(
                user_input.lower()
            )

            # keyword responses
            keyword_responses = {

                "dengue": """
Dengue symptoms include:

• High fever
• Severe headache
• Muscle pain
• Joint pain
• Rash
• Nausea
""",

                "cancer": """
Cancer symptoms may include:

• Weight loss
• Fatigue
• Persistent cough
• Lumps
• Skin changes
""",

                "diabetes": """
Diabetes symptoms include:

• Increased thirst
• Frequent urination
• Fatigue
• Blurred vision
""",

                "fever": """
Fever symptoms include:

• High temperature
• Chills
• Sweating
• Weakness
""",

                "cold": """
Cold symptoms include:

• Sneezing
• Runny nose
• Sore throat
• Mild fever
"""
            }

            # keyword matching
            for keyword, response in keyword_responses.items():

                if keyword in processed_input:

                    return {
                        'response': response,
                        'confidence': 0.95,
                        'intent': keyword,
                        'source_question': keyword,
                        'fallback': False
                    }

            # TF-IDF fallback
            user_vector = self.vectorizer.transform(
                [processed_input]
            )

            similarities = cosine_similarity(
                user_vector,
                self.question_vectors
            )[0]

            best_idx = np.argmax(similarities)

            best_confidence = similarities[best_idx]

            # low confidence fallback
            if best_confidence < confidence_threshold:

                return {
                    'response': """
Sorry, I could not understand your query.

Try asking about:
• dengue
• cancer
• fever
• diabetes
• cold
""",
                    'confidence': float(best_confidence),
                    'intent': None,
                    'source_question': None,
                    'fallback': True
                }

            best_row = self.df_qa.iloc[best_idx]

            return {
                'response': best_row['response'],
                'confidence': float(best_confidence),
                'intent': best_row['intent'],
                'source_question': best_row['question'],
                'fallback': False
            }

        except Exception as e:

            return {
                'response': f"Error: {str(e)}",
                'confidence': 0.0,
                'intent': None,
                'source_question': None,
                'fallback': True
            }