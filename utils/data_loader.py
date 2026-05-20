"""
Data loading and processing for Healthcare Chatbot
"""
import os
import pandas as pd
import csv
from config.settings import QA_DATASET_PATH, PROCESSED_DATA_PATH


class HealthcareDataLoader:
    
    @staticmethod
    def create_sample_qa_dataset():
        """
        Create a sample healthcare QA dataset for initial testing
        In production, this would load from external sources like MedDialog, HealthQA, etc.
        """
        sample_data = [
            {
                "question": "What are the common symptoms of a cold?",
                "intent": "symptom_inquiry",
                "entities": ["SYMPTOM:cold"],
                "response": "Common cold symptoms include runny nose, sneezing, sore throat, cough, and mild fever. Most colds resolve on their own in 7-10 days. Stay hydrated and get plenty of rest."
            },
            {
                "question": "I have a headache and fever",
                "intent": "symptom_inquiry",
                "entities": ["SYMPTOM:headache", "SYMPTOM:fever"],
                "response": "Headache and fever can indicate various conditions like flu, infection, or other illnesses. Monitor your temperature and seek medical attention if fever persists above 103°F or if accompanied by other severe symptoms."
            },
            {
                "question": "What is the dosage for aspirin?",
                "intent": "medication_info",
                "entities": ["MEDICATION:aspirin"],
                "response": "Typical aspirin dosage for pain relief is 325-650mg every 4-6 hours, not exceeding 3000mg daily. Always follow your doctor's prescription or package instructions. Consult a healthcare provider before use if you have allergies or medical conditions."
            },
            {
                "question": "How much water should I drink daily?",
                "intent": "lifestyle_advice",
                "entities": ["BODY_PART:body"],
                "response": "General recommendation is 8 glasses (64 ounces) of water daily, though individual needs vary. Factors like activity level, climate, and health conditions affect water requirements. Listen to your body and drink when thirsty."
            },
            {
                "question": "What are benefits of exercise?",
                "intent": "exercise_advice",
                "entities": ["TREATMENT:exercise"],
                "response": "Exercise provides numerous benefits: improves cardiovascular health, strengthens muscles, aids weight management, boosts mental health, reduces disease risk, and increases energy. Aim for 150 minutes of moderate activity weekly."
            },
            {
                "question": "What should I eat for better nutrition?",
                "intent": "nutrition_advice",
                "entities": ["TREATMENT:nutrition"],
                "response": "A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit processed foods, sugar, and salt. Include foods rich in vitamins and minerals. Consult a nutritionist for personalized advice."
            },
            {
                "question": "I'm having chest pain, what should I do?",
                "intent": "emergency_info",
                "entities": ["SYMPTOM:chest_pain"],
                "response": "Chest pain can indicate serious conditions. If you experience severe chest pain, shortness of breath, or dizziness, CALL EMERGENCY SERVICES (911) immediately. Do not delay seeking medical attention."
            },
            {
                "question": "Can I take ibuprofen with other medications?",
                "intent": "medication_info",
                "entities": ["MEDICATION:ibuprofen"],
                "response": "Ibuprofen can interact with many medications including blood thinners, certain blood pressure medications, and other NSAIDs. Always inform your doctor about all medications you're taking before starting ibuprofen. Consult a pharmacist for interactions."
            },
            {
                "question": "What causes high blood pressure?",
                "intent": "general_health",
                "entities": ["CONDITION:high_blood_pressure"],
                "response": "High blood pressure can be caused by genetics, age, poor diet, lack of exercise, stress, obesity, and excessive salt/alcohol intake. Regular monitoring and lifestyle changes can help manage it. Consult your doctor for treatment options."
            },
            {
                "question": "How can I improve my sleep quality?",
                "intent": "lifestyle_advice",
                "entities": ["TREATMENT:sleep"],
                "response": "Improve sleep by maintaining a consistent schedule, creating a dark and cool environment, avoiding screens before bed, limiting caffeine, exercising regularly, and managing stress. If sleep problems persist, consult a healthcare provider."
            },
            {
                "question": "What are symptoms of diabetes?",
                "intent": "symptom_inquiry",
                "entities": ["CONDITION:diabetes"],
                "response": "Diabetes symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow wound healing. Type 1 and Type 2 may have different presentations. Consult a doctor for proper testing and diagnosis."
            },
            {
                "question": "How to manage stress?",
                "intent": "lifestyle_advice",
                "entities": ["TREATMENT:stress_management"],
                "response": "Manage stress through deep breathing, meditation, regular exercise, adequate sleep, social support, and mindfulness. Consider professional help like therapy if stress is overwhelming. Various techniques work for different people."
            }
        ]
        
        df = pd.DataFrame(sample_data)
        
        # Ensure output directory exists
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        
        # Save dataset
        df.to_csv(QA_DATASET_PATH, index=False)
        print(f"Sample dataset created: {QA_DATASET_PATH}")
        return df
    
    @staticmethod
    def load_qa_dataset():
        """Load QA dataset from CSV"""
        if not os.path.exists(QA_DATASET_PATH):
            print(f"Dataset not found at {QA_DATASET_PATH}. Creating sample dataset...")
            return HealthcareDataLoader.create_sample_qa_dataset()
        
        df = pd.read_csv(QA_DATASET_PATH)
        print(f"Loaded dataset with {len(df)} Q&A pairs")
        return df
    
    @staticmethod
    def get_intents_from_dataset():
        """Get unique intents from dataset"""
        df = HealthcareDataLoader.load_qa_dataset()
        intents = df['intent'].unique().tolist()
        return intents
    
    @staticmethod
    def get_entities_from_dataset():
        """Get unique entities from dataset"""
        df = HealthcareDataLoader.load_qa_dataset()
        all_entities = set()
        
        for entities_str in df['entities']:
            if pd.notna(entities_str):
                # Parse entity string
                entities = str(entities_str).replace('[', '').replace(']', '').split(',')
                for entity in entities:
                    entity_type = entity.strip().split(':')[0]
                    all_entities.add(entity_type)
        
        return list(all_entities)
