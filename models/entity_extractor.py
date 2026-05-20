"""
Entity Extraction for Healthcare Chatbot
"""
import re
import pickle
import os
from config.settings import ENTITY_PATTERNS_PATH, ENTITY_TYPES


class EntityExtractor:
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.load_or_create_patterns()
    
    def _initialize_patterns(self):
        """Initialize entity extraction patterns"""
        patterns = {
            'SYMPTOM': [
                r'\b(headache|fever|cough|sore throat|runny nose|sneezing|fatigue|nausea|vomiting|diarrhea|shortness of breath|chest pain|dizziness|back pain|muscle pain|joint pain|rash|itching|bleeding)\b',
                r'\b(cold|flu|infection|migraine|arthritis|asthma|gastritis|bronchitis)\b'
            ],
            'BODY_PART': [
                r'\b(head|chest|back|stomach|arm|leg|hand|foot|eye|ear|nose|throat|heart|lung|liver|kidney|brain|spine|joint|muscle)\b'
            ],
            'MEDICATION': [
                r'\b(aspirin|ibuprofen|acetaminophen|paracetamol|penicillin|antibiotic|vaccine|insulin|metformin|lisinopril|atorvastatin|omeprazole|loratadine)\b',
                r'\b(tylenol|advil|bayer|lipitor|zocor|viagra|cialis)\b'
            ],
            'CONDITION': [
                r'\b(diabetes|hypertension|asthma|arthritis|cancer|heart disease|stroke|pneumonia|bronchitis|gastritis|depression|anxiety|flu|covid|tuberculosis|hepatitis)\b'
            ],
            'TREATMENT': [
                r'\b(exercise|medication|surgery|therapy|diet|nutrition|rest|sleep|vaccine|physical therapy|counseling|meditation|yoga)\b'
            ]
        }
        return patterns
    
    def load_or_create_patterns(self):
        """Load patterns from file or create new ones"""
        if os.path.exists(ENTITY_PATTERNS_PATH):
            try:
                with open(ENTITY_PATTERNS_PATH, 'rb') as f:
                    self.patterns = pickle.load(f)
                print(f"Loaded entity patterns from {ENTITY_PATTERNS_PATH}")
            except:
                self._save_patterns()
        else:
            self._save_patterns()
    
    def _save_patterns(self):
        """Save patterns to file"""
        os.makedirs(os.path.dirname(ENTITY_PATTERNS_PATH) or '.', exist_ok=True)
        with open(ENTITY_PATTERNS_PATH, 'wb') as f:
            pickle.dump(self.patterns, f)
        print(f"Saved entity patterns to {ENTITY_PATTERNS_PATH}")
    
    def extract(self, text):
        """Extract entities from text"""
        text_lower = text.lower()
        entities = []
        
        for entity_type, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entity_value = match.group(0)
                    entities.append({
                        'type': entity_type,
                        'value': entity_value,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Remove duplicates and sort by position
        unique_entities = []
        seen = set()
        for entity in sorted(entities, key=lambda x: x['start']):
            key = (entity['type'], entity['value'])
            if key not in seen:
                unique_entities.append(entity)
                seen.add(key)
        
        return unique_entities
    
    def extract_simplified(self, text):
        """Extract entities and return simplified list of (type, value) tuples"""
        entities = self.extract(text)
        return [(e['type'], e['value']) for e in entities]
    
    def add_pattern(self, entity_type, pattern):
        """Add a new extraction pattern"""
        if entity_type not in self.patterns:
            self.patterns[entity_type] = []
        
        if pattern not in self.patterns[entity_type]:
            self.patterns[entity_type].append(pattern)
            self._save_patterns()
