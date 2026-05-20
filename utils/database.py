"""
Database operations for Healthcare Chatbot using SQLite
"""
import sqlite3
import os
from datetime import datetime
from config.settings import DB_PATH


class ChatbotDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'bot')),
                content TEXT NOT NULL,
                intent TEXT,
                entities TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_conversation(self, session_id, user_name=None):
        """Create a new conversation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO conversations (session_id, user_name)
                VALUES (?, ?)
            ''', (session_id, user_name))
            conn.commit()
            conv_id = cursor.lastrowid
            return conv_id
        except sqlite3.IntegrityError:
            # Session already exists
            cursor.execute('SELECT id FROM conversations WHERE session_id = ?', (session_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()
    
    def add_message(self, conversation_id, role, content, intent=None, entities=None, confidence=None):
        """Add a message to a conversation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Convert entities list to string if needed
        entities_str = str(entities) if entities else None
        
        cursor.execute('''
            INSERT INTO messages (conversation_id, role, content, intent, entities, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (conversation_id, role, content, intent, entities_str, confidence))
        
        # Update conversation's updated_at timestamp
        cursor.execute('''
            UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (conversation_id,))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, conversation_id, limit=None):
        """Get all messages in a conversation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if limit:
            cursor.execute('''
                SELECT * FROM messages 
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (conversation_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM messages 
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            ''', (conversation_id,))
        
        messages = cursor.fetchall()
        conn.close()
        
        return [dict(msg) for msg in messages]
    
    def get_conversation_by_session(self, session_id):
        """Get conversation ID by session ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM conversations WHERE session_id = ?', (session_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def clear_conversation(self, conversation_id):
        """Clear all messages in a conversation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        conn.commit()
        conn.close()
    
    def get_all_conversations(self):
        """Get all conversations"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, session_id, user_name, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
        ''')
        
        conversations = cursor.fetchall()
        conn.close()
        
        return [dict(conv) for conv in conversations]
