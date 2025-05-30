import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uuid
import time
from datetime import datetime
import threading
from typing import Dict, List, Any

# Configure Streamlit for multi-user
st.set_page_config(
    page_title="Hipotronics Bot",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ConcurrentRAGSystem:
    """Thread-safe RAG system for multiple concurrent users"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.active_users = {}
        self.query_count = 0
        

    def load_models(self):
        """Load models once and cache them (shared across all users)"""
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load qa generation model

        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            tokenizer="google/flan-t5-small",
            device=-1  # CPU
        )


        
        return embedding_model, generator
    
    def load_knowledge_base(self):
        """Load pre-built FAISS index and documents (shared across users)"""
        try:
            # Load FAISS index
            index = faiss.read_index("troubleshooting_index.bin")
            
            # Load document chunks
            with open("troubleshooting_chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
                
            # Load metadata
            with open("troubleshooting_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
                
            return index, chunks, metadata
            
        except FileNotFoundError:
            st.error("Knowledge base not found. Please upload troubleshooting documents first.")
            return None, None, None
    
    def get_session_info(self):
        """Get or create session information for current user"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.user_name = f"User_{st.session_state.session_id}"
            st.session_state.messages = []
            st.session_state.query_history = []
            st.session_state.session_start = datetime.now()
            
        return st.session_state.session_id
    
    def update_active_users(self, session_id: str):
        """Track active users for monitoring"""
        with self.lock:
            self.active_users[session_id] = {
                'last_activity': datetime.now(),
                'query_count': len(st.session_state.get('query_history', []))
            }
            
            # Clean up inactive users (older than 30 minutes)
            cutoff_time = datetime.now()
            inactive_users = [
                uid for uid, info in self.active_users.items() 
                if (cutoff_time - info['last_activity']).seconds > 1800
            ]
            for uid in inactive_users:
                del self.active_users[uid]
    
    def search_documents(self, query: str, top_k: int = 3):
        """Search for relevant troubleshooting documents"""
        embedding_model, _ = self.load_models()
        index, chunks, metadata = self.load_knowledge_base()
        
        if index is None:
            return []
        
        # Create query embedding
        query_embedding = embedding_model.encode([query])
        
        # Search FAISS index
        scores, indices = index.search(query_embedding.astype('float32'), top_k)
        print(scores)
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(chunks) and score > 0.5:  # Relevance threshold
                results.append({
                    'content': chunks[idx],
                    'metadata': metadata[idx] if idx < len(metadata) else {},
                    'similarity': float(score),
                    'source': metadata[idx].get('source', 'Unknown') if idx < len(metadata) else 'Unknown'
                })
        
        return results
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
      """Generate response using retrieved context"""
      if not context_docs:
        return "I couldn't find relevant information in the troubleshooting guide for your question. Please try rephrasing or contact technical support."

      # Use top 2 most relevant context docs
      context = "\n\n".join([
          f"From {doc['source']}: {doc['content']}" 
          for doc in context_docs[:2]
      ])

      # Create instruction-style prompt for FLAN-T5
      prompt = f"""You are a helpful assistant. Use the troubleshooting guide below to answer the user's question.

  Troubleshooting Guide:
  {context}

  Question: {query}

  Answer:"""

      # Load generator
      _, generator = self.load_models()

      try:
          result = generator(prompt, max_new_tokens=300, do_sample=False)[0]['generated_text']
          print(result)
          answer = result.strip()
          print(answer)
          response_template = f"""**Answer:** {answer}

  **Sources:** {', '.join([doc['source'] for doc in context_docs[:2]])}
  """
      except Exception as e:
          print(e)
          st.exception(e)
          response_template = "Sorry, I couldn't generate a response due to an internal error."

      return response_template

    
    def is_repeated_query(self, query: str, threshold: float = 0.85) -> bool:
        """Check if query is too similar to recent queries"""
        if not st.session_state.get('query_history'):
            return False
            
        embedding_model, _ = self.load_models()
        query_embedding = embedding_model.encode([query])
        
        # Check against last 5 queries
        recent_queries = st.session_state.query_history[-5:]
        for prev_query in recent_queries:
            prev_embedding = embedding_model.encode([prev_query])
            similarity = np.dot(query_embedding[0], prev_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(prev_embedding[0])
            )
            
            if similarity > threshold:
                return True
        
        return False

# Initialize the RAG system
@st.cache_resource
def get_rag_system():
    return ConcurrentRAGSystem()

rag_system = get_rag_system()

def chatbot_fn(message, history):
    # Check for repeated query
    if rag_system.is_repeated_query(message):
        return "You've asked a similar question recently. Please rephrase."

    # Search documents
    relevant_docs = rag_system.search_documents(message)

    # Generate response
    response = rag_system.generate_response(message, relevant_docs)
    
    return response

# Launch Gradio chat interface
chat = gr.ChatInterface(fn=chatbot_fn, title="Hipotronics Assistant")
chat.launch()
