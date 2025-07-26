"""
RAG Chain yang menggabungkan retrieval dan generation
"""
import os
from typing import Dict, List
import requests
import json

class RAGChain:
    def __init__(self, vector_store, retriever, llm_provider="groq"):
        self.vector_store = vector_store
        self.retriever = retriever
        self.llm_provider = llm_provider
        
        # Setup LLM API
        if llm_provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model_name = "llama3-8b-8192"
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using LLM"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Anda adalah asisten AI yang ahli dalam sejarah kemerdekaan Indonesia. Jawab pertanyaan berdasarkan konteks yang diberikan dengan akurat dan informatif."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, question: str) -> Dict:
        """Main query method untuk RAG"""
        # 1. Retrieve relevant context
        context = self.retriever.retrieve_context(question, k=3)
        
        # 2. Format prompt
        prompt = self.retriever.format_prompt(question, context)
        
        # 3. Generate response
        response = self.generate_response(prompt)
        
        # 4. Return hasil lengkap
        return {
            "question": question,
            "context": context,
            "response": response,
            "prompt": prompt
        }