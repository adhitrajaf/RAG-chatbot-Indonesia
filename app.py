"""
Main application untuk RAG Chatbot
"""
import os
import sys
sys.path.append('src')

from src.text_processor import TextProcessor
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.rag_chain import RAGChain

class RAGChatbot:
    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        self.setup()
    
    def setup(self):
        """Setup RAG system"""
        print("🚀 Setting up RAG Chatbot...")
        
        # 1. Load processed chunks
        processor = TextProcessor()
        chunks = processor.load_chunks("data/processed/text_chunks.json")
        
        if not chunks:
            print("❌ No chunks found! Please run text processing first.")
            return
        
        # 2. Setup vector store
        print("📊 Building vector store...")
        self.vector_store = VectorStore()
        
        # Check if vector store already exists
        if os.path.exists("data/vector_db/vector_store.json"):
            print("📂 Loading existing vector store...")
            self.vector_store.load("data/vector_db/vector_store")
        else:
            print("🔄 Creating new vector store...")
            self.vector_store.build_index(chunks)
            self.vector_store.save("data/vector_db/vector_store")
        
        # 3. Setup retriever dan RAG chain
        self.retriever = RAGRetriever(self.vector_store)
        self.rag_chain = RAGChain(self.vector_store, self.retriever)
        
        print("✅ RAG Chatbot ready!")
    
    def chat(self):
        """Interactive chat interface"""
        print("\n🤖 RAG Chatbot - Sejarah Kemerdekaan Indonesia")
        print("Ketik 'quit' untuk keluar\n")
        
        while True:
            question = input("❓ Tanya: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Sampai jumpa!")
                break
            
            if not question:
                continue
            
            print("🔍 Mencari jawaban...")
            result = self.rag_chain.query(question)
            
            print(f"\n🤖 Jawaban: {result['response']}\n")
            print("-" * 50)

def main():
    chatbot = RAGChatbot()
    chatbot.chat()

if __name__ == "__main__":
    main()