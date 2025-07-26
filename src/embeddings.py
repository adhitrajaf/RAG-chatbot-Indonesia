"""
Text processor untuk memotong teks menjadi chunks dan membuat embeddings
"""
import json
import os
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Setup text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """
        Bersihkan teks dari karakter yang tidak diinginkan
        """
        # Hapus referensi Wikipedia seperti [1], [2], dll
        text = re.sub(r'\[\d+\]', '', text)
        
        # Hapus multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Hapus karakter aneh
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"]', '', text)
        
        return text.strip()
    
    def create_chunks(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Potong artikel menjadi chunks dengan metadata
        """
        all_chunks = []
        
        for article in articles:
            print(f"Processing: {article['title']}")
            
            # Bersihkan teks
            clean_content = self.clean_text(article['content'])
            
            # Split menjadi chunks
            chunks = self.text_splitter.split_text(clean_content)
            
            # Tambah metadata ke setiap chunk
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'content': chunk,
                    'source_title': article['title'],
                    'source_url': article['url'],
                    'source_type': article['source'],
                    'chunk_id': f"{article['title'].lower().replace(' ', '_')}_chunk_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                all_chunks.append(chunk_data)
            
            print(f"  → Created {len(chunks)} chunks")
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, str]], output_path: str):
        """
        Simpan chunks ke file JSON
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(chunks)} chunks to {output_path}")
    
    def load_chunks(self, input_path: str) -> List[Dict[str, str]]:
        """
        Load chunks dari file JSON
        """
        if os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"✅ Loaded {len(chunks)} chunks from {input_path}")
            return chunks
        else:
            print(f"❌ File not found: {input_path}")
            return []
    
    def get_chunk_stats(self, chunks: List[Dict[str, str]]) -> Dict:
        """
        Dapatkan statistik chunks
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk['content']) for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_characters': sum(chunk_lengths),
            'unique_sources': len(set(chunk['source_title'] for chunk in chunks))
        }
        
        return stats

def main():
    """
    Main function untuk test text processing
    """
    print("🔄 Starting text processing...")
    
    # Load artikel dari file JSON langsung (TANPA IMPORT ERROR)
    articles_path = "../data/raw_texts/all_articles.json"
    
    if not os.path.exists(articles_path):
        print("❌ No articles found! Please run 2_data_loader.py first")
        return
    
    # Load articles dari JSON
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"✅ Loaded {len(articles)} articles from {articles_path}")
    
    # Process artikel menjadi chunks
    processor = TextProcessor(chunk_size=1000, chunk_overlap=200)
    chunks = processor.create_chunks(articles)
    
    # Simpan chunks
    output_path = "../data/processed/text_chunks.json"
    processor.save_chunks(chunks, output_path)
    
    # Tampilkan statistik
    stats = processor.get_chunk_stats(chunks)
    print(f"\n📊 CHUNK STATISTICS:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average chunk length: {stats['avg_chunk_length']:.0f} characters")
    print(f"Min chunk length: {stats['min_chunk_length']} characters")
    print(f"Max chunk length: {stats['max_chunk_length']} characters")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Unique sources: {stats['unique_sources']}")
    
    # Tampilkan contoh chunk
    if chunks:
        print(f"\n📄 SAMPLE CHUNK:")
        print(f"Title: {chunks[0]['source_title']}")
        print(f"Content: {chunks[0]['content'][:200]}...")

if __name__ == "__main__":
    main()