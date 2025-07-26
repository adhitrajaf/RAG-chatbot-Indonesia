"""
Vector store untuk menyimpan embeddings dan melakukan similarity search
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.vector_db_path = os.getenv('VECTOR_DB_PATH', './data/vector_db')
        
        # Buat direktori jika belum ada
        os.makedirs(self.vector_db_path, exist_ok=True)
        
    def create_embeddings(self, chunks: List[Dict[str, str]]) -> np.ndarray:
        """
        Create embeddings untuk semua chunks
        """
        print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
        texts = [chunk['content'] for chunk in chunks]
        
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # Normalize untuk cosine similarity
        )
        
        print(f"✅ Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_index(self, chunks: List[Dict[str, str]], embeddings: np.ndarray = None):
        """
        Build FAISS index dari chunks dan embeddings
        """
        print("🔄 Building FAISS index...")
        
        self.chunks = chunks
        
        if embeddings is None:
            self.embeddings = self.create_embeddings(chunks)
        else:
            self.embeddings = embeddings
            
        # Pastikan embeddings sudah normalized
        if not np.allclose(np.linalg.norm(self.embeddings, axis=1), 1.0, atol=1e-6):
            print("📐 Normalizing embeddings...")
            faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index untuk cosine similarity
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product untuk cosine similarity
        
        # Add embeddings ke index
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5, min_score: float = 0.1) -> List[Dict]:
        """
        Search chunks yang mirip dengan query
        """
        if self.index is None:
            print("❌ Index not built yet!")
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search dalam index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Filter berdasarkan minimum score
            if score >= min_score:
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(score),
                    'rank': i + 1,
                    'index': int(idx)
                })
        
        return results
    
    def add_document(self, title: str, content: str, source_type: str = "Custom"):
        """
        Tambah dokumen baru ke vector store
        """
        # Buat chunk baru
        new_chunk = {
            'content': content,
            'source_title': title,
            'source_url': 'Custom Input',
            'source_type': source_type,
            'chunk_id': f"{title.lower().replace(' ', '_')}_custom",
            'chunk_index': 0,
            'total_chunks': 1,
            'char_count': len(content)
        }
        
        # Create embedding untuk chunk baru
        new_embedding = self.model.encode([content], normalize_embeddings=True)
        
        # Add ke chunks dan embeddings
        self.chunks.append(new_chunk)
        
        if self.embeddings is None:
            self.embeddings = new_embedding
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])
        
        # Add ke FAISS index
        if self.index is None:
            dimension = new_embedding.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        
        self.index.add(new_embedding.astype('float32'))
        
        print(f"✅ Added new document: {title}")
        return len(self.chunks) - 1  # Return index dari dokumen baru
    def save(self, base_path: str):
        """
        Simpan index FAISS dan data chunks.
        """
        if self.index is None or not self.chunks:
            print("❌ Index atau chunks kosong, tidak ada yang disimpan.")
            return

        # 1. Pastikan direktori tujuan ada
        output_dir = os.path.dirname(base_path)
        os.makedirs(output_dir, exist_ok=True)

        # 2. Buat path yang benar untuk setiap file
        index_path = f"{base_path}.faiss"
        chunks_path = f"{base_path}_chunks.json"

        # 3. Simpan index FAISS
        faiss.write_index(self.index, index_path)
        
        # 4. Simpan metadata chunks
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ Vector store berhasil disimpan di '{output_dir}'")
        print(f"   - Index: {index_path}")
        print(f"   - Chunks: {chunks_path}")    
        
    # def save(self, filename: str = "vector_store"):
    #     """
    #     Simpan vector store ke file
    #     """
    #     if self.chunks is None or self.embeddings is None:
    #         print("❌ Nothing to save!")
    #         return
        
    #     # Path files
    #     chunks_path = os.path.join(self.vector_db_path, f"{filename}_chunks.json")
    #     embeddings_path = os.path.join(self.vector_db_path, f"{filename}_embeddings.npy")
    #     index_path = os.path.join(self.vector_db_path, f"{filename}.faiss")
    #     metadata_path = os.path.join(self.vector_db_path, f"{filename}_metadata.json")
        
    #     # Save chunks
    #     with open(chunks_path, 'w', encoding='utf-8') as f:
    #         json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
    #     # Save embeddings
    #     np.save(embeddings_path, self.embeddings)
        
    #     # Save FAISS index
    #     if self.index:
    #         faiss.write_index(self.index, index_path)
        
    #     # Save metadata
    #     metadata = {
    #         'model_name': self.model_name,
    #         'total_chunks': len(self.chunks),
    #         'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
    #         'index_type': 'IndexFlatIP'
    #     }
        
    #     with open(metadata_path, 'w', encoding='utf-8') as f:
    #         json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    #     print(f"✅ Vector store saved to {self.vector_db_path}")
    #     print(f"   - Chunks: {chunks_path}")
    #     print(f"   - Embeddings: {embeddings_path}")
    #     print(f"   - Index: {index_path}")
    #     print(f"   - Metadata: {metadata_path}")
    
    def load(self, filename: str = "vector_store"):
        """
        Load vector store dari file
        """
        # Path files
        chunks_path = os.path.join(self.vector_db_path, f"{filename}_chunks.json")
        embeddings_path = os.path.join(self.vector_db_path, f"{filename}_embeddings.npy")
        index_path = os.path.join(self.vector_db_path, f"{filename}.faiss")
        metadata_path = os.path.join(self.vector_db_path, f"{filename}_metadata.json")
        
        # Check if all files exist
        required_files = [chunks_path, embeddings_path, index_path, metadata_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        try:
            # Load chunks
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            # Load embeddings
            self.embeddings = np.load(embeddings_path)
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"✅ Vector store loaded successfully!")
            print(f"   - Total chunks: {len(self.chunks)}")
            print(f"   - Embedding dimension: {self.embeddings.shape[1]}")
            print(f"   - Model: {metadata.get('model_name', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Dapatkan statistik vector store
        """
        if not self.chunks:
            return {}
        
        chunk_lengths = [len(chunk['content']) for chunk in self.chunks]
        sources = [chunk['source_title'] for chunk in self.chunks]
        
        stats = {
            'total_chunks': len(self.chunks),
            'avg_chunk_length': np.mean(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'unique_sources': len(set(sources)),
            'sources': list(set(sources)),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'index_size': self.index.ntotal if self.index else 0
        }
        
        return stats

def main():
    """
    Main function untuk test vector store
    """
    print("🚀 Testing Vector Store...")
    
    # Load chunks dari file processed
    chunks_path = "./data/processed/text_chunks.json"
    
    if not os.path.exists(chunks_path):
        print("❌ Chunks file not found! Please run embeddings.py first")
        return
    
    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Check if vector store already exists
    if vector_store.load():
        print("📂 Using existing vector store")
    else:
        print("🔄 Building new vector store...")
        
        # Load embeddings if exists
        embeddings_path = "./data/processed/embeddings.npy"
        embeddings = None
        
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            print(f"✅ Loaded existing embeddings: {embeddings.shape}")
        
        # Build index
        vector_store.build_index(chunks, embeddings)
        
        # Save vector store
        vector_store.save()
    
    # Test search
    print("\n🔍 Testing search functionality...")
    test_queries = [
        "Siapa yang memproklamasikan kemerdekaan Indonesia?",
        "Kapan Jepang menduduki Indonesia?",
        "Apa itu Agresi Militer Belanda?",
        "Siapa pemimpin organisasi Budi Utomo?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.search(query, k=3, min_score=0.2)
        
        if results:
            for result in results:
                print(f"  Score: {result['score']:.3f} | Source: {result['chunk']['source_title']}")
                print(f"  Content: {result['chunk']['content'][:100]}...")
        else:
            print("  No relevant results found")
    
    # Display stats
    stats = vector_store.get_stats()
    print(f"\n📊 VECTOR STORE STATISTICS:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Unique sources: {stats['unique_sources']}")
    print(f"Average chunk length: {stats['avg_chunk_length']:.0f} characters")

if __name__ == "__main__":
    main()