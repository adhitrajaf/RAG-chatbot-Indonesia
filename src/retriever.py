"""
Retriever untuk menggabungkan query processing dan context retrieval
"""
import re
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGRetriever:
    def __init__(self, vector_store, max_context_length: int = 3000):
        self.vector_store = vector_store
        self.max_context_length = max_context_length
        
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess user query untuk search yang lebih baik
        """
        # Bersihkan query tapi pertahankan struktur penting
        query = query.strip()
        
        # Hapus karakter yang tidak perlu
        query = re.sub(r'[^\w\s\?\!\.]', '', query)
        
        # Normalisasi spasi
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    def enhance_query(self, query: str) -> List[str]:
        """
        Enhance query dengan variasi untuk search yang lebih baik
        """
        enhanced_queries = [query]
        
        # Query expansions untuk topik sejarah Indonesia
        expansions = {
            'proklamasi': ['proklamasi kemerdekaan', 'merdeka', '17 agustus 1945'],
            'soekarno': ['sukarno', 'presiden pertama', 'bung karno'],
            'hatta': ['mohammad hatta', 'bung hatta', 'wakil presiden'],
            'belanda': ['hindia belanda', 'kolonial', 'penjajahan belanda'],
            'jepang': ['pendudukan jepang', 'dai nippon', 'jepang indonesia'],
            'agresi': ['agresi militer', 'operasi militer', 'serangan belanda'],
            'revolusi': ['revolusi nasional', 'perang kemerdekaan', 'perjuangan']
        }
        
        query_lower = query.lower()
        for key, variants in expansions.items():
            if key in query_lower:
                enhanced_queries.extend(variants)
        
        return enhanced_queries
    
    def retrieve_context(self, query: str, k: int = 5, min_score: float = 0.2) -> Dict:
        """
        Retrieve relevant context untuk RAG
        """
        # Preprocess query
        clean_query = self.preprocess_query(query)
        
        # Enhance query untuk search yang lebih baik
        enhanced_queries = self.enhance_query(clean_query)
        
        # Collect results dari semua query variants
        all_results = []
        seen_chunks = set()
        
        for eq in enhanced_queries:
            results = self.vector_store.search(eq, k=k, min_score=min_score)
            
            for result in results:
                chunk_id = result['chunk']['chunk_id']
                if chunk_id not in seen_chunks:
                    all_results.append(result)
                    seen_chunks.add(chunk_id)
        
        # Sort by score dan ambil top k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = all_results[:k]
        
        # Build context dari chunks yang relevan
        context_parts = []
        total_length = 0
        used_sources = set()
        
        for result in top_results:
            chunk = result['chunk']
            score = result['score']
            
            # Skip jika context sudah terlalu panjang
            if total_length + len(chunk['content']) > self.max_context_length:
                continue
            
            # Format context dengan metadata
            source_info = f"[Sumber: {chunk['source_title']}]"
            content = f"{source_info}\n{chunk['content']}"
            
            context_parts.append({
                'content': content,
                'score': score,
                'source': chunk['source_title'],
                'chunk_id': chunk['chunk_id']
            })
            
            total_length += len(content)
            used_sources.add(chunk['source_title'])
        
        # Gabungkan semua context
        full_context = "\n\n---\n\n".join([part['content'] for part in context_parts])
        
        return {
            'context': full_context,
            'context_parts': context_parts,
            'total_length': total_length,
            'used_sources': list(used_sources),
            'num_chunks': len(context_parts),
            'avg_score': sum([part['score'] for part in context_parts]) / len(context_parts) if context_parts else 0
        }
    
    def format_prompt(self, query: str, context_data: Dict) -> str:
        """
        Format prompt untuk LLM dengan context yang telah di-retrieve
        """
        context = context_data['context']
        sources = context_data['used_sources']
        
        # System instruction
        system_prompt = """Anda adalah asisten AI yang ahli dalam sejarah kemerdekaan Indonesia. 
Tugas Anda adalah menjawab pertanyaan berdasarkan konteks sejarah yang diberikan dengan akurat dan informatif.

INSTRUKSI:
1. Jawab berdasarkan informasi dalam konteks yang diberikan
2. Jika informasi tidak cukup dalam konteks, katakan dengan jelas
3. Berikan jawaban yang faktual dan objektif
4. Gunakan bahasa Indonesia yang baik dan benar
5. Sebutkan sumber informasi jika relevan"""
        
        # Format prompt
        prompt = f"""{system_prompt}

KONTEKS SEJARAH:
{context}

SUMBER REFERENSI: {', '.join(sources)}

PERTANYAAN: {query}

JAWABAN: Berdasarkan informasi sejarah di atas,"""
        
        return prompt
    
    def get_relevant_sources(self, query: str, k: int = 3) -> List[Dict]:
        """
        Dapatkan daftar sumber yang relevan dengan query
        """
        results = self.vector_store.search(query, k=k)
        
        sources = []
        seen_sources = set()
        
        for result in results:
            chunk = result['chunk']
            source_title = chunk['source_title']
            
            if source_title not in seen_sources:
                sources.append({
                    'title': source_title,
                    'url': chunk['source_url'],
                    'relevance_score': result['score'],
                    'type': chunk['source_type']
                })
                seen_sources.add(source_title)
        
        return sources
    
    def evaluate_retrieval(self, query: str, retrieved_context: Dict) -> Dict:
        """
        Evaluate kualitas retrieval
        """
        context_parts = retrieved_context['context_parts']
        
        if not context_parts:
            return {
                'quality': 'poor',
                'score': 0.0,
                'issues': ['No relevant context found'],
                'recommendations': ['Try rephrasing the query', 'Add more specific keywords']
            }
        
        avg_score = retrieved_context['avg_score']
        num_chunks = retrieved_context['num_chunks']
        context_length = retrieved_context['total_length']
        
        issues = []
        recommendations = []
        
        # Evaluate berdasarkan berbagai faktor
        if avg_score < 0.3:
            issues.append('Low relevance scores')
            recommendations.append('Consider expanding the knowledge base')
        
        if num_chunks < 2:
            issues.append('Limited context diversity')
            recommendations.append('Lower the minimum score threshold')
        
        if context_length > self.max_context_length * 0.9:
            issues.append('Context near maximum length')
            recommendations.append('Consider increasing max_context_length')
        
        # Determine overall quality
        if avg_score >= 0.5 and num_chunks >= 3:
            quality = 'excellent'
        elif avg_score >= 0.3 and num_chunks >= 2:
            quality = 'good'
        elif avg_score >= 0.2 and num_chunks >= 1:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'quality': quality,
            'score': avg_score,
            'num_chunks': num_chunks,
            'context_length': context_length,
            'issues': issues,
            'recommendations': recommendations
        }

def main():
    """
    Main function untuk test retriever
    """
    import sys
    sys.path.append('.')
    from src.vector_store import VectorStore
    
    print("🚀 Testing RAG Retriever...")
    
    # Load vector store
    vector_store = VectorStore()
    
    if not vector_store.load():
        print("❌ Could not load vector store! Please run 4_vector_store.py first")
        return
    
    # Initialize retriever
    retriever = RAGRetriever(vector_store)
    
    # Test queries
    test_queries = [
        "Siapa yang memproklamasikan kemerdekaan Indonesia?",
        "Kapan terjadi Agresi Militer Belanda I?",
        "Apa peran Soekarno dalam proklamasi kemerdekaan?",
        "Bagaimana kondisi Indonesia saat pendudukan Jepang?",
        "Siapa saja tokoh dalam organisasi Budi Utomo?"
    ]
    
    print("\n🔍 Testing retrieval for different queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        # Retrieve context
        context_data = retriever.retrieve_context(query, k=3)
        
        print(f"Retrieved {context_data['num_chunks']} chunks")
        print(f"Total context length: {context_data['total_length']} characters")
        print(f"Average relevance score: {context_data['avg_score']:.3f}")
        print(f"Sources used: {', '.join(context_data['used_sources'])}")
        
        # Evaluate retrieval quality
        evaluation = retriever.evaluate_retrieval(query, context_data)
        print(f"Retrieval quality: {evaluation['quality'].upper()}")
        
        if evaluation['issues']:
            print(f"Issues: {', '.join(evaluation['issues'])}")
        
        # Show sample context
        if context_data['context_parts']:
            first_chunk = context_data['context_parts'][0]
            print(f"Sample context: {first_chunk['content'][:200]}...")
        
        # Format prompt
        prompt = retriever.format_prompt(query, context_data)
        print(f"Prompt length: {len(prompt)} characters")

if __name__ == "__main__":
    main()