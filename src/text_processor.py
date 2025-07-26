# src/text_processor.py

import json
import os

class TextProcessor:
    """
    Class ini HANYA bertanggung jawab untuk memuat
    data chunk yang sudah diproses sebelumnya.
    Proses pembuatan chunk ada di dalam file embeddings.py.
    """
    def load_chunks(self, filepath: str) -> list:
        """
        Memuat chunks dari file JSON yang sudah ada.
        """
        if not os.path.exists(filepath):
            print(f"❌ Error: File chunks tidak ditemukan di {filepath}")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"✅ Berhasil memuat {len(chunks)} chunks dari {filepath}")
            return chunks
        except json.JSONDecodeError:
            print(f"❌ Error: Gagal membaca format JSON di {filepath}")
            return []