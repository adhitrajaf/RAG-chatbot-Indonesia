# Chatbot RAG Indonesia 🇮🇩

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![AI](https://img.shields.io/badge/AI-brightgreen?style=for-the-badge&logo=artificialintelligence)](https://www.ai.gov/)
[![Chatbot](https://img.shields.io/badge/Chatbot-blue?style=for-the-badge&logo=discord)](https://discord.com/)
[![RAG](https://img.shields.io/badge/RAG-lightgrey?style=for-the-badge)](https://research.google/blog/improving-language-model-accuracy-through-retrieval/)

RAG Chatbot - Sejarah Kemerdekaan Indonesia
Proyek ini adalah sebuah chatbot berbasis RAG (Retrieval-Augmented Generation) yang dirancang untuk menjawab pertanyaan seputar sejarah kemerdekaan Indonesia. Chatbot ini menggunakan data yang diambil dari Wikipedia, membangun basis pengetahuan vektor, dan menggunakan Large Language Model (LLM) dari Groq untuk menghasilkan jawaban yang relevan dan kontekstual.

Fitur Utama
1. Pengambilan Data Otomatis: Mengambil artikel-artikel relevan dari Wikipedia Indonesia menggunakan requests dan BeautifulSoup.
2. Pemrosesan Teks: Membersihkan dan membagi teks menjadi potongan-potongan (chunks) yang lebih kecil dan mudah dikelola.
3. Vector Database: Membangun database vektor menggunakan FAISS untuk pencarian kemiripan (similarity search) yang cepat dan efisien.
4. Embeddings: Mengubah potongan teks menjadi representasi vektor menggunakan model dari sentence-transformers.
5. Retrieval-Augmented Generation (RAG): Menggabungkan proses pencarian konteks yang relevan dengan kemampuan generasi teks dari LLM (Llama 3 via Groq API) untuk memberikan jawaban yang akurat.
6. Antarmuka Interaktif: Berinteraksi dengan chatbot melalui terminal (command-line).

Struktur Proyek
.
├── data/
│   ├── raw_texts/      # Menyimpan hasil scrape artikel mentah
│   ├── processed/      # Menyimpan hasil olahan teks (chunks)
│   └── vector_db/      # Menyimpan file FAISS index dan data vektor
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Modul untuk scrape data dari Wikipedia
│   ├── embeddings.py     # Modul untuk memproses teks menjadi chunks
│   ├── text_processor.py # Modul utilitas untuk memuat data teks
│   ├── vector_store.py   # Modul untuk mengelola database vektor FAISS
│   ├── retriever.py      # Modul untuk mengambil konteks yang relevan
│   └── rag_chain.py      # Modul yang mengorkestrasi alur RAG
├── .env                  # File untuk menyimpan API Key (TIDAK MASUK GIT)
├── .gitignore            # Daftar file/folder yang diabaikan oleh Git
├── app.py                # Titik masuk utama untuk menjalankan chatbot
└── requirements.txt      # Daftar semua dependensi Python

Instalasi & Persiapan
Ikuti langkah-langkah berikut untuk menjalankan proyek ini di komputer Anda.
1. Clone Repositori
git clone https://github.com/adhitrajaf/RAG-chatbot-Indonesia.git
cd RAG-chatbot-Indonesia

2. Buat dan Aktifkan Virtual Environment
Sangat disarankan untuk menggunakan virtual environment agar dependensi proyek tidak tercampur.
# Buat environment baru
python -m venv venv

# Aktifkan di Windows
.\venv\Scripts\activate

# Aktifkan di macOS/Linux
source venv/bin/activate

3. Install Dependensi
Install semua library yang dibutuhkan dari file requirements.txt.
pip install -r requirements.txt

4. Siapkan API Key
Proyek ini membutuhkan API Key dari Groq untuk mengakses LLM.
a. Buat file baru di folder utama bernama .env.
b. Isi file tersebut dengan format berikut:
GROQ_API_KEY="gsk_API_KEY_ANDA_DI_SINI"

CARA PENGGUNAAN
Proyek ini memiliki alur kerja yang terdiri dari beberapa langkah. Jalankan skrip berikut secara berurutan dari terminal.

Langkah 1: Ambil Data dari Wikipedia
Jalankan skrip ini untuk mengunduh artikel-artikel sejarah dan menyimpannya di data/raw_texts/.
python -m src.data_loader

Langkah 2: Proses Teks Menjadi Chunks
Jalankan skrip ini untuk membaca data mentah, membersihkannya, dan membaginya menjadi file chunks yang disimpan di data/processed/text_chunks.json.
python -m src.embeddings

Langkah 3: Jalankan Chatbot
Setelah data siap, jalankan aplikasi utama. Saat pertama kali dijalankan, skrip ini akan otomatis membuat vector database dari chunks yang ada.
python app.py

Selanjutnya, Anda bisa langsung berinteraksi dan mengajukan pertanyaan pada chatbot melalui terminal.
