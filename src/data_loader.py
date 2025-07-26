import requests
from bs4 import BeautifulSoup
import os
import time
import json
from typing import List, Dict

class WikipediaDataLoader:
    def __init__(self, output_dir: str = "../data/raw_texts"):
        self.output_dir = output_dir
        self.base_url = "https://id.wikipedia.org"
        
        # Buat direktori jika belum ada
        os.makedirs(output_dir, exist_ok=True)
        
        # URL artikel sejarah kemerdekaan Indonesia
        self.article_urls = {
            "hindia_belanda": "/wiki/Hindia_Belanda",
            "pendudukan_jepang": "/wiki/Pendudukan_Jepang_di_Indonesia",
            "proklamasi_kemerdekaan": "/wiki/Proklamasi_Kemerdekaan_Indonesia",
            "revolusi_nasional": "/wiki/Revolusi_Nasional_Indonesia",
            "agresi_militer_1": "/wiki/Agresi_Militer_Belanda_I",
            "agresi_militer_2": "/wiki/Agresi_Militer_Belanda_II",
            "soekarno": "/wiki/Soekarno",
            "hatta": "/wiki/Mohammad_Hatta",
            "budi_utomo": "/wiki/Budi_Utomo",
            "sumpah_pemuda": "/wiki/Sumpah_Pemuda"
        }
    
    def scrape_wikipedia_article(self, url: str) -> Dict[str, str]:
        """
        Scrape satu artikel Wikipedia
        """
        try:
            print(f"Scraping: {url}")
            
            # Headers untuk menghindari blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(self.base_url + url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Ambil judul artikel
            title = soup.find('h1', {'class': 'firstHeading'}).get_text().strip()
            
            # Ambil konten dari paragraf utama
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            paragraphs = content_div.find_all('p')
            
            # Gabungkan semua paragraf
            content = ""
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 50:  # Filter paragraf yang terlalu pendek
                    content += text + "\n\n"
            
            return {
                "title": title,
                "content": content.strip(),
                "url": self.base_url + url,
                "source": "Wikipedia Indonesia"
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
    
    def scrape_all_articles(self) -> List[Dict[str, str]]:
        """
        Scrape semua artikel yang sudah didefinisikan
        """
        articles = []
        
        for article_name, url in self.article_urls.items():
            print(f"\n--- Scraping {article_name} ---")
            
            article_data = self.scrape_wikipedia_article(url)
            
            if article_data:
                # Simpan ke file individual
                filename = f"{article_name}.json"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(article_data, f, ensure_ascii=False, indent=2)
                
                articles.append(article_data)
                print(f"✅ Saved: {filename}")
            else:
                print(f"❌ Failed to scrape: {article_name}")
            
            # Delay untuk menghindari rate limiting
            time.sleep(2)
        
        # Simpan semua artikel dalam satu file
        all_articles_path = os.path.join(self.output_dir, "all_articles.json")
        with open(all_articles_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Total articles scraped: {len(articles)}")
        print(f"✅ All articles saved to: {all_articles_path}")
        
        return articles
    
    def load_existing_articles(self) -> List[Dict[str, str]]:
        """
        Load artikel yang sudah di-scrape sebelumnya
        """
        all_articles_path = os.path.join(self.output_dir, "all_articles.json")
        
        if os.path.exists(all_articles_path):
            with open(all_articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"✅ Loaded {len(articles)} existing articles")
            return articles
        else:
            print("❌ No existing articles found")
            return []
    
    def add_custom_text(self, title: str, content: str, save: bool = True) -> Dict[str, str]:
        """
        Tambah teks custom ke knowledge base
        """
        article_data = {
            "title": title,
            "content": content,
            "url": "Custom Input",
            "source": "User Input"
        }
        
        if save:
            # Simpan ke file
            filename = f"custom_{title.lower().replace(' ', '_')}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Custom article saved: {filename}")
        
        return article_data

def main():
    """
    Main function untuk test scraping
    """
    print("🚀 Starting Wikipedia scraping for Indonesian Independence History...")
    
    loader = WikipediaDataLoader()
    
    # Cek apakah sudah ada artikel yang di-scrape
    existing_articles = loader.load_existing_articles()
    
    if not existing_articles:
        print("📥 No existing articles found. Starting fresh scraping...")
        articles = loader.scrape_all_articles()
    else:
        print("📂 Found existing articles. Use them or re-scrape?")
        choice = input("Enter 'r' to re-scrape, any other key to use existing: ")
        
        if choice.lower() == 'r':
            articles = loader.scrape_all_articles()
        else:
            articles = existing_articles
    
    # Tampilkan summary
    print(f"\n📊 SUMMARY:")
    print(f"Total articles: {len(articles)}")
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['title']} ({len(article['content'])} chars)")

if __name__ == "__main__":
    main()