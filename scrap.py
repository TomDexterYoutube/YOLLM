import os
import requests
from bs4 import BeautifulSoup
from googlesearch import search  # pip install googlesearch-python

os.system("clear")

class GoogleTextCrawler:
    def __init__(self, storage):
        self.storage = storage
        os.makedirs(self.storage['root_dir'], exist_ok=True)

    def crawl(self, keyword, max_num=10):
        urls = list(search(keyword, num_results=max_num))
        print(f"Found {len(urls)} URLs for '{keyword}'")

        for i, url in enumerate(urls):
            try:
                print(f"[{i+1}/{len(urls)}] Fetching: {url}")
                r = requests.get(url, timeout=5)
                soup = BeautifulSoup(r.text, 'html.parser')
                texts = soup.get_text(separator='\n', strip=True)

                file_path = os.path.join(self.storage['root_dir'], f"page_{i+1}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(texts)
                print(f"Saved to {file_path}")
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")

if __name__ == "__main__":
    query = input("keyword/ ")
    max_num = int(input("max pages/ "))
    path = "scrap"

    crawler = GoogleTextCrawler(storage={'root_dir': path})
    crawler.crawl(keyword=query, max_num=max_num)
