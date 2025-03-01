import requests
from bs4 import BeautifulSoup
import json

def scrape_reviews():
    url = "https://openreview.net/forum?id=00SnKBGTsz"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return
    
    soup = BeautifulSoup(response.text, "html.parser")
    reviews = soup.find_all("div", class_="note depth-odd")
    
    all_reviews = []
    for review in reviews:
        review_data = {
            "content": review.get_text(strip=True)
        }
        all_reviews.append(review_data)
    
    # Save reviews to a JSON file
    with open("reviews.json", "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully scraped {len(all_reviews)} reviews")

if __name__ == "__main__":
    scrape_reviews() 