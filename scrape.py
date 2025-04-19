import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import sys
from datetime import datetime

RAW_FOLDER = "data/raw"
os.makedirs(RAW_FOLDER, exist_ok=True)

companies = {
    "Flipkart": "https://www.trustpilot.com/review/www.flipkart.com",
    "Amazon": "https://www.trustpilot.com/review/www.amazon.in",
    "Meesho": "https://www.trustpilot.com/review/meesho.com",
    "Myntra": "https://www.trustpilot.com/review/www.myntra.com"
}

def scrape_company_reviews(company_name, base_url, pages=10):
    all_reviews = []
    for page in range(1, pages + 1):
        print(f"[{company_name}] Scraping page {page}...")
        url = f"{base_url}?page={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        review_blocks = soup.find_all("article")

        for review in review_blocks:
            try:
                title = review.find("h2").text.strip() if review.find("h2") else "No Title"
                body = review.find("p").text.strip() if review.find("p") else "No Text"
                rating = review.find("div", {"data-service-review-rating": True})
                rating = rating["data-service-review-rating"] if rating else "N/A"
                date_tag = review.find("time")
                review_date = date_tag["datetime"].split("T")[0] if date_tag else "N/A"

                all_reviews.append({
                    "Company": company_name,
                    "Review Title": title,
                    "Rating": rating,
                    "Review Text": body,
                    "Review Date": review_date
                })
            except Exception as e:
                print(f"Error parsing review for {company_name}: {e}")
                continue

        time.sleep(1)

    new_df = pd.DataFrame(all_reviews)
    filename = os.path.join(RAW_FOLDER, f"{company_name.lower()}_reviews.csv")

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["Review Text", "Review Date"], inplace=True)
        combined_df.to_csv(filename, index=False)
        print(f"✅ [{company_name}] Updated: {len(combined_df)} total unique reviews.")
    else:
        new_df.to_csv(filename, index=False)
        print(f"✅ [{company_name}] First scrape done. {len(new_df)} reviews saved.")

# Read companies from command-line args
selected_companies = sys.argv[1:] if len(sys.argv) > 1 else companies.keys()

# Scrape only selected companies
for name in selected_companies:
    if name in companies:
        scrape_company_reviews(name, companies[name], pages=50)
    else:
        print(f"⚠️ Unknown company: {name}")
