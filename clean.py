import os
import pandas as pd
import re
import json
from nltk.corpus import stopwords
import nltk

# Ensure NLTK stopwords are available
nltk.download('stopwords')

RAW_FOLDER = "data/raw"
CLEANED_FOLDER = "data/cleaned"
KEYWORDS_FILE = "expanded_category_keywords.json"

os.makedirs(CLEANED_FOLDER, exist_ok=True)

# Load keyword mapping
with open(KEYWORDS_FILE, "r") as f:
    CATEGORY_KEYWORDS = json.load(f)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def assign_category(review_text):
    review_text = str(review_text).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in review_text for keyword in keywords):
            return category
    return "General"

# Process all raw CSVs
for filename in os.listdir(RAW_FOLDER):
    if filename.endswith(".csv"):
        file_path = os.path.join(RAW_FOLDER, filename)
        df = pd.read_csv(file_path)

        # Drop rows with missing review text or rating
        df.dropna(subset=["Review Text", "Rating"], inplace=True)

        # Clean review text and remove stopwords
        df["Review Text"] = df["Review Text"].apply(clean_text)

        # Assign product category
        df["Product Category"] = df["Review Text"].apply(assign_category)

        # Save cleaned file
        cleaned_filename = filename.replace(".csv", "_cleaned.csv")
        df.to_csv(os.path.join(CLEANED_FOLDER, cleaned_filename), index=False)
        print(f"✅ Cleaned and categorized: {cleaned_filename}")
