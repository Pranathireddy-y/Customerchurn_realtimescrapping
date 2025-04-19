import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CLEANED_FOLDER = "data/cleaned"
EDA_OUTPUT_FOLDER = "eda_output"
os.makedirs(EDA_OUTPUT_FOLDER, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()
KEYWORDS_FLAG = ["late", "refund", "scam", "fake", "delay", "cancel", "worst", "cheated", "bad", "broken"]

def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    return tokens

def classify_sentiment(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

for filename in os.listdir(CLEANED_FOLDER):
    if filename.endswith("_cleaned.csv"):
        file_path = os.path.join(CLEANED_FOLDER, filename)
        df = pd.read_csv(file_path)
        company_name = filename.replace("_cleaned.csv", "").capitalize()
        company_name_lower = company_name.lower()

        company_output = os.path.join(EDA_OUTPUT_FOLDER, company_name)
        os.makedirs(company_output, exist_ok=True)

        print(f"Analyzing {company_name}...")

        sentiment_scores = df["Review Text"].astype(str).apply(lambda x: analyzer.polarity_scores(x))
        df["Sentiment Score"] = sentiment_scores.apply(lambda x: x['compound'])
        df["Sentiment"] = df["Sentiment Score"].apply(classify_sentiment)
        df.to_csv(file_path, index=False)

        # 1. Rating Distribution
        plt.figure(figsize=(6, 4))
        df['Rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title(f"{company_name} - Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Number of Reviews")
        plt.tight_layout()
        plt.savefig(f"{company_output}/ratings.png")
        plt.close()

        # 2. Sentiment Distribution
        plt.figure(figsize=(6, 4))
        df["Sentiment"].value_counts().plot(kind='bar', color='salmon')
        plt.title(f"{company_name} - Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{company_output}/sentiment_distribution.png")
        plt.close()

        # 3. Top Words (filtered)
        all_words = []
        for text in df["Review Text"]:
            all_words.extend(clean_and_tokenize(text))

        filtered_words = [w for w in all_words if w != company_name_lower and w not in ["www", "com"]]
        top_words = Counter(filtered_words).most_common(11)[1:]
        top_words_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])

        plt.figure(figsize=(8, 4))
        plt.bar(top_words_df["Word"], top_words_df["Frequency"], color='orange')
        plt.title(f"{company_name} - Top 10 Words (Filtered)")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{company_output}/top_words.png")
        plt.close()

        # 4. Sample Negative Reviews
        negative_reviews = df[df["Rating"] == 1]["Review Text"].head(5)
        with open(f"{company_output}/negative_reviews.txt", "w", encoding="utf-8") as f:
            f.write(f"Sample Negative Reviews (Rating = 1)\n\n")
            for review in negative_reviews:
                f.write(f"- {review}\n\n")

        # 5. Review Trend Over Time
        if "Review Date" in df.columns:
            df["Review Date"] = pd.to_datetime(df["Review Date"], errors='coerce')
            date_counts = df["Review Date"].value_counts().sort_index()
            plt.figure(figsize=(8, 4))
            date_counts.plot(kind='line', marker='o', color='green')
            plt.title(f"{company_name} - Review Volume Over Time")
            plt.xlabel("Date")
            plt.ylabel("Reviews")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{company_output}/review_volume_trend.png")
            plt.close()

            # 6. Sentiment Trend Over Time
            sentiment_by_date = df.groupby("Review Date")["Sentiment Score"].mean()
            plt.figure(figsize=(8, 4))
            sentiment_by_date.plot(kind='line', color='purple', marker='x')
            plt.title(f"{company_name} - Average Sentiment Over Time")
            plt.xlabel("Date")
            plt.ylabel("Avg Sentiment Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{company_output}/sentiment_trend.png")
            plt.close()

        # 7. Longest & Shortest Reviews
        df["Text Length"] = df["Review Text"].astype(str).apply(len)
        longest_reviews = df.sort_values(by="Text Length", ascending=False)["Review Text"].head(3)
        shortest_reviews = df.sort_values(by="Text Length", ascending=True)["Review Text"].head(3)
        with open(f"{company_output}/extreme_reviews.txt", "w", encoding="utf-8") as f:
            f.write("Top 3 Longest Reviews:\n\n")
            for review in longest_reviews:
                f.write(f"- {review}\n\n")
            f.write("\nTop 3 Shortest Reviews:\n\n")
            for review in shortest_reviews:
                f.write(f"- {review}\n\n")

        # 8. Flagged Keywords
        flagged_reviews = []
        for text in df["Review Text"]:
            if any(keyword in str(text).lower() for keyword in KEYWORDS_FLAG):
                flagged_reviews.append(text)

        with open(f"{company_output}/flagged_keywords_reviews.txt", "w", encoding="utf-8") as f:
            f.write("Reviews Containing Flagged Keywords:\n\n")
            for review in flagged_reviews[:10]:
                f.write(f"- {review}\n\n")

        # 9. Sentiment/Rating Mismatch
        mismatch_reviews = df[((df["Rating"] >= 4) & (df["Sentiment"] == "Negative")) |
                              ((df["Rating"] <= 2) & (df["Sentiment"] == "Positive"))]
        with open(f"{company_output}/sentiment_rating_mismatch.txt", "w", encoding="utf-8") as f:
            f.write("Potential Mismatches (e.g. Rating 5 but sentiment Negative):\n\n")
            for review in mismatch_reviews["Review Text"].head(5):
                f.write(f"- {review}\n\n")

print("✅ Advanced EDA complete! All insights saved in company folders under eda_output/")
