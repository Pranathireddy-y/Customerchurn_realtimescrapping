import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json
import shutil

CLEANED_FOLDER = "data/cleaned"
OUTPUT_FOLDER = "model_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Clear old outputs
for file in os.listdir(OUTPUT_FOLDER):
    os.remove(os.path.join(OUTPUT_FOLDER, file))

all_results = []

for file in os.listdir(CLEANED_FOLDER):
    if not file.endswith("_cleaned.csv"):
        continue

    company = file.replace("_cleaned.csv", "")
    df = pd.read_csv(os.path.join(CLEANED_FOLDER, file))
    df = df.dropna(subset=["Review Text", "Rating", "Product Category"])
    df["Rating"] = pd.to_numeric(df["Rating"], errors='coerce')
    df = df.dropna(subset=["Rating"])

    df["Churn"] = (df["Rating"] <= 2).astype(int)

    for category in df["Product Category"].unique():
        subset = df[df["Product Category"] == category]
        if len(subset["Churn"].unique()) < 2 or len(subset) < 10:
            print(f"⏭️ Skipping {company} - {category} (not enough reviews)")
            continue

        X = subset["Review Text"]
        y = subset["Churn"]

        tfidf = TfidfVectorizer(max_features=1000)
        X_tfidf = tfidf.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        for model_name, model in [("LogisticRegression", LogisticRegression(max_iter=1000)), ("RandomForest", RandomForestClassifier())]:
            print(f"🔄 Training {model_name} for {company} - {category}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            f1 = report['1']['f1-score']
            acc = report['accuracy']
            prec = report['1']['precision']
            rec = report['1']['recall']
            churn_pct = round(subset["Churn"].mean() * 100, 2)

            all_results.append({
                "Company": company,
                "Category": category,
                "Model": model_name,
                "Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1 Score": round(f1, 3),
                "Churn %": churn_pct
            })

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
            plt.title(f"{company} - {category} - {model_name}", fontsize=11)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"cm_{company}_{category}_{model_name}.png"))
            plt.close()

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_FOLDER, "model_comparison_metrics.csv"), index=False)

# Save churn predictions as JSON
churn_json = dict()
for row in all_results:
    company = row["Company"]
    category = row["Category"]
    churn = row["Churn %"]
    if company not in churn_json:
        churn_json[company] = dict()
    churn_json[company][category] = churn

with open(os.path.join(OUTPUT_FOLDER, "churn_predictions.json"), "w") as f:
    json.dump(churn_json, f, indent=2)

# --------- SAVE VISUAL INSIGHTS ---------

metrics_df = pd.read_csv(os.path.join(OUTPUT_FOLDER, "model_comparison_metrics.csv"))

# Category-wise churn charts
companies = metrics_df["Company"].unique()
for company in companies:
    company_df = metrics_df[metrics_df["Company"] == company]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=company_df, x="Category", y="Churn %", hue="Model")
    plt.title(f"{company.replace('_reviews', '').capitalize()} - Churn Percentage by Category", fontsize=12)
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{company.lower()}_churn_by_category.png"))
    plt.close()

# Average Accuracy and F1 Score
avg_scores = metrics_df.groupby("Model")[["Accuracy", "F1 Score"]].mean().reset_index()

if not avg_scores.empty:
    plt.figure(figsize=(10, 5))
    x = range(len(avg_scores))
    plt.bar([i - 0.15 for i in x], avg_scores["Accuracy"], width=0.3, label="Accuracy")
    plt.bar([i + 0.15 for i in x], avg_scores["F1 Score"], width=0.3, label="F1 Score")
    plt.xticks(x, avg_scores["Model"])
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Average Accuracy & F1 Score per Model", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "avg_model_accuracy_f1.png"))
    plt.close()

# Top 10 churn categories
top_churn = metrics_df.sort_values(by="Churn %", ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_churn, x="Churn %", y="Company", hue="Category", palette="Reds_r")
plt.title("Top 10 Churn Categories", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "top_churn_categories.png"))
plt.close()

# Top 10 models by F1 score
top_f1 = metrics_df.sort_values(by="F1 Score", ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_f1, x="F1 Score", y="Company", hue="Category", palette="Greens")
plt.title("Top 10 Best Performing Models", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "top_f1_models.png"))
plt.close()

print("✅ All models trained. Old files cleaned. Fresh results and clean charts saved to model_output/")