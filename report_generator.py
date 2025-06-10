import os
import json
from datetime import datetime
from glob import glob
import pandas as pd
from fpdf import FPDF

# Define paths
OUTPUT_FOLDER = "model_output"
PDF_PATH = "Churn_Analysis_Report.pdf"

# Load metrics and churn data
metrics_csv = os.path.join(OUTPUT_FOLDER, "model_comparison_metrics.csv")
churn_json = os.path.join(OUTPUT_FOLDER, "churn_predictions.json")

if not os.path.exists(metrics_csv) or not os.path.exists(churn_json):
    raise FileNotFoundError("Required files not found in model_output/. Please run train.py first.")

metrics_df = pd.read_csv(metrics_csv)
with open(churn_json, "r") as f:
    churn_data = json.load(f)

# Count scraped data
total_reviews = metrics_df.shape[0]

# Get date and summary insights
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
most_churn_row = metrics_df.sort_values(by="Churn %", ascending=False).iloc[0]
best_model_row = metrics_df.sort_values(by="F1 Score", ascending=False).iloc[0]

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Cover Page
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Customer Churn Analysis - Capstone Report", ln=True)
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, f"Date: {now}", ln=True)
pdf.ln(5)
pdf.multi_cell(0, 10, f'''
This report summarizes the customer churn analysis pipeline run on latest scraped reviews.
The scraping covered multiple companies from Trustpilot and churn models were trained on real-time data.

- Total Models Trained: {total_reviews}
- Most At-Risk Category: {most_churn_row['Category']} on {most_churn_row['Company']} ({most_churn_row['Churn %']}% churn)
- Best Performing Model: {best_model_row['Model']} on {best_model_row['Company']} - {best_model_row['Category']} (F1 Score: {best_model_row['F1 Score']})
''')

# Add Churn Summary Table from JSON
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Churn Prediction Summary by Company & Category", ln=True)
pdf.set_font("Arial", size=10)

pdf.set_fill_color(200, 220, 255)
pdf.cell(50, 8, "Company", 1, 0, 'C', True)
pdf.cell(60, 8, "Category", 1, 0, 'C', True)
pdf.cell(30, 8, "Churn %", 1, 1, 'C', True)

for company, categories in churn_data.items():
    for category, churn in categories.items():
        pdf.cell(50, 8, company.replace("_reviews", ""), 1)
        pdf.cell(60, 8, category, 1)
        pdf.cell(30, 8, f"{churn}%", 1, 1)

# Add Model Comparison Table from CSV
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Model Performance Metrics", ln=True)
pdf.set_font("Arial", size=8)

headers = ["Company", "Category", "Model", "Accuracy", "Precision", "Recall", "F1 Score", "Churn %"]
for h in headers:
    pdf.cell(25, 6, h, 1, 0, 'C', True)
pdf.ln()

for _, row in metrics_df.iterrows():
    for h in headers:
        pdf.cell(25, 6, str(row[h]), 1)
    pdf.ln()

# Add PNG Charts
image_files = sorted(glob(os.path.join(OUTPUT_FOLDER, "*.png")))
for img_path in image_files:
    pdf.add_page()
    pdf.image(img_path, w=180)

pdf.output(PDF_PATH)
print(f"PDF report generated: {PDF_PATH}")
