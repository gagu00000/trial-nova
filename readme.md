# NovaMart Marketing Analytics Dashboard

An interactive **Streamlit dashboard** built for NovaMart's marketing analytics team to visualize:
- Campaign performance
- Customer insights
- Product sales analytics
- Geographic distribution
- Marketing funnel & attribution
- ML model evaluation

This dashboard supports 20+ visualizations and is designed for board-level presentations with a premium color theme.

---

## 🚀 Features
- Multi‑page Streamlit navigation
- 11 interconnected datasets
- KPI cards
- Time‑series charts (line, area)
- Comparison charts (bar, grouped bar, stacked bar)
- Distribution charts (histogram, box, violin)
- Relationship charts (scatter, bubble, heatmap)
- Maps (choropleth & bubble)
- Funnel, donut, treemap, sunburst
- ML model charts: ROC, Confusion Matrix, Learning Curve, Feature Importance
- Caching for fast performance
- Responsive layout

---

## 📂 Project Structure
```
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
    |-- campaign_performance.csv
    |-- customer_data.csv
    |-- product_sales.csv
    |-- lead_scoring_results.csv
    |-- feature_importance.csv
    |-- learning_curve.csv
    |-- geographic_data.csv
    |-- channel_attribution.csv
    |-- funnel_data.csv
    |-- customer_journey.csv
    |-- correlation_matrix.csv
```

> ⚠️ Important: Create a folder named **data/** in your GitHub repo and upload all 11 CSV files into it.

---

## 🛠️ Installation
Clone your GitHub repo:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run Streamlit app:
```bash
streamlit run app.py
```

---

## 📡 Deploy on Streamlit Cloud
1. Push this repository to GitHub
2. Go to https://streamlit.io/cloud
3. Select **New app** → choose your repo
4. Set:
   - Main file: `app.py`
   - Python version: 3.10+
   - Dependencies: automatically taken from `requirements.txt`
5. Deploy

When deployed, upload your dataset folder **data/** directly into your GitHub repo for Streamlit Cloud to read.

---

## 📊 Usage
- Use the left sidebar to navigate across pages
- Filter charts using dropdowns & sliders
- Hover for tooltips and detailed metrics

---

## 📞 Support
If you need help configuring the dashboard, extending visualizations, or preparing the insights report or board presentation, feel free to reach out.

---

Enjoy exploring NovaMart’s Marketing Analytics Dashboard! 🚀

