# 🏥 Health Equity Insights Platform (HEIP)

> **Course:** Applied Analytics — Community Health Equity  
> **Team:** Raju Koppadi · Jason Fang · Thanuja Kalla  
> **Dataset:** Synthea Synthetic EHR — California (n = 2,272)

---

## 🎯 Mission

HEIP is a strategic analytical tool designed to identify **Vertical Equity Gaps** in healthcare.  
By intersecting clinical costs, personal wealth, and demographic identity, the platform empowers  
**NGOs and Public Health officials** to advocate for resource redistribution for underserved populations.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR-USERNAME/heip-app.git
cd heip-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## 📋 Pages & Features

| Page | Key Features |
|---|---|
| **📊 Dashboard** | Population KPIs, equity gap alert, race/income/insurance bar charts, Top-10 table |
| **🗺️ Interactive Map** | Tile choropleth scatter-map (avg income, avg insurance %, avg claim cost); city-level summary table; race breakdown by city; insurance vs income scatter |
| **🔍 Deep-Dive County** | City selector; custom X-axis sliders for income, claim cost & insurance distributions; yearly encounter trend charts |
| **⚖️ Intersectional Comparison** | Consistent colour legend across both plots; box plots (race × gender); grouped bar (income band × race); heatmap |
| **🤖 Predictive Forecasting** | Algorithm rationale + model comparison table; feature importance; **multi-graph grid** (choose X-axis, Y-axis & outcome toggle) |
| **📬 Contact & Feedback** | NGO contact form with urgency selector; platform feedback form with rating slider; partner NGO cards |

---

## 🔬 Predictive Algorithm — Design Decisions

We evaluated three models:

| Model | R² | Notes |
|---|---|---|
| Linear Regression | ~0.25 | Fails to capture income × race non-linear interactions |
| Random Forest | ~0.61 | Good accuracy, less interpretable |
| **Gradient Boosting ✓** | **~0.68** | Best MAE, robust to outliers, clear feature importances for NGO stakeholders |

**Features:** race, gender, age, income, insurance %, encounter year, income band  
**Target:** `total_claim_cost`

> ⚠️ The `encounter_year` variable represents the **year of the medical encounter**, not the patient's birth year.

---

## 🎨 Design System

- **Primary:** Teal `#1a9e8f` / Dark Teal `#0d7a6e`
- **Section headers:** Light mint-teal background `#e0f5f1` with teal left border
- **Hero banner:** Gradient teal header
- **Consistent colour legends:** Race uses the same hex per group across all charts; income bands are fixed colours

---

## 📁 Project Structure

```
heip-app/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## 🌐 Data Source

Synthea™ Synthetic Patient Generator — California population  
Command: `java -jar synthea.jar -s 54321 California`  
Reproducible seed: `54321`

---

## 📚 References

- Walonoski et al. (2018). *Synthea: An approach for generating synthetic EHRs.* BMC Medical Informatics.  
- WHO (2021). *Health Equity Assessment Toolkit (HEAT).*  
- National Academy of Medicine (2018). *Integrating Social Needs into Health Care Delivery.*
