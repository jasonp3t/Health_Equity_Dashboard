import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Equity Insights Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --mint:      #e0f5f1;
    --teal:      #1a9e8f;
    --teal-dark: #0d7a6e;
    --teal-mid:  #2bbfae;
    --amber:     #f59e0b;
    --coral:     #ef4444;
    --navy:      #0f3460;
    --text-main: #1e293b;
    --card-bg:   #ffffff;
    --bg-main:   #f0faf8;
  }
  .stApp { background: var(--bg-main) !important; }
  .hero-header {
    background: linear-gradient(135deg, #0d7a6e 0%, #1a9e8f 50%, #2bbfae 100%);
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(13,122,110,.25);
    color: white;
  }
  .section-header {
    background: var(--mint);
    border-left: 5px solid var(--teal);
    border-radius: 8px;
    padding: .75rem 1.2rem;
    margin: 1.4rem 0 .9rem;
    color: var(--teal-dark);
    font-size: 1.18rem;
    font-weight: 700;
  }
  .metric-card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.3rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,.07);
    border-top: 4px solid var(--teal);
  }
  .metric-value { font-size: 2rem; font-weight: 800; color: var(--teal-dark); }
  .metric-label { font-size: .82rem; color: #64748b; font-weight: 600; }
  .equity-gap-box {
    background: #fef2f2;
    border-left: 4px solid var(--coral);
    padding: 1rem;
    border-radius: 8px;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d7a6e 0%, #0f3460 100%) !important;
  }
  [data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data Generator ──────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=2272):
    np.random.seed(42)
    races = ['White','Black','Asian','Hawaiian','Native','Other']
    race_w = np.array([1640,134,370,33,35,60])/2272
    race = np.random.choice(races, n, p=race_w)
    gender = np.random.choice(['Male','Female'], n, p=[0.479, 0.521])
    age = np.random.randint(0, 111, n)
    income_means = {'White':72000,'Asian':68000,'Black':38000,'Hawaiian':45000,'Native':32000,'Other':50000}
    income = np.array([np.random.normal(income_means[r], 15000) for r in race]).clip(10000,250000)
    insurance_means = {'White':85,'Asian':80,'Black':62,'Hawaiian':68,'Native':55,'Other':70}
    insurance_pct = np.array([np.random.normal(insurance_means[r], 10) for r in race]).clip(10,100)
    
    total_claim = np.array([129 * (1 + np.random.exponential(8)) + max(0, (50000 - income[i]) * 0.002) for i, r in enumerate(race)])
    
    ca_cities = {'Los Angeles':(34.05,-118.24,272), 'San Francisco':(37.77,-122.41,180), 'San Diego':(32.71,-117.16,145)}
    city = np.random.choice(list(ca_cities.keys()), n)
    lat = np.array([ca_cities[c][0] + np.random.normal(0,.1) for c in city])
    lon = np.array([ca_cities[c][1] + np.random.normal(0,.1) for c in city])
    enc_year = np.random.choice(range(2015,2024), n)
    income_band = pd.cut(income, bins=[0,25000,50000,75000,100000,1e9], labels=['<$25k','$25-50k','$50-75k','$75-100k','>$100k'])

    return pd.DataFrame({'race':race,'gender':gender,'age':age,'city':city,'lat':lat,'lon':lon,'income':income,'income_band':income_band,'insurance_pct':insurance_pct,'total_claim_cost':total_claim,'encounter_year':enc_year})

df = generate_data()

RACE_COLORS = {'White':'#2196F3','Black':'#F44336','Asian':'#4CAF50','Hawaiian':'#FF9800','Native':'#9C27B0','Other':'#607D8B'}
def section(title): st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 HEIP Navigation")
    page = st.radio("", ["📊 Dashboard", "🤖 Predictive Forecasting", "📬 Contact"])

st.markdown('<div class="hero-header"><h1>🏥 Health Equity Insights Platform</h1><p>Predictive Healthcare Analytics & Equity Gap Identification</p></div>', unsafe_allow_html=True)

# ─── Dashboard ───────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    section("📌 Key Metrics")
    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Patients</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">${df.total_claim_cost.mean():,.0f}</div><div class="metric-label">Avg Claim Cost</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">${df.income.mean():,.0f}</div><div class="metric-label">Avg Income</div></div>', unsafe_allow_html=True)

# ─── Predictive Forecasting (The Improved Section) ───────────────────────────
elif page == "🤖 Predictive Forecasting":
    section("🤖 2030 Horizon: Predictive Cost Analysis")

    @st.cache_data
    def train_optimized_model(df):
        dfc = df.copy()
        le_r, le_g, le_i = LabelEncoder(), LabelEncoder(), LabelEncoder()
        dfc['race_enc'] = le_r.fit_transform(dfc['race'])
        dfc['gender_enc'] = le_g.fit_transform(dfc['gender'])
        dfc['income_enc'] = le_i.fit_transform(dfc['income_band'].astype(str))
        
        feats = ['race_enc', 'gender_enc', 'age', 'income', 'insurance_pct', 'encounter_year', 'income_enc']
        X, y = dfc[feats], dfc['total_claim_cost']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)
        
        # Improvement: Using 'absolute_error' loss to directly minimize MAE
        model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, loss='absolute_error', random_state=42)
        model.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_te, model.predict(X_te))
        return model, mae, feats, le_r, le_g, le_i

    model, mae, feats, le_r, le_g, le_i = train_optimized_model(df)

    # Display Accuracy
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">${mae:,.2f}</div><div class="metric-label">Mean Absolute Error (Lowered)</div></div>', unsafe_allow_html=True)
    c2.info("Model optimized using Gradient Boosting with Absolute Error Loss for higher precision in cost estimation.")

    # 2030 Projection Engine
    section("📈 Projected Trends to 2030")
    col_a, col_b = st.columns(2)
    p_race = col_a.selectbox("Select Race for Projection:", df.race.unique())
    p_gender = col_b.selectbox("Select Gender for Projection:", df.gender.unique())

    future_years = list(range(2024, 2031))
    base_stats = df[(df.race == p_race) & (df.gender == p_gender)]
    
    # Generate future scenario
    future_df = pd.DataFrame([{
        'race_enc': le_r.transform([p_race])[0],
        'gender_enc': le_g.transform([p_gender])[0],
        'age': base_stats['age'].mean(),
        'income': base_stats['income'].mean(),
        'insurance_pct': base_stats['insurance_pct'].mean(),
        'encounter_year': yr,
        'income_enc': le_i.transform([str(base_stats['income_band'].mode()[0])])[0]
    } for yr in future_years])

    future_preds = model.predict(future_df[feats])
    hist_yr = base_stats.groupby('encounter_year')['total_claim_cost'].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_yr.encounter_year, y=hist_yr.total_claim_cost, name='Historical', line=dict(color='#1a9e8f', width=4)))
    fig.add_trace(go.Scatter(x=future_years, y=future_preds, name='2030 Forecast', line=dict(color='#ef4444', width=4, dash='dash')))
    fig.update_layout(title=f"Predicted Claim Cost for {p_race} Population", xaxis_title="Year", yaxis_title="Cost ($)", plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

# ─── Contact ──────────────────────────────────────────────────────────────────
elif page == "📬 Contact":
    section("📬 Partner Outreach")
    st.text_input("Name")
    st.text_area("Message")
    if st.button("Submit"): st.balloons()
