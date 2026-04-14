import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Machine Learning Suite
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. PAGE CONFIG & NGO STYLING ---
st.set_page_config(page_title="HEIP | NGO Health Equity Platform", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    .main {background-color: #fdfdfd;}
    .big-header {
        background-color: #334155; 
        color: white; 
        padding: 30px; 
        border-radius: 10px; 
        font-size: 42px; 
        font-weight: 800; 
        text-align: center;
        margin-bottom: 25px;
    }
    .section-header {
        background-color: #ccfbf1; 
        color: #0f172a; 
        padding: 12px 20px; 
        border-radius: 8px; 
        font-size: 22px; 
        font-weight: 700; 
        margin-top: 25px; 
        margin-bottom: 15px; 
        border-left: 8px solid #2dd4bf;
    }
    .mission-box {
        background-color: #f8fafc; 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #e2e8f0; 
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

NGO_PALETTE = px.colors.qualitative.Safe 

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # Load Patients
    p = pd.read_csv(data_dir / "patients.csv")
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # Load Encounters
    e_files = list(data_dir.glob("encounters*.csv"))
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    e['YEAR'] = pd.to_datetime(e['START']).dt.year 
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. AUTO-ML TOURNAMENT ENGINE ---
def run_auto_ml(df, target):
    yearly = df.groupby('YEAR')[target].mean().reset_index()
    if len(yearly) < 4: return None, None, 0, 0
    
    X, y = yearly[['YEAR']].values, yearly[target].values
    
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso (L1)": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    }
    
    best_name, best_rmse, best_r2, best_proj = "", float('inf'), 0, None
    
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        
        if rmse < best_rmse:
            best_rmse, best_r2, best_name = rmse, r2, name
            future_x = np.array(range(yearly['YEAR'].max() + 1, 2031)).reshape(-1, 1)
            future_y = model.predict(future_x)
            best_proj = pd.concat([yearly.assign(Status='Actual'), 
                                   pd.DataFrame({'YEAR': future_x.flatten(), target: future_y, 'Status': 'Projected'})])
            
    return best_proj, best_name, best_rmse, best_r2

# --- 4. MAIN NAVIGATION ---
try:
    raw_df = load_and_prep_data()
    
    st.sidebar.markdown("## 🏥 Strategic Filters")
    sel_genders = st.sidebar.multiselect("Gender", sorted(raw_df['GENDER'].unique()), raw_df['GENDER'].unique())
    sel_races = st.sidebar.multiselect("Race", sorted(raw_df['RACE'].unique()), raw_df['RACE'].unique())
    sel_income = st.sidebar.multiselect("Income Tier", ['Low', 'Middle', 'High'], ['Low', 'Middle', 'High'])
    
    df = raw_df[(raw_df['GENDER'].isin(sel_genders)) & (raw_df['RACE'].isin(sel_races)) & (raw_df['INCOME_TIER'].isin(sel_income))]
    
    page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Comparative Analysis", "ML Predictive Grid"])

    if page == "Overview":
        st.markdown('<div class="big-header">Health Equity Insights Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">App Summary & Mission</div>', unsafe_allow_html=True)
        st.markdown('<div class="mission-box">Analyzing <strong>Vertical Equity Gaps</strong> using intersectional data and automated ML selection across 5 distinct models.</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🎯 Key Stakeholders")
            st.write("* NGOs & Policy Makers\n* Public Health Officials")
        with c2:
            st.subheader("📊 Primary Finding")
            st.info("💡 Insight: Low-Income groups face higher cost burdens relative to insurance coverage growth.")

        st.markdown('<div class="section-header">Contact NGO Analysts</div>', unsafe_allow_html=True)
        with st.form("feedback"):
            st.text_input("Full Name"); st.text_input("Email"); st.text_area("Observations")
            if st.form_submit_button("Submit"): st.success("Feedback Logged!")

    elif page == "Interactive Map":
        st.markdown('<div class="big-header">Regional Hotspot Map</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">California Expenditure Bubble Map</div>', unsafe_allow_html=True)
        
        map_stats = df.groupby('COUNTY').agg({'TOTAL_CLAIM_COST': 'mean', 'INCOME': 'mean', 'INSURANCE_COVERAGE_PCT': 'mean', 'LAT': 'mean', 'LON': 'mean'}).reset_index()
        fig = px.scatter_mapbox(map_stats, lat="LAT", lon="LON", color="TOTAL_CLAIM_COST", size="TOTAL_CLAIM_COST",
                                hover_name="COUNTY", hover_data={'LAT':False, 'LON':False, 'INCOME':':,.0f', 'INSURANCE_COVERAGE_PCT':':.1f%'},
                                color_continuous_scale="Teal", size_max=25, zoom=5, mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True, key="main_map")

    elif page == "Comparative Analysis":
        st.markdown('<div class="big-header">Intersectional Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Demographic Equity Metrics</div>', unsafe_allow_html=True)
        metric = st.selectbox("Select Metric:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        c1, c2 = st.columns(2)
        with c1:
            demo_a = st.selectbox("Group A:", ['GENDER', 'RACE', 'INCOME_TIER'], key="a")
            st.plotly_chart(px.bar(df.groupby(demo_a)[metric].mean().reset_index(), x=demo_a, y=metric, color=demo_a, color_discrete_sequence=NGO_PALETTE), use_container_width=True, key="c1")
        with c2:
            demo_b = st.selectbox("Group B:", ['INCOME_TIER', 'RACE', 'GENDER'], key="b")
            st.plotly_chart(px.bar(df.groupby(demo_b)[metric].mean().reset_index(), x=demo_b, y=metric, color=demo_b, color_discrete_sequence=NGO_PALETTE), use_container_width=True, key="c2")

    elif page == "ML Predictive Grid":
        st.markdown('<div class="big-header">Automated ML Forecast Grid</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Optimized 2030 Projections (Automated Tournament)</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1: row_f = st.selectbox("Vertical Axis (Rows):", ['RACE', 'GENDER'], index=0)
        with c2: col_f = st.selectbox("Horizontal Axis (Columns):", ['GENDER', 'INCOME_TIER'], index=1)
        with c3: target = st.selectbox("Metric to Forecast:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])

        grid_results, leaderboard = [], []
        for r in sorted(df[row_f].unique()):
            for c in sorted(df[col_f].unique()):
                subset = df[(df[row_f] == r) & (df[col_f] == c)]
                res, winner, rmse, r2 = run_auto_ml(subset, target)
                if res is not None:
                    res[row_f], res[col_f] = r, c
                    grid_results.append(res)
                    leaderboard.append({"Intersection": f"{r} | {c}", "Champion Model": winner, "Accuracy (R²)": f"{r2:.4f}"})
        
        if grid_results:
            st.write("**Model Tournament Leaderboard**")
            st.dataframe(pd.DataFrame(leaderboard), use_container_width=True)
            
            fig = px.line(pd.concat(grid_results), x="YEAR", y=target, color="Status", facet_row=row_f, facet_col=col_f, 
                         markers=True, color_discrete_map={'Actual':'#94a3b8','Projected':'#fb7185'})
            fig.update_layout(height=250 * len(df[row_f].unique()))
            st.plotly_chart(fig, use_container_width=True, key="final_grid")
        else:
            st.warning("Insufficient data for ML competition.")

except Exception as e:
    st.error("🚨 System Update Required. Check data folder.")
    st.exception(e)
       

   
            
