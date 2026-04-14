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
st.set_page_config(page_title="HEIP | NGO Health Equity OS", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    .main {background-color: #fdfdfd;}
    .big-header {background-color: #334155; color: white; padding: 30px; border-radius: 10px; font-size: 38px; font-weight: 800; text-align: center; margin-bottom: 25px;}
    .section-header {background-color: #ccfbf1; color: #0f172a; padding: 12px 20px; border-radius: 8px; font-size: 20px; font-weight: 700; margin-top: 20px; border-left: 8px solid #2dd4bf;}
    .mission-box {background-color: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

NGO_PALETTE = px.colors.qualitative.Safe 

# --- 2. MULTI-PART DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # Load Patients
    p_path = data_dir / "patients.csv"
    if not p_path.exists(): return pd.DataFrame()
    p = pd.read_csv(p_path)
    
    # Financial Cleaning & GAP Calculations (Feedback Implementation)
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    # Specific Insurance Metrics requested
    p['INSURANCE_COVERAGE_AMT'] = p['HEALTHCARE_COVERAGE']
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['OUT_OF_POCKET_GAP'] = (p['HEALTHCARE_EXPENSES'] - p['HEALTHCARE_COVERAGE']).clip(lower=0)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # Load All Encounter Parts
    e_files = list(data_dir.glob("encounters*.csv"))
    if not e_files: return pd.DataFrame()
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    
    # Year Extraction
    e['START'] = pd.to_datetime(e['START'], errors='coerce')
    e = e.dropna(subset=['START'])
    e['YEAR'] = e['START'].dt.year 
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. AUTO-ML TOURNAMENT ENGINE ---
@st.cache_resource
def run_auto_ml(df_json, target):
    df_subset = pd.read_json(df_json)
    yearly = df_subset.groupby('YEAR')[target].mean().reset_index()
    if len(yearly) < 2: return None, None, 0, 0
    
    X, y = yearly[['YEAR']].values, yearly[target].values
    models = {
        "Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=20, learning_rate=0.1, random_state=42)
    }
    
    best_name, best_rmse, best_r2, best_proj = "", float('inf'), 0, None
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        if rmse <= best_rmse:
            best_rmse, best_r2, best_name = rmse, r2, name
            future_x = np.array(range(int(yearly['YEAR'].max()) + 1, 2031)).reshape(-1, 1)
            future_y = model.predict(future_x)
            best_proj = pd.concat([yearly.assign(Status='Actual'), 
                                   pd.DataFrame({'YEAR': future_x.flatten(), target: future_y, 'Status': 'Projected'})], ignore_index=True)
            
    return best_proj, best_name, best_rmse, best_r2

# --- 4. MAIN APP ---
try:
    df = load_and_prep_data()
    
    if not df.empty:
        st.sidebar.markdown("## 🏥 Strategic Filters")
        sel_race = st.sidebar.multiselect("Race", sorted(df['RACE'].unique()), df['RACE'].unique())
        sel_gender = st.sidebar.multiselect("Gender", sorted(df['GENDER'].unique()), df['GENDER'].unique())
        f_df = df[(df['RACE'].isin(sel_race)) & (df['GENDER'].isin(sel_gender))]
        
        page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "ML Predictive Grid"])

        if page == "Overview":
            st.markdown('<div class="big-header">Health Equity Insights Platform</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">App Summary & Mission</div>', unsafe_allow_html=True)
            st.markdown('<div class="mission-box">Identifying <strong>Vertical Equity Gaps</strong> by measuring the disparity between clinical costs and insurance coverage.</div>', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Avg Healthcare Exp.", f"${df['HEALTHCARE_EXPENSES'].mean():,.0f}")
            with c2: st.metric("Avg Coverage Gap", f"${df['OUT_OF_POCKET_GAP'].mean():,.0f}", delta_color="inverse")
            with c3: st.metric("Avg Coverage %", f"{df['INSURANCE_COVERAGE_PCT'].mean():.1f}%")

        elif page == "Interactive Map":
            st.markdown('<div class="big-header">California Regional Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Insurance & Expenditure Map</div>', unsafe_allow_html=True)
            
            m_stats = f_df.groupby('COUNTY').agg({
                'TOTAL_CLAIM_COST':'mean',
                'INSURANCE_COVERAGE_AMT':'mean',
                'INSURANCE_COVERAGE_PCT':'mean',
                'LAT':'mean', 'LON':'mean'
            }).reset_index()

            # Feedback: Tooltip now shows both Amt and Pct
            fig = px.scatter_mapbox(m_stats, lat="LAT", lon="LON", color="INSURANCE_COVERAGE_PCT", size="TOTAL_CLAIM_COST",
                                    hover_name="COUNTY", 
                                    hover_data={
                                        'LAT':False, 'LON':False, 
                                        'INSURANCE_COVERAGE_AMT':':,.0f', 
                                        'INSURANCE_COVERAGE_PCT':':.1f%',
                                        'TOTAL_CLAIM_COST':':,.0f'
                                    },
                                    color_continuous_scale="Viridis", zoom=5, mapbox_style="carto-positron")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-header">Deep-Dive: Insurance Metrics by Demographic</div>', unsafe_allow_html=True)
            sel_county = st.selectbox("Select County:", sorted(df['COUNTY'].unique()))
            c_df = df[df['COUNTY'] == sel_county]
            
            x_ax = st.selectbox("X-Axis (Demographic):", ['GENDER', 'RACE', 'INCOME_TIER'])
            y_ax = st.selectbox("Insurance Metric (Y-Axis):", ['INSURANCE_COVERAGE_AMT', 'INSURANCE_COVERAGE_PCT', 'OUT_OF_POCKET_GAP'])
            st.plotly_chart(px.bar(c_df.groupby(x_ax)[y_ax].mean().reset_index(), x=x_ax, y=y_ax, color=x_ax, color_discrete_sequence=NGO_PALETTE), use_container_width=True)

        elif page == "ML Predictive Grid":
            st.markdown('<div class="big-header">Intersectional Forecast Grid</div>', unsafe_allow_html=True)
            target = st.selectbox("Forecast Target:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT', 'OUT_OF_POCKET_GAP'])
            
            grid_data = []
            for r in sorted(f_df['RACE'].unique()):
                for g in sorted(f_df['GENDER'].unique()):
                    subset = f_df[(f_df['RACE'] == r) & (f_df['GENDER'] == g)]
                    res, winner, rmse, r2 = run_auto_ml(subset.to_json(), target)
                    if res is not None:
                        res['RACE'], res['GENDER'] = r, g
                        grid_data.append(res)
            
            if grid_data:
                fig = px.line(pd.concat(grid_data), x="YEAR", y=target, color="Status", facet_row="RACE", facet_col="GENDER", 
                             markers=True, color_discrete_map={'Actual':'#94a3b8','Projected':'#fb7185'})
                fig.update_layout(height=1000)
                st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Initialization Error: {e}")
