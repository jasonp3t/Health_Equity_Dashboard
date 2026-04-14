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
st.set_page_config(page_title="HEIP | Health Equity OS", layout="wide", page_icon="🏥")

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
    
    # 1. Load Patients
    p_path = data_dir / "patients.csv"
    if not p_path.exists():
        st.error(f"🚨 Missing patients.csv in {data_dir}")
        return pd.DataFrame()
    p = pd.read_csv(p_path)
    
    # Clean Dollars
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # 2. Load Encounter Parts (Combines part_1, part_2, etc.)
    e_files = list(data_dir.glob("encounters*.csv"))
    if not e_files:
        st.error(f"🚨 No encounter files found in {data_dir}")
        return pd.DataFrame()
    
    st.sidebar.success(f"📂 Loaded {len(e_files)} encounter data parts.")
    e_list = [pd.read_csv(f) for f in e_files if f.stat().st_size > 0]
    e = pd.concat(e_list, ignore_index=True)
    
    # Convert Dates
    e['START'] = pd.to_datetime(e['START'], errors='coerce')
    e = e.dropna(subset=['START'])
    e['YEAR'] = e['START'].dt.year 
    
    # 3. Final Merge
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. AUTO-ML TOURNAMENT ENGINE ---
def run_auto_ml(df, target):
    yearly = df.groupby('YEAR')[target].mean().reset_index()
    if len(yearly) < 2: return None, None, 0, 0
    
    X, y = yearly[['YEAR']].values, yearly[target].values
    
    # Tournament selection based on data density
    models = {"Linear Regression": LinearRegression()}
    if len(yearly) >= 4:
        models.update({
            "Lasso (L1)": Lasso(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=30, learning_rate=0.1, random_state=42)
        })
    
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
            
            best_proj = pd.concat([
                yearly.assign(Status='Actual'), 
                pd.DataFrame({'YEAR': future_x.flatten(), target: future_y, 'Status': 'Projected'})
            ], ignore_index=True)
            
    return best_proj, best_name, best_rmse, best_r2

# --- 4. MAIN APP ---
try:
    df = load_and_prep_data()
    
    if not df.empty:
        st.sidebar.markdown("---")
        sel_race = st.sidebar.multiselect("Filter by Race", sorted(df['RACE'].unique()), df['RACE'].unique())
        sel_gender = st.sidebar.multiselect("Filter by Gender", sorted(df['GENDER'].unique()), df['GENDER'].unique())
        
        f_df = df[(df['RACE'].isin(sel_race)) & (df['GENDER'].isin(sel_gender))]
        
        page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "ML Predictive Grid"])

        if page == "Overview":
            st.markdown('<div class="big-header">Health Equity Insights Platform</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Project Summary</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="mission-box">
                Identifying <strong>Vertical Equity Gaps</strong> by processing multiple encounter streams 
                and applying an Automated ML Tournament to find the best intersectional models.
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Records Processed", len(df))
                st.metric("Patient Population", df['PATIENT'].nunique())
            with c2:
                st.write("**Data Completeness Check:**")
                st.write(f"Years: {sorted(df['YEAR'].unique())}")

        elif page == "Interactive Map":
            st.markdown('<div class="big-header">Regional Expenditure Map</div>', unsafe_allow_html=True)
            m_stats = f_df.groupby('COUNTY').agg({'TOTAL_CLAIM_COST':'mean','LAT':'mean','LON':'mean'}).reset_index()
            fig = px.scatter_mapbox(m_stats, lat="LAT", lon="LON", color="TOTAL_CLAIM_COST", size="TOTAL_CLAIM_COST",
                                    hover_name="COUNTY", color_continuous_scale="Teal", zoom=5, mapbox_style="carto-positron")
            st.plotly_chart(fig, use_container_width=True)

        elif page == "Population Comparison":
            st.markdown('<div class="big-header">Comparison Analysis</div>', unsafe_allow_html=True)
            metric = st.selectbox("Metric:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(px.bar(f_df.groupby('RACE')[metric].mean().reset_index(), x='RACE', y=metric, color='RACE', color_discrete_sequence=NGO_PALETTE), use_container_width=True, key="c1")
            with c2: st.plotly_chart(px.bar(f_df.groupby('GENDER')[metric].mean().reset_index(), x='GENDER', y=metric, color='GENDER', color_discrete_sequence=NGO_PALETTE), use_container_width=True, key="c2")

        elif page == "ML Predictive Grid":
            st.markdown('<div class="big-header">Automated ML Forecast Grid</div>', unsafe_allow_html=True)
            target = st.selectbox("Select Target Variable:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
            
            grid_data = []
            for r in sorted(f_df['RACE'].unique()):
                for g in sorted(f_df['GENDER'].unique()):
                    subset = f_df[(f_df['RACE'] == r) & (f_df['GENDER'] == g)]
                    res, winner, rmse, r2 = run_auto_ml(subset, target)
                    if res is not None:
                        res['RACE'], res['GENDER'] = r, g
                        grid_data.append(res)
            
            if grid_data:
                final_grid = pd.concat(grid_data)
                fig = px.line(final_grid, x="YEAR", y=target, color="Status", facet_row="RACE", facet_col="GENDER", 
                             markers=True, color_discrete_map={'Actual':'#94a3b8','Projected':'#fb7185'})
                fig.update_layout(height=1000)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Insufficient data to trend. Check encounter year ranges.")

except Exception as e:
    st.error(f"🚨 Initialization Error: {e}")


       
   
