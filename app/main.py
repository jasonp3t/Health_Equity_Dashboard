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
    .big-header {background-color: #334155; color: white; padding: 30px; border-radius: 10px; font-size: 38px; font-weight: 800; text-align: center; margin-bottom: 25px;}
    .section-header {background-color: #ccfbf1; color: #0f172a; padding: 12px 20px; border-radius: 8px; font-size: 20px; font-weight: 700; margin-top: 20px; border-left: 8px solid #2dd4bf;}
    .mission-box {background-color: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

NGO_PALETTE = px.colors.qualitative.Safe 

# --- 2. ROBUST DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    # Adjusted path logic to find data folder
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # 1. Load Patients
    p_path = data_dir / "patients.csv"
    if not p_path.exists():
        st.error(f"Missing file: {p_path}")
        return pd.DataFrame()
    
    p = pd.read_csv(p_path)
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # 2. Load Encounters (Handle multiple files or single file)
    e_files = list(data_dir.glob("encounters*.csv"))
    if not e_files:
        st.error("No encounters CSV files found in /data folder.")
        return pd.DataFrame()
    
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    
    # Critical: Convert date and handle missing years
    e['START'] = pd.to_datetime(e['START'], errors='coerce')
    e = e.dropna(subset=['START'])
    e['YEAR'] = e['START'].dt.year 
    
    # 3. Merge
    merged = pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')
    return merged

# --- 3. AUTO-ML TOURNAMENT ENGINE ---
def run_auto_ml(df, target):
    yearly = df.groupby('YEAR')[target].mean().reset_index()
    
    # Requirement: At least 2 points for a line, 4 for advanced ML
    if len(yearly) < 2: 
        return None, None, 0, 0
    
    X, y = yearly[['YEAR']].values, yearly[target].values
    
    # Scale model complexity based on data availability
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
            # Predict to 2030
            max_year = int(yearly['YEAR'].max())
            future_x = np.array(range(max_year + 1, 2031)).reshape(-1, 1)
            future_y = model.predict(future_x)
            
            hist = yearly.assign(Status='Actual')
            proj = pd.DataFrame({'YEAR': future_x.flatten(), target: future_y, 'Status': 'Projected'})
            best_proj = pd.concat([hist, proj], ignore_index=True)
            
    return best_proj, best_name, best_rmse, best_r2

# --- 4. MAIN APP ---
try:
    df = load_and_prep_data()
    
    if df.empty:
        st.warning("⚠️ Data Load Failed. Ensure your 'data' folder contains 'patients.csv' and 'encounters.csv'.")
    else:
        st.sidebar.markdown("## 🏥 Strategic Filters")
        sel_genders = st.sidebar.multiselect("Gender", sorted(df['GENDER'].unique()), df['GENDER'].unique())
        sel_races = st.sidebar.multiselect("Race", sorted(df['RACE'].unique()), df['RACE'].unique())
        
        filtered_df = df[(df['GENDER'].isin(sel_genders)) & (df['RACE'].isin(sel_races))]
        
        page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "ML Predictive Grid"])

        if page == "Overview":
            st.markdown('<div class="big-header">Health Equity Insights Platform</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">App Summary & Mission</div>', unsafe_allow_html=True)
            st.markdown('<div class="mission-box">Analyzing <strong>Vertical Equity Gaps</strong> using intersectional data.</div>', unsafe_allow_html=True)
            
            # Diagnostic Data Check
            with st.expander("🛠️ System Diagnostic (Check this if pages are empty)"):
                st.write(f"Total Records: {len(df)}")
                st.write(f"Years found in data: {sorted(df['YEAR'].unique())}")
                st.write(f"Counties found: {df['COUNTY'].nunique()}")

        elif page == "Interactive Map":
            st.markdown('<div class="big-header">Regional Hotspot Map</div>', unsafe_allow_html=True)
            map_stats = filtered_df.groupby('COUNTY').agg({'TOTAL_CLAIM_COST': 'mean', 'LAT': 'mean', 'LON': 'mean'}).reset_index()
            fig = px.scatter_mapbox(map_stats, lat="LAT", lon="LON", color="TOTAL_CLAIM_COST", size="TOTAL_CLAIM_COST",
                                    hover_name="COUNTY", color_continuous_scale="Teal", zoom=5, mapbox_style="carto-positron")
            st.plotly_chart(fig, use_container_width=True)

        elif page == "Population Comparison":
            st.markdown('<div class="big-header">Intersectional Comparison</div>', unsafe_allow_html=True)
            metric = st.selectbox("Select Metric:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.bar(filtered_df.groupby('RACE')[metric].mean().reset_index(), x='RACE', y=metric, color='RACE', color_discrete_sequence=NGO_PALETTE), use_container_width=True, key="p1")
            with c2:
                st.plotly_chart(px.bar(filtered_df.groupby('GENDER')[metric].mean().reset_index(), x='GENDER', y=metric, color='GENDER', color_discrete_sequence=NGO_PALETTE), use_container_width=True, key="p2")

        elif page == "ML Predictive Grid":
            st.markdown('<div class="big-header">Automated ML Forecast Grid</div>', unsafe_allow_html=True)
            target = st.selectbox("Metric to Forecast:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
            
            # Use columns for grid if data exists
            grid_results = []
            for r in sorted(filtered_df['RACE'].unique()):
                for g in sorted(filtered_df['GENDER'].unique()):
                    subset = filtered_df[(filtered_df['RACE'] == r) & (filtered_df['GENDER'] == g)]
                    res, winner, rmse, r2 = run_auto_ml(subset, target)
                    if res is not None:
                        res['RACE'], res['GENDER'] = r, g
                        grid_results.append(res)
            
            if grid_results:
                final_grid = pd.concat(grid_results)
                fig = px.line(final_grid, x="YEAR", y=target, color="Status", facet_row="RACE", facet_col="GENDER", 
                             markers=True, color_discrete_map={'Actual':'#94a3b8','Projected':'#fb7185'})
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Insufficient historical data to trend. Ensure your encounters CSV spans multiple years.")

except Exception as e:
    st.error(f"Critical System Error: {e}")
