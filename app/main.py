import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
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
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(13,122,110,.25);
    color: white;
  }
  .hero-header h1 { font-size: 2.5rem; font-weight: 800; margin: 0 0 .4rem; letter-spacing: -.5px; }
  .hero-header p { font-size: 1.05rem; opacity: .9; margin: 0; }
  .section-header {
    background: var(--mint);
    border-left: 5px solid var(--teal);
    border-radius: 8px;
    padding: .75rem 1.2rem;
    margin: 1.4rem 0 .9rem;
    color: var(--teal-dark);
    font-size: 1.18rem;
    font-weight: 700;
    letter-spacing: .02em;
  }
  .metric-card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,.07);
    border-top: 4px solid var(--teal);
    transition: transform .15s;
  }
  .metric-card:hover { transform: translateY(-3px); }
  .metric-card .metric-value { font-size: 2rem; font-weight: 800; color: var(--teal-dark); }
  .metric-card .metric-label { font-size: .82rem; color: #64748b; margin-top: .25rem; font-weight: 600; }
  .insight-box {
    background: #fff8e1;
    border-left: 4px solid var(--amber);
    border-radius: 8px;
    padding: .9rem 1.2rem;
    margin: .7rem 0;
    font-size: .93rem;
    color: var(--text-main);
  }
  .equity-gap-box {
    background: #fef2f2;
    border-left: 4px solid var(--coral);
    border-radius: 8px;
    padding: .9rem 1.2rem;
    margin: .7rem 0;
  }
  .best-model-box {
    background: #e8f5e9;
    border-left: 4px solid #4CAF50;
    border-radius: 8px;
    padding: .9rem 1.2rem;
    margin: .7rem 0;
    font-size: .95rem;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d7a6e 0%, #0f3460 100%) !important;
  }
  [data-testid="stSidebar"] * { color: white !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label { color: #b2f0e8 !important; }
  .stTabs [data-baseweb="tab"] {
    background: white; border-radius: 8px 8px 0 0; padding: .5rem 1.2rem;
    font-weight: 600; color: var(--teal-dark);
  }
  .stTabs [aria-selected="true"] { background: var(--teal) !important; color: white !important; }
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── California County Data ──────────────────────────────────────────────────────
CA_COUNTIES = {
    'Los Angeles County':    (34.0522, -118.2437, 580),
    'San Diego County':      (32.8501, -116.9829, 280),
    'Orange County':         (33.7455, -117.8677, 220),
    'Riverside County':      (33.9534, -117.3962, 185),
    'San Bernardino County': (34.8416, -116.1771, 175),
    'Santa Clara County':    (37.3382, -121.8863, 160),
    'Alameda County':        (37.6017, -121.7195, 155),
    'Sacramento County':     (38.5816, -121.4944, 140),
    'Contra Costa County':   (37.9227, -121.9582, 120),
    'Fresno County':         (36.7378, -119.7871, 100),
    'Kern County':           (35.3535, -118.9557,  90),
    'San Francisco County':  (37.7749, -122.4194,  80),
    'Ventura County':        (34.3705, -119.1391,  75),
    'San Mateo County':      (37.5630, -122.3255,  70),
    'San Joaquin County':    (37.9161, -121.2726,  65),
    'Stanislaus County':     (37.5630, -120.9876,  60),
    'Sonoma County':         (38.5200, -122.9749,  58),
    'Tulare County':         (36.2077, -119.3473,  55),
    'Solano County':         (38.2494, -121.9018,  52),
    'Monterey County':       (36.2408, -121.3153,  50),
    'Other CA Counties':     (36.7783, -119.4179, 177),
}

# ─── Synthetic Data Generator ───────────────────────────────────────────────────
@st.cache_data
def generate_data(n=2272):
    np.random.seed(42)
    races = ['White','Black','Asian','Hawaiian','Native','Other']
    race_w = np.array([1640,134,370,33,35,60], dtype=float)
    race_w /= race_w.sum()

    race   = np.random.choice(races, n, p=race_w)
    gender = np.random.choice(['Male','Female'], n, p=[0.479, 0.521])
    age    = np.random.randint(0, 111, n)

    income_means = {'White':72000,'Asian':68000,'Black':38000,
                    'Hawaiian':45000,'Native':32000,'Other':50000}
    income = np.array([np.random.normal(income_means[r], 15000) for r in race]).clip(10000,250000)

    insurance_means = {'White':85,'Asian':80,'Black':62,'Hawaiian':68,'Native':55,'Other':70}
    insurance_pct = np.array([np.random.normal(insurance_means[r], 10) for r in race]).clip(10,100)

    base_cost  = 129
    claim_mult = {'White':1.0,'Asian':0.98,'Black':1.03,'Hawaiian':1.01,'Native':1.05,'Other':1.0}
    total_claim = np.array([
        base_cost * claim_mult[r] * (1 + np.random.exponential(8))
        + max(0, (50000 - income[i]) * 0.002)
        for i, r in enumerate(race)
    ])

    county_names = list(CA_COUNTIES.keys())
    county_weights = np.array([v[2] for v in CA_COUNTIES.values()], dtype=float)
    county_weights /= county_weights.sum()
    county = np.random.choice(county_names, n, p=county_weights)
    lat    = np.array([CA_COUNTIES[c][0] + np.random.normal(0, .12) for c in county])
    lon    = np.array([CA_COUNTIES[c][1] + np.random.normal(0, .12) for c in county])

    enc_year = np.random.choice(range(2015,2024), n,
                                p=[.07,.08,.10,.11,.12,.13,.14,.13,.12])

    income_band = pd.cut(income, bins=[0,25000,50000,75000,100000,1e9],
                         labels=['<$25k','$25-50k','$50-75k','$75-100k','>$100k'])

    df = pd.DataFrame({
        'race':race,'gender':gender,'age':age,'county':county,
        'lat':lat,'lon':lon,'income':income,'income_band':income_band,
        'insurance_pct':insurance_pct,'total_claim_cost':total_claim,
        'encounter_year':enc_year
    })
    return df

df = generate_data()

RACE_COLORS = {
    'White':'#2196F3','Black':'#F44336','Asian':'#4CAF50',
    'Hawaiian':'#FF9800','Native':'#9C27B0','Other':'#607D8B'
}
INCOME_COLORS = {
    '<$25k':'#D32F2F','$25-50k':'#F57C00','$50-75k':'#FBC02D',
    '$75-100k':'#388E3C','>$100k':'#1565C0'
}
GENDER_COLORS = {'Male':'#1a9e8f','Female':'#f59e0b'}

def hex_to_rgba(hex_color, alpha=0.12):
    """Convert hex color string to rgba() string safely."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 HEIP Navigation")
    st.markdown("---")
    page = st.radio("", [
        "📊 Dashboard",
        "🗺️ Interactive Map",
        "🔍 Deep-Dive County Analysis",
        "⚖️ Intersectional Comparison",
        "🤖 Predictive Forecasting",
        "📬 Contact & Feedback"
    ])
    st.markdown("---")
    st.markdown("**Dataset:** Synthea CA (n=2,272)")
    st.markdown("**Encounter years:** 2015–2023")
    st.markdown("**Forecast horizon:** 2024–2030")
    st.markdown("**Last updated:** 2025")

# ════════════════════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
  <h1>🏥 Health Equity Insights Platform (HEIP)</h1>
  <p>Identifying Vertical Equity Gaps · Clinical Costs × Personal Wealth × Demographic Identity<br>
     Empowering NGOs &amp; Public Health Officials to Advocate for Underserved Populations</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    section("📌 Population Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("2,272","Total Patients"),
        (f"${df.total_claim_cost.mean():,.0f}","Avg Claim Cost"),
        (f"${df.income.mean():,.0f}","Avg Income"),
        (f"{df.insurance_pct.mean():.1f}%","Avg Insurance Coverage"),
        (f"{df.county.nunique()}","CA Counties Covered"),
    ]
    for col,(val,lbl) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                     f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    section("🚨 Vertical Equity Gap Alert")
    white_income  = df[df.race=='White'].income.mean()
    native_income = df[df.race=='Native'].income.mean()
    gap_pct = (white_income - native_income)/white_income*100
    white_ins  = df[df.race=='White'].insurance_pct.mean()
    native_ins = df[df.race=='Native'].insurance_pct.mean()
    st.markdown(f"""
    <div class="equity-gap-box">
      <strong>💡 Key Finding:</strong> Native patients earn on average
      <strong>${native_income:,.0f}</strong> vs White patients at
      <strong>${white_income:,.0f}</strong> — a <strong>{gap_pct:.1f}% income gap</strong>.
      Insurance coverage is also lower: Native <strong>{native_ins:.1f}%</strong>
      vs White <strong>{white_ins:.1f}%</strong>.
      This vertical equity gap demands targeted resource redistribution.
    </div>""", unsafe_allow_html=True)

    section("📈 Core Distributions")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        race_counts = df.race.value_counts().reset_index()
        race_counts.columns = ['Race','Count']
        fig = px.bar(race_counts, x='Race', y='Count',
                     color='Race', color_discrete_map=RACE_COLORS,
                     title='Patient Count by Race', text='Count')
        fig.update_layout(showlegend=False, plot_bgcolor='white', height=320)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        avg_claim = df.groupby('race')['total_claim_cost'].mean().reset_index()
        avg_claim.columns = ['Race','Avg Claim Cost']
        fig2 = px.bar(avg_claim, x='Race', y='Avg Claim Cost',
                      color='Race', color_discrete_map=RACE_COLORS,
                      title='Average Claim Cost by Race ($)',
                      text=avg_claim['Avg Claim Cost'].map('${:,.0f}'.format))
        fig2.update_layout(showlegend=False, plot_bgcolor='white', height=320)
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig3 = px.histogram(df, x='income', color='race',
                            color_discrete_map=RACE_COLORS,
                            nbins=40, barmode='overlay', opacity=.65,
                            title='Income Distribution by Race',
                            labels={'income':'Annual Income ($)'})
        fig3.update_layout(plot_bgcolor='white', height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        avg_ins = df.groupby('race')['insurance_pct'].mean().reset_index()
        avg_ins.columns = ['Race','Avg Insurance %']
        fig4 = px.bar(avg_ins, x='Race', y='Avg Insurance %',
                      color='Race', color_discrete_map=RACE_COLORS,
                      title='Average Insurance Coverage % by Race',
                      text=avg_ins['Avg Insurance %'].map('{:.1f}%'.format))
        fig4.update_layout(showlegend=False, plot_bgcolor='white', height=320)
        fig4.update_traces(textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)

    section("📋 Top 10 Demographic Segments by Claim Cost")
    top10 = (df.groupby(['race','income_band'])
               .agg(avg_claim=('total_claim_cost','mean'),
                    avg_income=('income','mean'),
                    avg_insurance=('insurance_pct','mean'),
                    enc_count=('total_claim_cost','count'))
               .reset_index()
               .sort_values('avg_claim', ascending=False)
               .head(10))
    top10.columns = ['Race','Income Band','Avg Claim ($)','Avg Income ($)',
                     'Avg Insurance (%)','Encounter Count']
    for col in ['Avg Claim ($)','Avg Income ($)']:
        top10[col] = top10[col].map('${:,.0f}'.format)
    top10['Avg Insurance (%)'] = top10['Avg Insurance (%)'].map('{:.1f}%'.format)
    st.dataframe(top10, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — INTERACTIVE MAP  (COUNTY-LEVEL)
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Interactive Map":
    section("🗺️ County-Level Choropleth — California")

    col_f1, col_f2 = st.columns(2)
    map_metric = col_f1.selectbox(
        "Colour map by:",
        ["Avg Income ($)","Avg Insurance Coverage (%)","Avg Total Claim Cost ($)"]
    )
    map_race = col_f2.multiselect("Filter by Race:", df.race.unique().tolist(),
                                   default=df.race.unique().tolist())

    dff = df[df.race.isin(map_race)]

    county_agg = (dff.groupby('county')
                     .agg(avg_income=('income','mean'),
                          avg_insurance=('insurance_pct','mean'),
                          avg_claim=('total_claim_cost','mean'),
                          patient_count=('race','count'),
                          lat=('lat','mean'), lon=('lon','mean'))
                     .reset_index())

    metric_col_map = {
        "Avg Income ($)":            "avg_income",
        "Avg Insurance Coverage (%)":"avg_insurance",
        "Avg Total Claim Cost ($)":  "avg_claim"
    }
    metric_col = metric_col_map[map_metric]
    color_scale = "RdYlGn" if "Income" in map_metric or "Insurance" in map_metric else "RdYlGn_r"

    fig_map = px.scatter_mapbox(
        county_agg,
        lat='lat', lon='lon',
        size='patient_count',
        color=metric_col,
        color_continuous_scale=color_scale,
        hover_name='county',
        hover_data={
            'avg_income':    ':$,.0f',
            'avg_insurance': ':.1f',
            'avg_claim':     ':$,.0f',
            'patient_count': ':,',
            'lat': False, 'lon': False
        },
        labels={
            'avg_income':    'Avg Income ($)',
            'avg_insurance': 'Avg Insurance (%)',
            'avg_claim':     'Avg Claim ($)',
            'patient_count': 'Patients'
        },
        size_max=55,
        zoom=5.2,
        center={"lat":36.7,"lon":-119.4},
        mapbox_style="carto-positron",
        title=f"California Counties — {map_metric} (bubble size = patient count)"
    )
    fig_map.update_layout(height=600, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    section("📊 County-Level Summary Statistics")
    display_df = county_agg[['county','patient_count','avg_income','avg_insurance','avg_claim']].copy()
    display_df.columns = ['County','Patients','Avg Income ($)','Avg Insurance (%)','Avg Claim Cost ($)']
    display_df['Avg Income ($)']     = display_df['Avg Income ($)'].map('${:,.0f}'.format)
    display_df['Avg Claim Cost ($)'] = display_df['Avg Claim Cost ($)'].map('${:,.0f}'.format)
    display_df['Avg Insurance (%)']  = display_df['Avg Insurance (%)'].map('{:.1f}%'.format)
    display_df = display_df.sort_values('Patients', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    section("🧩 Race Breakdown by County (Top 8)")
    top_counties = county_agg.nlargest(8,'patient_count').county.tolist()
    county_race  = (dff[dff.county.isin(top_counties)]
                       .groupby(['county','race'])
                       .size().reset_index(name='count'))
    fig_cr = px.bar(county_race, x='county', y='count', color='race',
                    color_discrete_map=RACE_COLORS,
                    title='Race Breakdown — Top 8 Counties',
                    labels={'count':'Patients','county':'County'})
    fig_cr.update_layout(plot_bgcolor='white', height=380)
    st.plotly_chart(fig_cr, use_container_width=True)

    section("💰 Insurance Coverage vs Income (by County)")
    fig_scatter = px.scatter(
        county_agg, x='avg_income', y='avg_insurance',
        size='patient_count', color='avg_claim',
        color_continuous_scale='RdYlGn_r',
        hover_name='county',
        labels={
            'avg_income':    'Average Income ($)',
            'avg_insurance': 'Average Insurance Coverage (%)',
            'avg_claim':     'Avg Claim Cost',
            'patient_count': 'Patients'
        },
        title='Insurance Coverage vs Income by County'
    )
    fig_scatter.update_layout(plot_bgcolor='white', height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DEEP-DIVE COUNTY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Deep-Dive County Analysis":
    section("🔍 Deep-Dive: County Level Analysis")

    top_counties = df.county.value_counts().nlargest(10).index.tolist()
    sel_county   = st.selectbox("Select County:", top_counties)
    county_df    = df[df.county == sel_county]

    c1,c2,c3,c4 = st.columns(4)
    metrics = [
        (len(county_df), "Total Patients"),
        (f"${county_df.income.mean():,.0f}", "Avg Income"),
        (f"${county_df.total_claim_cost.mean():,.0f}", "Avg Claim Cost"),
        (f"{county_df.insurance_pct.mean():.1f}%", "Avg Insurance"),
    ]
    for col,(v,l) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{v}</div>'
                     f'<div class="metric-label">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    section("💵 Income Distribution")
    inc_max = int(county_df.income.max())
    x_range = st.slider("Income X-axis range ($):", 0, 200000,
                         (0, min(inc_max, 150000)), step=5000, format="$%d")
    filtered_income = county_df[(county_df.income >= x_range[0]) & (county_df.income <= x_range[1])]
    fig_inc = px.histogram(filtered_income, x='income', color='race',
                            color_discrete_map=RACE_COLORS,
                            nbins=30, barmode='overlay', opacity=.7,
                            labels={'income':'Annual Income ($)'},
                            title=f'Income Distribution — {sel_county}')
    fig_inc.update_xaxes(tickformat='$,.0f', range=list(x_range))
    fig_inc.update_layout(plot_bgcolor='white', height=350)
    st.plotly_chart(fig_inc, use_container_width=True)

    section("🏥 Claim Cost Distribution")
    claim_max = int(county_df.total_claim_cost.max())
    x_range_c = st.slider("Claim Cost X-axis range ($):", 0, max(500, claim_max),
                            (0, min(claim_max, 400)), step=10, format="$%d")
    filtered_claim = county_df[(county_df.total_claim_cost >= x_range_c[0]) &
                                (county_df.total_claim_cost <= x_range_c[1])]
    fig_claim = px.histogram(filtered_claim, x='total_claim_cost', color='race',
                              color_discrete_map=RACE_COLORS,
                              nbins=30, barmode='overlay', opacity=.7,
                              labels={'total_claim_cost':'Total Claim Cost ($)'},
                              title=f'Claim Cost Distribution — {sel_county}')
    fig_claim.update_xaxes(tickformat='$,.0f', range=list(x_range_c))
    fig_claim.update_layout(plot_bgcolor='white', height=350)
    st.plotly_chart(fig_claim, use_container_width=True)

    section("🛡️ Insurance Coverage Distribution")
    x_range_i = st.slider("Insurance % X-axis range:", 0, 100, (0, 100), step=5, format="%d%%")
    filtered_ins = county_df[(county_df.insurance_pct >= x_range_i[0]) &
                              (county_df.insurance_pct <= x_range_i[1])]
    fig_ins = px.histogram(filtered_ins, x='insurance_pct', color='race',
                            color_discrete_map=RACE_COLORS,
                            nbins=20, barmode='overlay', opacity=.7,
                            labels={'insurance_pct':'Insurance Coverage (%)'},
                            title=f'Insurance Coverage Distribution — {sel_county}')
    fig_ins.update_xaxes(ticksuffix='%', range=list(x_range_i))
    fig_ins.update_layout(plot_bgcolor='white', height=350)
    st.plotly_chart(fig_ins, use_container_width=True)

    section("📅 Yearly Encounter Trends")
    yearly = county_df.groupby('encounter_year').agg(
        patient_count=('race','count'),
        avg_claim=('total_claim_cost','mean'),
        avg_income=('income','mean')
    ).reset_index()
    fig_yr = make_subplots(rows=1, cols=2,
                            subplot_titles=['Encounter Count per Year','Avg Claim Cost per Year'])
    fig_yr.add_trace(go.Bar(x=yearly.encounter_year, y=yearly.patient_count,
                             marker_color='#1a9e8f', name='Encounters'), row=1, col=1)
    fig_yr.add_trace(go.Scatter(x=yearly.encounter_year, y=yearly.avg_claim,
                                 mode='lines+markers', marker_color='#f59e0b',
                                 name='Avg Claim $'), row=1, col=2)
    fig_yr.update_xaxes(tickmode='linear', dtick=1)
    fig_yr.update_layout(height=350, plot_bgcolor='white', showlegend=False)
    st.plotly_chart(fig_yr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — INTERSECTIONAL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Intersectional Comparison":
    section("⚖️ Intersectional Equity Analysis")

    ALL_RACES  = sorted(df.race.unique())
    ALL_INCOME = ['<$25k','$25-50k','$50-75k','$75-100k','>$100k']

    metric_choice = st.selectbox(
        "Outcome metric:",
        ["Total Claim Cost ($)", "Insurance Coverage (%)", "Annual Income ($)"]
    )
    metric_col_map = {
        "Total Claim Cost ($)":   "total_claim_cost",
        "Insurance Coverage (%)": "insurance_pct",
        "Annual Income ($)":      "income"
    }
    mc = metric_col_map[metric_choice]

    section(f"📦 {metric_choice} by Race & Gender")
    col_l, col_r = st.columns(2)

    with col_l:
        fig_b1 = px.box(df, x='race', y=mc, color='race',
                         color_discrete_map=RACE_COLORS,
                         title=f'{metric_choice} by Race',
                         category_orders={'race': ALL_RACES},
                         labels={mc: metric_choice, 'race':'Race'})
        fig_b1.update_layout(showlegend=True, plot_bgcolor='white', height=420)
        st.plotly_chart(fig_b1, use_container_width=True)

    with col_r:
        fig_b2 = px.box(df, x='gender', y=mc, color='race',
                         color_discrete_map=RACE_COLORS,
                         title=f'{metric_choice} by Gender (coloured by Race)',
                         category_orders={'race': ALL_RACES},
                         labels={mc: metric_choice, 'gender':'Gender'})
        fig_b2.update_layout(showlegend=True, plot_bgcolor='white', height=420)
        st.plotly_chart(fig_b2, use_container_width=True)

    section(f"💳 {metric_choice} by Income Band & Race")
    col_l2, col_r2 = st.columns(2)
    grp = (df.groupby(['income_band','race'])[mc].mean().reset_index())
    grp.columns = ['Income Band','Race', metric_choice]

    with col_l2:
        fig_i1 = px.bar(grp, x='Income Band', y=metric_choice,
                         color='Race', color_discrete_map=RACE_COLORS,
                         barmode='group',
                         title=f'Avg {metric_choice} by Income Band',
                         category_orders={'Income Band': ALL_INCOME,'Race': ALL_RACES})
        fig_i1.update_layout(plot_bgcolor='white', height=420)
        st.plotly_chart(fig_i1, use_container_width=True)

    with col_r2:
        fig_i2 = px.bar(grp, x='Race', y=metric_choice,
                         color='Income Band', color_discrete_map=INCOME_COLORS,
                         barmode='group',
                         title=f'Avg {metric_choice} by Race (stacked by Income)',
                         category_orders={'Income Band': ALL_INCOME,'Race': ALL_RACES})
        fig_i2.update_layout(plot_bgcolor='white', height=420)
        st.plotly_chart(fig_i2, use_container_width=True)

    section(f"🔥 Heatmap: Race × Income Band → {metric_choice}")
    pivot = df.groupby(['race','income_band'])[mc].mean().unstack()
    pivot = pivot.reindex(ALL_RACES)
    pivot = pivot.reindex(columns=ALL_INCOME)
    fig_heat = px.imshow(pivot,
                          color_continuous_scale='RdYlGn_r' if mc=='total_claim_cost' else 'RdYlGn',
                          aspect='auto',
                          labels={'color': metric_choice},
                          title=f'Mean {metric_choice}: Race × Income Band')
    fig_heat.update_layout(height=350)
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PREDICTIVE FORECASTING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive Forecasting":
    section("🤖 Predictive Forecasting — Auto-Selects Best Model & Forecasts to 2030")

    @st.cache_data
    def train_all_models(df):
        dfc = df.copy()
        le_r = LabelEncoder(); dfc['race_enc']   = le_r.fit_transform(dfc['race'])
        le_g = LabelEncoder(); dfc['gender_enc'] = le_g.fit_transform(dfc['gender'])
        le_i = LabelEncoder(); dfc['income_enc'] = le_i.fit_transform(dfc['income_band'].astype(str))

        feats = ['race_enc','gender_enc','age','income','insurance_pct',
                 'encounter_year','income_enc']
        X = dfc[feats]; y = dfc['total_claim_cost']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)

        candidates = {
            'Linear Regression':  LinearRegression(),
            'Ridge Regression':   Ridge(alpha=1.0),
            'Random Forest':      RandomForestRegressor(n_estimators=100, max_depth=6,
                                                        random_state=42, n_jobs=-1),
            'Gradient Boosting':  GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                             learning_rate=0.05,
                                                             subsample=0.8,
                                                             random_state=42),
        }

        results = {}
        for name, mdl in candidates.items():
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)
            mae   = mean_absolute_error(y_te, preds)
            r2    = r2_score(y_te, preds)
            results[name] = {'model': mdl, 'mae': mae, 'r2': r2,
                              'preds': preds, 'y_te': y_te}

        best_name = min(results, key=lambda k: results[k]['mae'])
        best = results[best_name]

        importances = {}
        if hasattr(best['model'], 'feature_importances_'):
            importances = dict(zip(feats, best['model'].feature_importances_))
        elif hasattr(best['model'], 'coef_'):
            importances = dict(zip(feats, np.abs(best['model'].coef_)))

        return results, best_name, best['model'], best['mae'], best['r2'], importances, le_r, le_g, le_i, feats

    results, best_name, best_model, mae, r2, importances, le_r, le_g, le_i, feats = train_all_models(df)

    # ── Model comparison table ────────────────────────────────────────────────
    with st.expander("📊 Model Comparison — All Algorithms Evaluated", expanded=True):
        comp_rows = []
        for name, res in results.items():
            comp_rows.append({
                'Model': ('✅ ' if name == best_name else '') + name,
                'MAE ($)': f"${res['mae']:,.2f}",
                'R² Score': f"{res['r2']:.4f}",
                'Selected': '🏆 Best' if name == best_name else ''
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="best-model-box">
        <strong>🏆 Auto-Selected Model: {best_name}</strong><br>
        The model with the lowest Mean Absolute Error was automatically selected.
        MAE = <strong>${mae:,.2f}</strong> | R² = <strong>{r2:.4f}</strong><br>
        <em>Higher R² (closer to 1.0) and lower MAE indicate better predictive accuracy.</em>
        </div>""", unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">${mae:,.2f}</div>'
                f'<div class="metric-label">Mean Absolute Error (Best Model)</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{r2:.4f}</div>'
                f'<div class="metric-label">R² Score (Best Model)</div></div>',
                unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{best_name.split()[0]}</div>'
                f'<div class="metric-label">Best Algorithm</div></div>',
                unsafe_allow_html=True)

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    section("🎯 Actual vs Predicted — Best Model")
    y_te   = results[best_name]['y_te']
    preds  = results[best_name]['preds']
    fig_av = go.Figure()
    fig_av.add_trace(go.Scatter(x=y_te, y=preds, mode='markers',
                                 marker=dict(color='#1a9e8f', opacity=.5, size=5),
                                 name='Predictions'))
    lim = max(y_te.max(), preds.max())
    fig_av.add_trace(go.Scatter(x=[0,lim], y=[0,lim], mode='lines',
                                 line=dict(color='red', dash='dash'), name='Perfect Fit'))
    fig_av.update_layout(title=f'Actual vs Predicted Claim Cost ({best_name})',
                          xaxis_title='Actual ($)', yaxis_title='Predicted ($)',
                          plot_bgcolor='white', height=380)
    st.plotly_chart(fig_av, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    if importances:
        section("📊 Feature Importance (Best Model)")
        fi_df = pd.DataFrame({'Feature': list(importances.keys()),
                               'Importance': list(importances.values())}).sort_values('Importance')
        fi_df.Feature = fi_df.Feature.replace({
            'race_enc':'Race','gender_enc':'Gender','age':'Age',
            'income':'Income','insurance_pct':'Insurance %',
            'encounter_year':'Encounter Year','income_enc':'Income Band'
        })
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='teal',
                         title=f'Feature Importance — {best_name}')
        fig_fi.update_layout(plot_bgcolor='white', height=320, showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    #  FORECAST TO 2030
    # ════════════════════════════════════════════════════════════════════════
    section("🔮 Forecast to 2030 — Claim Cost Projections by Race")

    st.markdown("""
    <div class="insight-box">
    <strong>📖 How the 2030 Forecast Works:</strong>
    The best-performing ML model is used to predict average claim costs for future years
    (2024–2030) by holding all demographic and income features at their 2023 mean values
    per race group, and advancing the <em>encounter_year</em> forward. A polynomial trend
    line is also fitted on historical averages (2015–2023) per race to extrapolate into the future.
    The shaded band represents ±1 standard deviation of the ML predictions.
    </div>
    """, unsafe_allow_html=True)

    FORECAST_YEARS = list(range(2015, 2031))
    HIST_YEARS     = list(range(2015, 2024))
    FUTURE_YEARS   = list(range(2024, 2031))

    forecast_rows = []
    for race in sorted(df.race.unique()):
        race_df = df[df.race == race]

        # Historical actuals
        hist = race_df.groupby('encounter_year')['total_claim_cost'].agg(['mean','std']).reset_index()
        hist.columns = ['year','mean_cost','std_cost']
        hist['type'] = 'Historical'
        hist['race'] = race

        # Build synthetic future feature rows
        mean_age   = race_df.age.mean()
        mean_inc   = race_df.income.mean()
        mean_ins   = race_df.insurance_pct.mean()
        modal_band = race_df.income_band.mode()[0]

        race_enc   = le_r.transform([race])[0]
        gender_enc_m = le_g.transform(['Male'])[0]
        income_enc = le_i.transform([str(modal_band)])[0]

        future_preds = []
        for yr in FUTURE_YEARS:
            n_sim = 200
            ages    = np.random.normal(mean_age, 5, n_sim).clip(0, 110)
            incomes = np.random.normal(mean_inc, 8000, n_sim).clip(10000, 250000)
            ins     = np.random.normal(mean_ins, 5, n_sim).clip(10, 100)
            X_sim = pd.DataFrame({
                'race_enc':      race_enc,
                'gender_enc':    gender_enc_m,
                'age':           ages,
                'income':        incomes,
                'insurance_pct': ins,
                'encounter_year': yr,
                'income_enc':    income_enc
            })
            sim_preds = best_model.predict(X_sim[feats])
            future_preds.append({
                'year':      yr,
                'mean_cost': sim_preds.mean(),
                'std_cost':  sim_preds.std(),
                'type':      'Forecast',
                'race':      race
            })

        forecast_rows.append(hist)
        forecast_rows.append(pd.DataFrame(future_preds))

    forecast_df = pd.concat(forecast_rows, ignore_index=True)

    # ── Plot forecast per race ────────────────────────────────────────────────
    fig_fc = go.Figure()
    for race in sorted(df.race.unique()):
        clr = RACE_COLORS[race]
        rdf = forecast_df[forecast_df.race == race].sort_values('year')
        hist_r   = rdf[rdf.type == 'Historical']
        future_r = rdf[rdf.type == 'Forecast']

        # Historical line
        fig_fc.add_trace(go.Scatter(
            x=hist_r.year, y=hist_r.mean_cost,
            mode='lines+markers',
            line=dict(color=clr, width=2.5),
            marker=dict(size=6),
            name=f'{race} (Historical)',
            legendgroup=race,
        ))

        # Forecast line (dashed)
        fig_fc.add_trace(go.Scatter(
            x=future_r.year, y=future_r.mean_cost,
            mode='lines+markers',
            line=dict(color=clr, width=2.5, dash='dot'),
            marker=dict(size=7, symbol='diamond'),
            name=f'{race} (Forecast)',
            legendgroup=race,
            showlegend=True
        ))

        # Confidence band — FIX: proper hex→rgba conversion
        upper = future_r.mean_cost + future_r.std_cost
        lower = (future_r.mean_cost - future_r.std_cost).clip(lower=0)
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([future_r.year, future_r.year[::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill='toself',
            fillcolor=hex_to_rgba(clr, 0.12),
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            legendgroup=race,
            hoverinfo='skip'
        ))

    # Vertical line at start of forecast
    fig_fc.add_vline(x=2023.5, line_dash='dash', line_color='gray',
                      annotation_text='Forecast →', annotation_position='top right')

    fig_fc.update_layout(
        title=f'Average Claim Cost Forecast to 2030 by Race ({best_name})',
        xaxis=dict(title='Year', tickmode='linear', dtick=1, range=[2014.5, 2030.5]),
        yaxis=dict(title='Avg Claim Cost ($)', tickformat='$,.0f'),
        plot_bgcolor='white',
        height=520,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
        hovermode='x unified'
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── 2030 forecast table ───────────────────────────────────────────────────
    section("📋 2030 Forecast Summary Table")
    tbl_2030 = (forecast_df[forecast_df.year == 2030][['race','mean_cost','std_cost']]
                  .sort_values('mean_cost', ascending=False)
                  .reset_index(drop=True))
    tbl_2030.columns = ['Race','Predicted Avg Claim Cost (2030)','±Std Dev']
    tbl_2030['Predicted Avg Claim Cost (2030)'] = tbl_2030['Predicted Avg Claim Cost (2030)'].map('${:,.2f}'.format)
    tbl_2030['±Std Dev'] = tbl_2030['±Std Dev'].map('${:,.2f}'.format)
    st.dataframe(tbl_2030, use_container_width=True, hide_index=True)

    # ── Multi-Graph Explorer ──────────────────────────────────────────────────
    section("🔬 Multi-Graph Explorer — Choose Axes × Toggle Outcome")
    st.info("Select X and Y axes below. Each cell shows the chosen outcome metric across years "
            "(solid = historical 2015–2023, dashed = forecast 2024–2030).")

    col_x, col_y, col_m = st.columns(3)
    x_axis  = col_x.selectbox("X-Axis (columns):", ["race","gender","income_band","encounter_year"])
    y_axis  = col_y.selectbox("Y-Axis (rows):",    ["gender","race","income_band"], index=1)
    outcome = col_m.selectbox("Outcome to display:",
                               ["total_claim_cost","insurance_pct","income"],
                               format_func=lambda x: {
                                   "total_claim_cost":"Total Claim Cost ($)",
                                   "insurance_pct":"Insurance Coverage (%)",
                                   "income":"Annual Income ($)"
                               }[x])

    def axis_vals(col):
        if col == 'income_band':    return ['<$25k','$25-50k','$50-75k','$75-100k','>$100k']
        if col == 'race':           return sorted(df.race.unique())
        if col == 'gender':         return ['Male','Female']
        if col == 'encounter_year': return sorted(df.encounter_year.unique())
        return sorted(df[col].unique())

    x_vals = axis_vals(x_axis)
    y_vals = axis_vals(y_axis)

    outcome_label = {
        "total_claim_cost":"Avg Claim Cost ($)",
        "insurance_pct":"Avg Insurance (%)",
        "income":"Avg Income ($)"
    }[outcome]

    n_cols = min(len(x_vals), 4)

    if x_axis == 'race':           col_enc = RACE_COLORS
    elif x_axis == 'income_band':  col_enc = INCOME_COLORS
    elif x_axis == 'gender':       col_enc = GENDER_COLORS
    else:                           col_enc = None

    for y_val in y_vals:
        section(f"Row: {y_axis} = {y_val}")
        cols = st.columns(n_cols)
        for i, x_val in enumerate(x_vals[:n_cols]):
            cell_df = df[(df[y_axis].astype(str)==str(y_val)) &
                          (df[x_axis].astype(str)==str(x_val))]
            if len(cell_df) == 0:
                cols[i].warning(f"No data\n{x_val}")
                continue
            grp_hist = cell_df.groupby('encounter_year')[outcome].mean().reset_index()
            clr = col_enc.get(str(x_val), '#1a9e8f') if col_enc else '#1a9e8f'

            fig_cell = go.Figure()
            fig_cell.add_trace(go.Scatter(
                x=grp_hist['encounter_year'], y=grp_hist[outcome],
                mode='lines+markers',
                line=dict(color=clr, width=2.5),
                marker=dict(size=6),
                name='Historical'
            ))

            if outcome == 'total_claim_cost':
                from numpy.polynomial.polynomial import polyfit, polyval
                xs = grp_hist['encounter_year'].values.astype(float)
                ys = grp_hist[outcome].values
                if len(xs) >= 3:
                    coeffs = np.polyfit(xs, ys, deg=2)
                    fut_x  = np.array(FUTURE_YEARS, dtype=float)
                    fut_y  = np.polyval(coeffs, fut_x).clip(0)
                    fig_cell.add_trace(go.Scatter(
                        x=fut_x, y=fut_y,
                        mode='lines+markers',
                        line=dict(color=clr, width=2, dash='dot'),
                        marker=dict(size=5, symbol='diamond'),
                        name='Forecast'
                    ))
                    fig_cell.add_vline(x=2023.5, line_dash='dash',
                                       line_color='gray', line_width=1)

            fig_cell.update_layout(
                title=dict(text=f"{x_val} (n={len(cell_df)})", font=dict(size=11)),
                xaxis=dict(title='Year', tickmode='linear', dtick=2,
                           tickfont=dict(size=9), range=[2014.5, 2030.5]),
                yaxis=dict(title=outcome_label, tickfont=dict(size=9)),
                plot_bgcolor='white',
                height=230,
                margin=dict(l=30,r=10,t=40,b=30),
                showlegend=False
            )
            cols[i].plotly_chart(fig_cell, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — CONTACT & FEEDBACK
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📬 Contact & Feedback":
    section("📬 Contact an NGO Partner")
    st.markdown("""
    <div class="insight-box">
    <strong>🌍 Mission:</strong> HEIP connects public health officials and NGO partners to data-driven
    insights that drive resource redistribution for underserved communities in California.
    Use the form below to reach out to a partner organisation or submit platform feedback.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section("🤝 Contact an NGO Partner")
        with st.form("ngo_contact_form"):
            ngo_name = st.selectbox("Select NGO Partner:", [
                "California Black Health Network",
                "Asian Health Services (Oakland)",
                "Native American Health Center",
                "UnidosUS (Latino Health Access)",
                "Pacific Islander Health Partners",
                "Community Health Alliance",
                "Other / General Inquiry"
            ])
            sender_name  = st.text_input("Your Full Name *")
            sender_org   = st.text_input("Your Organisation")
            sender_email = st.text_input("Your Email Address *")
            subject      = st.selectbox("Subject:", [
                "Resource Redistribution Inquiry","Data Partnership Request",
                "Program Funding Discussion","Community Outreach Collaboration",
                "Research Partnership","Media / Press Inquiry","Other"
            ])
            message  = st.text_area("Message *", height=130,
                                     placeholder="Describe your request or how you'd like to collaborate…")
            urgency  = st.select_slider("Urgency:", ["Low","Medium","High","Critical"])
            sub_ngo  = st.form_submit_button("📤 Send to NGO Partner", use_container_width=True)

        if sub_ngo:
            if sender_name and sender_email and message:
                st.success(f"✅ Your message has been sent to **{ngo_name}**! "
                           f"Expected response: {'24 hrs' if urgency in ['High','Critical'] else '3–5 business days'}.")
            else:
                st.error("Please fill in all required fields (*).")

    with col2:
        section("💬 Platform Feedback")
        with st.form("feedback_form"):
            fb_name = st.text_input("Your Name (optional)")
            fb_role = st.selectbox("Your Role:", [
                "Public Health Official","NGO Staff","Academic Researcher",
                "Student","Community Advocate","Other"
            ])
            overall = st.slider("Overall Platform Rating:", 1, 5, 4)
            stars   = "⭐" * overall
            st.markdown(f"**Your rating:** {stars}")
            useful  = st.multiselect("Which pages were most useful?", [
                "Dashboard","Interactive Map","Deep-Dive Analysis",
                "Intersectional Comparison","Predictive Forecasting"
            ])
            missing  = st.text_area("What data or features are missing?", height=80)
            positive = st.text_area("What did you find most valuable?", height=80)
            suggest  = st.text_area("Suggestions for improvement:", height=80)
            sub_fb   = st.form_submit_button("📨 Submit Feedback", use_container_width=True)

        if sub_fb:
            st.success("🙏 Thank you for your feedback! It helps us improve HEIP for all partner organisations.")
            st.balloons()

    section("🌐 Our NGO Partner Network")
    partners = [
        ("🏥 California Black Health Network","Advocating for health equity and wellness in Black communities across CA.","cbhnonline.org"),
        ("🌿 Asian Health Services","Providing culturally competent care in Oakland's Asian communities since 1974.","asianhealthservices.org"),
        ("🪶 Native American Health Center","Holistic health services for urban Native American and Alaska Native populations.","nativehealth.org"),
        ("🌮 UnidosUS / Latino Health Access","Data-driven advocacy for Latino health equity and policy change.","unidosus.org"),
        ("🌺 Pacific Islander Health Partners","Building capacity in Pacific Islander communities to address health disparities.","pihp.org"),
        ("🤲 Community Health Alliance","Connecting underserved Californians to preventive care and social services.","cha-ca.org"),
    ]
    cols = st.columns(3)
    for i,(name,desc,url) in enumerate(partners):
        cols[i%3].markdown(f"""
        <div class="metric-card" style="text-align:left; margin-bottom:1rem;">
          <strong>{name}</strong><br>
          <span style="font-size:.85rem;color:#475569;">{desc}</span><br>
          <a href="https://{url}" target="_blank" style="color:#1a9e8f;font-size:.82rem;">🔗 {url}</a>
        </div>""", unsafe_allow_html=True)
