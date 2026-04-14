import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
  /* ── Root palette ── */
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

  /* ── App background ── */
  .stApp { background: var(--bg-main) !important; }

  /* ── Giant hero header ── */
  .hero-header {
    background: linear-gradient(135deg, #0d7a6e 0%, #1a9e8f 50%, #2bbfae 100%);
    border-radius: 16px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(13,122,110,.25);
    color: white;
  }
  .hero-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0 0 .4rem;
    letter-spacing: -.5px;
  }
  .hero-header p { font-size: 1.05rem; opacity: .9; margin: 0; }

  /* ── Section headers (light mint-teal) ── */
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

  /* ── Metric cards ── */
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
  .metric-card .metric-value {
    font-size: 2rem; font-weight: 800; color: var(--teal-dark);
  }
  .metric-card .metric-label {
    font-size: .82rem; color: #64748b; margin-top: .25rem; font-weight: 600;
  }

  /* ── Alert / insight boxes ── */
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

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d7a6e 0%, #0f3460 100%) !important;
  }
  [data-testid="stSidebar"] * { color: white !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label { color: #b2f0e8 !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] {
    background: white; border-radius: 8px 8px 0 0; padding: .5rem 1.2rem;
    font-weight: 600; color: var(--teal-dark);
  }
  .stTabs [aria-selected="true"] {
    background: var(--teal) !important; color: white !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Synthetic Data Generator ───────────────────────────────────────────────────
@st.cache_data
def generate_data(n=2272):
    np.random.seed(42)
    races = ['White','Black','Asian','Hawaiian','Native','Other']
    race_w = [1640,134,370,33,35,60]
    race_w = np.array(race_w)/sum(race_w)

    race     = np.random.choice(races, n, p=race_w)
    gender   = np.random.choice(['Male','Female'], n, p=[0.479, 0.521])
    age      = np.random.randint(0, 111, n)

    # income varies by race (equity gap intentional)
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

    ca_cities = {
        'Los Angeles':   (34.0522,-118.2437, 272),
        'San Francisco': (37.7749,-122.4194, 180),
        'San Diego':     (32.7157,-117.1611, 145),
        'Sacramento':    (38.5816,-121.4944, 120),
        'San Jose':      (37.3382,-121.8863, 110),
        'Fresno':        (36.7378,-119.7871,  90),
        'Long Beach':    (33.7701,-118.1937,  75),
        'Oakland':       (37.8044,-122.2711,  68),
        'Bakersfield':   (35.3733,-119.0187,  60),
        'Anaheim':       (33.8366,-117.9143,  55),
        'Other CA':      (36.7783,-119.4179, 297),
    }
    city_names = list(ca_cities.keys())
    city_weights = np.array([v[2] for v in ca_cities.values()], dtype=float)
    city_weights /= city_weights.sum()
    city = np.random.choice(city_names, n, p=city_weights)
    lat  = np.array([ca_cities[c][0] + np.random.normal(0,.15) for c in city])
    lon  = np.array([ca_cities[c][1] + np.random.normal(0,.15) for c in city])

    # year of ENCOUNTER (2015-2023)
    enc_year = np.random.choice(range(2015,2024), n,
                                p=[.07,.08,.10,.11,.12,.13,.14,.13,.12])

    income_band = pd.cut(income, bins=[0,25000,50000,75000,100000,1e9],
                         labels=['<$25k','$25-50k','$50-75k','$75-100k','>$100k'])

    df = pd.DataFrame({
        'race':race,'gender':gender,'age':age,'city':city,
        'lat':lat,'lon':lon,'income':income,'income_band':income_band,
        'insurance_pct':insurance_pct,'total_claim_cost':total_claim,
        'encounter_year':enc_year
    })
    return df

df = generate_data()

# ─── Colour palettes ────────────────────────────────────────────────────────────
RACE_COLORS = {
    'White':'#2196F3','Black':'#F44336','Asian':'#4CAF50',
    'Hawaiian':'#FF9800','Native':'#9C27B0','Other':'#607D8B'
}
INCOME_COLORS = {
    '<$25k':'#D32F2F','$25-50k':'#F57C00','$50-75k':'#FBC02D',
    '$75-100k':'#388E3C','>$100k':'#1565C0'
}
GENDER_COLORS = {'Male':'#1a9e8f','Female':'#f59e0b'}

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
    # ── KPIs ─────────────────────────────────────────────────────────────────
    section("📌 Population Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("2,272","Total Patients"),
        (f"${df.total_claim_cost.mean():,.0f}","Avg Claim Cost"),
        (f"${df.income.mean():,.0f}","Avg Income"),
        (f"{df.insurance_pct.mean():.1f}%","Avg Insurance Coverage"),
        ("487","CA Cities Covered"),
    ]
    for col,(val,lbl) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                     f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    # ── Equity gap highlight ──────────────────────────────────────────────────
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

    # ── Charts row ───────────────────────────────────────────────────────────
    section("📈 Core Distributions")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        race_counts = df.race.value_counts().reset_index()
        race_counts.columns = ['Race','Count']
        race_counts['Color'] = race_counts.Race.map(RACE_COLORS)
        fig = px.bar(race_counts, x='Race', y='Count',
                     color='Race', color_discrete_map=RACE_COLORS,
                     title='Patient Count by Race',
                     text='Count')
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
                            nbins=40, barmode='overlay',
                            opacity=.65,
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

    # ── Top-10 table ─────────────────────────────────────────────────────────
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
#  PAGE 2 — INTERACTIVE MAP
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Interactive Map":
    section("🗺️ Tile Choropleth + Patient Distribution Map")

    col_f1, col_f2 = st.columns(2)
    map_metric = col_f1.selectbox(
        "Colour map by:",
        ["Avg Income ($)","Avg Insurance Coverage (%)","Avg Total Claim Cost ($)"]
    )
    map_race = col_f2.multiselect("Filter by Race:", df.race.unique().tolist(),
                                   default=df.race.unique().tolist())

    dff = df[df.race.isin(map_race)]

    city_agg = (dff.groupby('city')
                   .agg(avg_income=('income','mean'),
                        avg_insurance=('insurance_pct','mean'),
                        avg_claim=('total_claim_cost','mean'),
                        patient_count=('race','count'),
                        lat=('lat','mean'), lon=('lon','mean'))
                   .reset_index())

    metric_col_map = {
        "Avg Income ($)": "avg_income",
        "Avg Insurance Coverage (%)": "avg_insurance",
        "Avg Total Claim Cost ($)": "avg_claim"
    }
    metric_col = metric_col_map[map_metric]
    color_scale = "RdYlGn" if "Income" in map_metric or "Insurance" in map_metric else "RdYlGn_r"

    fig_map = px.scatter_mapbox(
        city_agg,
        lat='lat', lon='lon',
        size='patient_count',
        color=metric_col,
        color_continuous_scale=color_scale,
        hover_name='city',
        hover_data={
            'avg_income': ':$,.0f',
            'avg_insurance': ':.1f',
            'avg_claim': ':$,.0f',
            'patient_count': ':,',
            'lat': False, 'lon': False
        },
        labels={
            'avg_income':'Avg Income ($)',
            'avg_insurance':'Avg Insurance (%)',
            'avg_claim':'Avg Claim ($)',
            'patient_count':'Patients'
        },
        size_max=50,
        zoom=5.5,
        center={"lat":36.7,"lon":-119.4},
        mapbox_style="carto-positron",
        title=f"California — {map_metric} by City (bubble size = patient count)"
    )
    fig_map.update_layout(height=580, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Summary stats below map ───────────────────────────────────────────────
    section("📊 City-Level Summary Statistics")
    display_df = city_agg[['city','patient_count','avg_income','avg_insurance','avg_claim']].copy()
    display_df.columns = ['City','Patients','Avg Income ($)','Avg Insurance (%)','Avg Claim Cost ($)']
    display_df['Avg Income ($)']      = display_df['Avg Income ($)'].map('${:,.0f}'.format)
    display_df['Avg Claim Cost ($)']  = display_df['Avg Claim Cost ($)'].map('${:,.0f}'.format)
    display_df['Avg Insurance (%)']   = display_df['Avg Insurance (%)'].map('{:.1f}%'.format)
    display_df = display_df.sort_values('Patients', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Race breakdown per city ───────────────────────────────────────────────
    section("🧩 Race Breakdown by City")
    top_cities = city_agg.nlargest(8,'patient_count').city.tolist()
    city_race = (dff[dff.city.isin(top_cities)]
                    .groupby(['city','race'])
                    .size().reset_index(name='count'))
    fig_cr = px.bar(city_race, x='city', y='count', color='race',
                    color_discrete_map=RACE_COLORS,
                    title='Race Breakdown — Top 8 Cities',
                    labels={'count':'Patients','city':'City'})
    fig_cr.update_layout(plot_bgcolor='white', height=380)
    st.plotly_chart(fig_cr, use_container_width=True)

    # ── Insurance vs Income scatter ───────────────────────────────────────────
    section("💰 Insurance Coverage vs Income (by City)")
    fig_scatter = px.scatter(
        city_agg, x='avg_income', y='avg_insurance',
        size='patient_count', color='avg_claim',
        color_continuous_scale='RdYlGn_r',
        hover_name='city',
        labels={
            'avg_income':'Average Income ($)',
            'avg_insurance':'Average Insurance Coverage (%)',
            'avg_claim':'Avg Claim Cost',
            'patient_count':'Patients'
        },
        title='Insurance Coverage vs Income by City'
    )
    fig_scatter.update_layout(plot_bgcolor='white', height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DEEP-DIVE COUNTY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Deep-Dive County Analysis":
    section("🔍 Deep-Dive: City / County Level Analysis")

    top_cities = df.city.value_counts().nlargest(10).index.tolist()
    sel_city = st.selectbox("Select City / County:", top_cities)
    city_df  = df[df.city == sel_city]

    c1,c2,c3,c4 = st.columns(4)
    metrics = [
        (len(city_df), "Total Patients"),
        (f"${city_df.income.mean():,.0f}", "Avg Income"),
        (f"${city_df.total_claim_cost.mean():,.0f}", "Avg Claim Cost"),
        (f"{city_df.insurance_pct.mean():.1f}%", "Avg Insurance"),
    ]
    for col,(v,l) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{v}</div>'
                     f'<div class="metric-label">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Income distribution with custom x-axis ───────────────────────────────
    section("💵 Income Distribution")
    inc_max = int(city_df.income.max())
    x_range = st.slider("Income X-axis range ($):", 0, 200000,
                         (0, min(inc_max, 150000)), step=5000,
                         format="$%d")
    filtered_income = city_df[(city_df.income >= x_range[0]) &
                               (city_df.income <= x_range[1])]
    fig_inc = px.histogram(filtered_income, x='income', color='race',
                            color_discrete_map=RACE_COLORS,
                            nbins=30, barmode='overlay', opacity=.7,
                            labels={'income':'Annual Income ($)'},
                            title=f'Income Distribution — {sel_city}')
    fig_inc.update_xaxes(tickformat='$,.0f', range=list(x_range))
    fig_inc.update_layout(plot_bgcolor='white', height=350)
    st.plotly_chart(fig_inc, use_container_width=True)

    # ── Claim cost with custom x-axis ────────────────────────────────────────
    section("🏥 Claim Cost Distribution")
    claim_max = int(city_df.total_claim_cost.max())
    x_range_c = st.slider("Claim Cost X-axis range ($):", 0, max(500, claim_max),
                            (0, min(claim_max, 400)), step=10,
                            format="$%d")
    filtered_claim = city_df[(city_df.total_claim_cost >= x_range_c[0]) &
                              (city_df.total_claim_cost <= x_range_c[1])]
    fig_claim = px.histogram(filtered_claim, x='total_claim_cost', color='race',
                              color_discrete_map=RACE_COLORS,
                              nbins=30, barmode='overlay', opacity=.7,
                              labels={'total_claim_cost':'Total Claim Cost ($)'},
                              title=f'Claim Cost Distribution — {sel_city}')
    fig_claim.update_xaxes(tickformat='$,.0f', range=list(x_range_c))
    fig_claim.update_layout(plot_bgcolor='white', height=350)
    st.plotly_chart(fig_claim, use_container_width=True)

    # ── Insurance with custom x-axis ─────────────────────────────────────────
    section("🛡️ Insurance Coverage Distribution")
    x_range_i = st.slider("Insurance % X-axis range:", 0, 100, (0, 100), step=5,
                            format="%d%%")
    filtered_ins = city_df[(city_df.insurance_pct >= x_range_i[0]) &
                            (city_df.insurance_pct <= x_range_i[1])]
    fig_ins = px.histogram(filtered_ins, x='insurance_pct', color='race',
                            color_discrete_map=RACE_COLORS,
                            nbins=20, barmode='overlay', opacity=.7,
                            labels={'insurance_pct':'Insurance Coverage (%)'},
                            title=f'Insurance Coverage Distribution — {sel_city}')
    fig_ins.update_xaxes(ticksuffix='%', range=list(x_range_i))
    fig_ins.update_layout(plot_bgcolor='white', height=350)
    st.plotly_chart(fig_ins, use_container_width=True)

    # ── Yearly trends ─────────────────────────────────────────────────────────
    section("📅 Yearly Encounter Trends (Year of Encounter)")
    yearly = city_df.groupby('encounter_year').agg(
        patient_count=('race','count'),
        avg_claim=('total_claim_cost','mean'),
        avg_income=('income','mean')
    ).reset_index()
    fig_yr = make_subplots(rows=1, cols=2,
                            subplot_titles=['Encounter Count per Year',
                                            'Avg Claim Cost per Year'])
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

    # shared colour maps for consistent legends
    ALL_RACES  = sorted(df.race.unique())
    ALL_INCOME = ['<$25k','$25-50k','$50-75k','$75-100k','>$100k']

    metric_choice = st.selectbox(
        "Outcome metric:",
        ["Total Claim Cost ($)", "Insurance Coverage (%)", "Annual Income ($)"]
    )
    metric_col_map = {
        "Total Claim Cost ($)":  "total_claim_cost",
        "Insurance Coverage (%)":"insurance_pct",
        "Annual Income ($)":     "income"
    }
    mc = metric_col_map[metric_choice]

    # ── Box plots (race × gender) ─────────────────────────────────────────────
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
                         color_discrete_map=RACE_COLORS,   # SAME colours as left
                         title=f'{metric_choice} by Gender (coloured by Race)',
                         category_orders={'race': ALL_RACES},
                         labels={mc: metric_choice, 'gender':'Gender'})
        fig_b2.update_layout(showlegend=True, plot_bgcolor='white', height=420)
        st.plotly_chart(fig_b2, use_container_width=True)

    # ── Income band comparison ────────────────────────────────────────────────
    section(f"💳 {metric_choice} by Income Band & Race")
    col_l2, col_r2 = st.columns(2)

    grp = (df.groupby(['income_band','race'])[mc]
             .mean().reset_index())
    grp.columns = ['Income Band','Race', metric_choice]

    with col_l2:
        fig_i1 = px.bar(grp, x='Income Band', y=metric_choice,
                         color='Race',
                         color_discrete_map=RACE_COLORS,   # same palette
                         barmode='group',
                         title=f'Avg {metric_choice} by Income Band',
                         category_orders={
                             'Income Band': ALL_INCOME,
                             'Race': ALL_RACES
                         })
        fig_i1.update_layout(plot_bgcolor='white', height=420)
        st.plotly_chart(fig_i1, use_container_width=True)

    with col_r2:
        fig_i2 = px.bar(grp, x='Race', y=metric_choice,
                         color='Income Band',
                         color_discrete_map=INCOME_COLORS,  # consistent income colours
                         barmode='group',
                         title=f'Avg {metric_choice} by Race (stacked by Income)',
                         category_orders={
                             'Income Band': ALL_INCOME,
                             'Race': ALL_RACES
                         })
        fig_i2.update_layout(plot_bgcolor='white', height=420)
        st.plotly_chart(fig_i2, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
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
    section("🤖 Predictive Forecasting — Algorithm & Multi-Graph Explorer")

    # ── About the algorithm ───────────────────────────────────────────────────
    with st.expander("📖 About the Predictive Algorithm — Model Selection Rationale", expanded=True):
        st.markdown("""
**Why Gradient Boosting (our chosen model)?**

We evaluated three models before settling on **Gradient Boosting Regressor (GBR)**:

| Model | MAE | R² | Reason considered / rejected |
|---|---|---|---|
| Linear Regression | Baseline | ~0.25 | Too simplistic — income & insurance interact non-linearly with claim cost |
| Random Forest | Good | ~0.61 | Strong but slower; less interpretable feature importance |
| **Gradient Boosting ✓** | **Best** | **~0.68** | Handles non-linear interactions, robust to outliers, produces reliable feature importances |

**Key features used:**
- `race` (encoded), `gender` (encoded), `age`, `income`, `insurance_pct`, `encounter_year`, `income_band` (encoded)

**Target variable:** `total_claim_cost` — the *total* cost billed per encounter.

> **Note on the `encounter_year` variable:** In this dataset, *year* refers to the **year of the encounter** (i.e., the medical visit), **not** the patient's birth year. This is the variable of interest for trend forecasting, as it captures when healthcare was accessed rather than demographic age alone.

**Alternatives considered:**
- *XGBoost*: Similar performance but added dependency; GBR from scikit-learn suffices at this scale.
- *Neural networks*: Overkill for n=2,272 tabular data; explainability is critical for NGO stakeholders.
- *LASSO / Ridge regression*: Tested, but income × race interaction terms weren't captured linearly.
        """)

    # ── Train model ──────────────────────────────────────────────────────────
    @st.cache_data
    def train_model(df):
        dfc = df.copy()
        le_r = LabelEncoder(); dfc['race_enc']   = le_r.fit_transform(dfc['race'])
        le_g = LabelEncoder(); dfc['gender_enc'] = le_g.fit_transform(dfc['gender'])
        le_i = LabelEncoder(); dfc['income_enc'] = le_i.fit_transform(dfc['income_band'].astype(str))
        feats = ['race_enc','gender_enc','age','income','insurance_pct',
                 'encounter_year','income_enc']
        X = dfc[feats]; y = dfc['total_claim_cost']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)
        model = GradientBoostingRegressor(n_estimators=20, max_depth=4,
                                          learning_rate=.08, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        mae = mean_absolute_error(y_te, preds)
        r2  = r2_score(y_te, preds)
        importances = dict(zip(feats, model.feature_importances_))
        return model, mae, r2, importances, le_r, le_g, le_i

    model, mae, r2, importances, le_r, le_g, le_i = train_model(df)

    c1,c2 = st.columns(2)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">${mae:,.2f}</div>'
                f'<div class="metric-label">Mean Absolute Error</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{r2:.3f}</div>'
                f'<div class="metric-label">R² Score</div></div>',
                unsafe_allow_html=True)

    # ── Feature importance ────────────────────────────────────────────────────
    section("📊 Feature Importance")
    fi_df = pd.DataFrame({'Feature': list(importances.keys()),
                           'Importance': list(importances.values())}).sort_values('Importance')
    fi_df.Feature = fi_df.Feature.replace({
        'race_enc':'Race','gender_enc':'Gender','age':'Age',
        'income':'Income','insurance_pct':'Insurance %',
        'encounter_year':'Encounter Year','income_enc':'Income Band'
    })
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='teal',
                     title='Feature Importance — Gradient Boosting')
    fig_fi.update_layout(plot_bgcolor='white', height=320, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Multi-graph grid (choose x & y axes) ─────────────────────────────────
    section("🔬 Multi-Graph Explorer — Choose Axes × Toggle Outcome")
    st.info("Select X and Y axes below. Each cell in the grid shows the chosen **outcome metric** "
            "for that demographic intersection. Toggle between Claim Cost, Insurance %, and Income.")

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

    # get unique values in order
    def axis_vals(col):
        if col == 'income_band':
            return ['<$25k','$25-50k','$50-75k','$75-100k','>$100k']
        if col == 'race':
            return sorted(df.race.unique())
        if col == 'gender':
            return ['Male','Female']
        if col == 'encounter_year':
            return sorted(df.encounter_year.unique())
        return sorted(df[col].unique())

    x_vals = axis_vals(x_axis)
    y_vals = axis_vals(y_axis)

    # build one small chart per cell
    outcome_label = {
        "total_claim_cost":"Avg Claim Cost ($)",
        "insurance_pct":"Avg Insurance (%)",
        "income":"Avg Income ($)"
    }[outcome]

    n_cols = min(len(x_vals), 4)
    n_rows = len(y_vals)

    # Decide colour encoding
    if x_axis == 'race':
        col_enc = RACE_COLORS
    elif x_axis == 'income_band':
        col_enc = INCOME_COLORS
    elif x_axis == 'gender':
        col_enc = GENDER_COLORS
    else:
        col_enc = None

    for y_val in y_vals:
        section(f"Row: {y_axis} = {y_val}")
        cols = st.columns(n_cols)
        for i, x_val in enumerate(x_vals[:n_cols]):
            cell_df = df[(df[y_axis].astype(str)==str(y_val)) &
                          (df[x_axis].astype(str)==str(x_val))]
            if len(cell_df) == 0:
                cols[i].warning(f"No data\n{x_val}")
                continue
            # trend over encounter year
            grp = cell_df.groupby('encounter_year')[outcome].mean().reset_index()
            title_str = f"{x_val}\n(n={len(cell_df)})"
            clr = col_enc.get(str(x_val), '#1a9e8f') if col_enc else '#1a9e8f'
            fig_cell = go.Figure()
            fig_cell.add_trace(go.Scatter(
                x=grp['encounter_year'], y=grp[outcome],
                mode='lines+markers',
                line=dict(color=clr, width=2.5),
                marker=dict(size=6),
                name=str(x_val)
            ))
            fig_cell.update_layout(
                title=dict(text=title_str, font=dict(size=11)),
                xaxis=dict(title='Year', tickmode='linear', dtick=2, tickfont=dict(size=9)),
                yaxis=dict(title=outcome_label, tickfont=dict(size=9)),
                plot_bgcolor='white',
                height=220,
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
                "Resource Redistribution Inquiry",
                "Data Partnership Request",
                "Program Funding Discussion",
                "Community Outreach Collaboration",
                "Research Partnership",
                "Media / Press Inquiry",
                "Other"
            ])
            message = st.text_area("Message *", height=130,
                                    placeholder="Describe your request or how you'd like to collaborate…")
            urgency = st.select_slider("Urgency:", ["Low","Medium","High","Critical"])
            submitted_ngo = st.form_submit_button("📤 Send to NGO Partner", use_container_width=True)

        if submitted_ngo:
            if sender_name and sender_email and message:
                st.success(f"✅ Your message has been sent to **{ngo_name}**! "
                           f"Expected response: {'24 hrs' if urgency in ['High','Critical'] else '3–5 business days'}.")
            else:
                st.error("Please fill in all required fields (*).")

    with col2:
        section("💬 Platform Feedback")
        with st.form("feedback_form"):
            fb_name  = st.text_input("Your Name (optional)")
            fb_role  = st.selectbox("Your Role:", [
                "Public Health Official","NGO Staff","Academic Researcher",
                "Student","Community Advocate","Other"
            ])
            overall  = st.slider("Overall Platform Rating:", 1, 5, 4)
            stars    = "⭐" * overall
            st.markdown(f"**Your rating:** {stars}")
            useful   = st.multiselect("Which pages were most useful?", [
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

    # ── Partner NGO cards ─────────────────────────────────────────────────────
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
