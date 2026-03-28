import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="מערכת חיזוי הפרחה - קנאביס",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# RTL עיצוב
st.markdown("""
<style>
    body { direction: rtl; }
    .main { direction: rtl; }
    .stMetric { direction: rtl; }
    h1, h2, h3 { text-align: right; }
    .stSelectbox label { direction: rtl; }
    .big-metric {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .alert-green { background: #d4edda; border-right: 4px solid #28a745; padding: 10px; border-radius: 5px; }
    .alert-yellow { background: #fff3cd; border-right: 4px solid #ffc107; padding: 10px; border-radius: 5px; }
    .alert-red { background: #f8d7da; border-right: 4px solid #dc3545; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    rf = joblib.load('models/rf_model.pkl')
    gb = joblib.load('models/gb_model.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    mapping = joblib.load('models/mapping.pkl')
    return rf, gb, feature_cols, mapping

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/training_dataset.csv')
    return df

def get_season(month):
    if month in [12, 1, 2]: return 'חורף'
    if month in [3, 4, 5]: return 'אביב'
    if month in [6, 7, 8]: return 'קיץ'
    return 'סתיו'

def predict_days(model, feature_cols, mapping, greenhouse, strain, start_date):
    season = get_season(start_date.month)
    greenhouses = mapping['חממות']
    strains = mapping['זנים']
    seasons = mapping['עונות']

    gh_code = greenhouses.index(greenhouse) if greenhouse in greenhouses else 0
    strain_code = strains.index(strain) if strain in strains else 0
    season_code = seasons.index(season) if season in seasons else 0

    df = pd.read_csv('data/processed/training_dataset.csv')
    sensor_means = {}
    sensor_cols = [c for c in feature_cols if '_mean' in c or '_std' in c]
    for col in sensor_cols:
        gh_data = df[df['חממה'] == greenhouse][col]
        sensor_means[col] = gh_data.mean() if not gh_data.empty else df[col].mean()

    row = {
        'חממה_קוד': gh_code,
        'זן_קוד': strain_code,
        'עונה_קוד': season_code,
        'חודש_התחלה': start_date.month,
        **sensor_means
    }
    X = pd.DataFrame([row])[feature_cols]
    pred = model.predict(X)[0]
    return round(pred, 1), season

# טעינה
try:
    rf, gb, feature_cols, mapping = load_models()
    df = load_data()
    models_loaded = True
except:
    models_loaded = False

# תפריט צד
st.sidebar.title("🌿 מערכת חיזוי הפרחה")
st.sidebar.markdown("---")
page = st.sidebar.radio("ניווט", ["🏠 דשבורד", "🔮 חיזוי אצווה", "📊 ניתוח נתונים", "📅 גאנט"])

# ===== דשבורד =====
if page == "🏠 דשבורד":
    st.title("🌿 מערכת חיזוי הפרחה - קנאביס")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("סה״כ אצוות", len(df), "היסטוריה")
    with col2:
        st.metric("ממוצע ימי הפרחה", f"{df['סה״כ ימים בהפרחה'].mean():.1f}", "ימים")
    with col3:
        st.metric("מספר זנים", df['זן'].nunique(), "זנים שונים")
    with col4:
        st.metric("מספר חממות", df['חממה'].nunique(), "חממות פעילות")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 ימי הפרחה לפי חממה")
        fig = px.box(df, x='חממה', y='סה״כ ימים בהפרחה',
                     color='חממה', title="התפלגות ימי הפרחה לפי חממה")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌱 הזנים הנפוצים ביותר")
        top_strains = df['זן'].value_counts().head(10)
        fig2 = px.bar(x=top_strains.values, y=top_strains.index,
                      orientation='h', title="10 זנים נפוצים",
                      color=top_strains.values, color_continuous_scale='Greens')
        fig2.update_layout(height=350, yaxis_title="זן", xaxis_title="מספר אצוות")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 ממוצע ימי הפרחה לפי עונה")
    season_avg = df.groupby('עונה')['סה״כ ימים בהפרחה'].mean().reset_index()
    fig3 = px.bar(season_avg, x='עונה', y='סה״כ ימים בהפרחה',
                  color='עונה', title="ממוצע ימי הפרחה לפי עונה",
                  color_discrete_map={'חורף':'#74b9ff','אביב':'#55efc4','קיץ':'#fdcb6e','סתיו':'#e17055'})
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# ===== חיזוי =====
elif page == "🔮 חיזוי אצווה":
    st.title("🔮 חיזוי משך הפרחה")
    st.markdown("---")

    if not models_loaded:
        st.error("המודלים לא נטענו. הרץ קודם את ml_model.py")
    else:
        col1, col2 = st.columns(2)
        with col1:
            greenhouse = st.selectbox("בחר חממה", mapping['חממות'])
            strain = st.selectbox("בחר זן", mapping['זנים'])
        with col2:
            start_date = st.date_input("תאריך כניסה להפרחה", datetime.today())

        if st.button("🔮 חשב חיזוי", use_container_width=True):
            pred_rf, season = predict_days(rf, feature_cols, mapping, greenhouse, strain, start_date)
            pred_gb, _ = predict_days(gb, feature_cols, mapping, greenhouse, strain, start_date)
            avg_pred = round((pred_rf + pred_gb) / 2, 1)
            end_date = datetime.combine(start_date, datetime.min.time()) + timedelta(days=avg_pred)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ ימי הפרחה צפויים", f"{avg_pred} ימים")
            with col2:
                st.metric("📅 תאריך קציר משוער", end_date.strftime("%d/%m/%Y"))
            with col3:
                st.metric("🌤️ עונה", season)

            # רמת ביטחון
            diff = abs(pred_rf - pred_gb)
            if diff < 2:
                st.markdown('<div class="alert-green">✅ רמת ביטחון גבוהה - שני המודלים מסכימים</div>', unsafe_allow_html=True)
            elif diff < 4:
                st.markdown('<div class="alert-yellow">⚠️ רמת ביטחון בינונית - פער של {:.1f} ימים בין המודלים</div>'.format(diff), unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-red">🔴 רמת ביטחון נמוכה - בדוק את הנתונים</div>', unsafe_allow_html=True)

            # השוואה להיסטוריה
            hist = df[(df['חממה'] == greenhouse) & (df['זן'] == strain)]['סה״כ ימים בהפרחה']
            if len(hist) > 0:
                st.markdown(f"**📚 היסטוריה:** {len(hist)} אצוות קודמות של זן זה בחממה זו | ממוצע: {hist.mean():.1f} ימים")

# ===== ניתוח =====
elif page == "📊 ניתוח נתונים":
    st.title("📊 ניתוח נתונים")
    st.markdown("---")

    selected_gh = st.multiselect("בחר חממות", df['חממה'].unique(), default=list(df['חממה'].unique())[:3])
    filtered = df[df['חממה'].isin(selected_gh)] if selected_gh else df

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(filtered, x='חודש_התחלה', y='סה״כ ימים בהפרחה',
                        color='חממה', title="ימי הפרחה לפי חודש התחלה",
                        trendline='ols')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        strain_perf = filtered.groupby('זן')['סה״כ ימים בהפרחה'].agg(['mean','std','count']).reset_index()
        strain_perf = strain_perf[strain_perf['count'] >= 3].sort_values('mean')
        fig2 = px.bar(strain_perf, x='זן', y='mean', error_y='std',
                      title="ממוצע ימי הפרחה לפי זן (מינימום 3 אצוות)",
                      color='mean', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 טבלת נתונים")
    st.dataframe(filtered[['מספר אצווה','זן','חממה','תאריך תחילת הפרחה','סה״כ ימים בהפרחה','עונה']],
                use_container_width=True)

# ===== גאנט =====
elif page == "📅 גאנט":
    st.title("📅 גאנט הפרחה")
    st.markdown("---")

    df['תאריך תחילת הפרחה'] = pd.to_datetime(df['תאריך תחילת הפרחה'], errors='coerce')
    df['תאריך סיום'] = df['תאריך תחילת הפרחה'] + pd.to_timedelta(df['סה״כ ימים בהפרחה'], unit='D')
    df_gantt = df.dropna(subset=['תאריך תחילת הפרחה','תאריך סיום']).tail(50)

    fig = px.timeline(df_gantt,
                      x_start='תאריך תחילת הפרחה',
                      x_end='תאריך סיום',
                      y='חממה',
                      color='זן',
                      title="גאנט אצוות אחרונות",
                      hover_data=['מספר אצווה','סה״כ ימים בהפרחה'])
    fig.update_yaxes(categoryorder='category ascending')
    fig.update_layout(height=500, xaxis_title="תאריך", yaxis_title="חממה")
    st.plotly_chart(fig, use_container_width=True)

