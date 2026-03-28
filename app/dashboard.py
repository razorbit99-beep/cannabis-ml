import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os
import joblib

st.set_page_config(
    page_title="מערכת חיזוי הפרחה - קנאביס",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body { direction: rtl; }
    .main { direction: rtl; }
    h1, h2, h3 { text-align: right; }
    .stMetric { direction: rtl; }
</style>
""", unsafe_allow_html=True)

def find_file(filename, folders=['app', 'models', '.']):
    for folder in folders:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None

@st.cache_data
def load_data():
    path = find_file('training_dataset.csv', ['app', 'data/processed', '.'])
    if path:
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_models():
    try:
        gb = joblib.load(find_file('gb_model.pkl'))
        feature_cols = joblib.load(find_file('feature_cols.pkl'))
        mapping = joblib.load(find_file('mapping.pkl'))
        return gb, feature_cols, mapping
    except:
        return None, None, None

def get_season(month):
    if month in [12, 1, 2]: return 'חורף'
    if month in [3, 4, 5]: return 'אביב'
    if month in [6, 7, 8]: return 'קיץ'
    return 'סתיו'

def predict_ml(model, feature_cols, mapping, df, greenhouse, strain, start_date):
    season = get_season(start_date.month)
    greenhouses = mapping['חממות']
    strains = mapping['זנים']
    seasons = mapping['עונות']

    gh_code = greenhouses.index(greenhouse) if greenhouse in greenhouses else 0
    strain_code = strains.index(strain) if strain in strains else 0
    season_code = seasons.index(season) if season in seasons else 0

    sensor_cols = [c for c in feature_cols if '_mean' in c or '_std' in c]
    sensor_means = {}
    for col in sensor_cols:
        gh_data = df[df['חממה'] == greenhouse][col] if col in df.columns else pd.Series()
        sensor_means[col] = gh_data.mean() if not gh_data.empty else 0

    row = {
        'חממה_קוד': gh_code,
        'זן_קוד': strain_code,
        'עונה_קוד': season_code,
        'חודש_התחלה': start_date.month,
        **sensor_means
    }
    X = pd.DataFrame([row])[feature_cols]
    return round(float(model.predict(X)[0]), 1), season

df = load_data()
gb, feature_cols, mapping = load_models()

if df is None:
    st.error("לא נמצאו נתונים")
    st.stop()

st.sidebar.title("🌿 מערכת חיזוי הפרחה")
st.sidebar.markdown("---")
if gb is not None:
    st.sidebar.success("✅ מודל ML פעיל")
else:
    st.sidebar.warning("⚠️ מצב ממוצע היסטורי")

page = st.sidebar.radio("ניווט", ["🏠 דשבורד", "🔮 חיזוי אצווה", "📊 ניתוח נתונים", "📅 גאנט"])

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
        fig = px.box(df, x='חממה', y='סה״כ ימים בהפרחה', color='חממה',
                     title="התפלגות ימי הפרחה לפי חממה")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌱 הזנים הנפוצים ביותר")
        top_strains = df['זן'].value_counts().head(10)
        fig2 = px.bar(x=top_strains.values, y=top_strains.index, orientation='h',
                      title="10 זנים נפוצים", color=top_strains.values,
                      color_continuous_scale='Greens')
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    if 'עונה' in df.columns:
        season_avg = df.groupby('עונה')['סה״כ ימים בהפרחה'].mean().reset_index()
        fig3 = px.bar(season_avg, x='עונה', y='סה״כ ימים בהפרחה', color='עונה',
                      title="ממוצע ימי הפרחה לפי עונה",
                      color_discrete_map={'חורף':'#74b9ff','אביב':'#55efc4','קיץ':'#fdcb6e','סתיו':'#e17055'})
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

elif page == "🔮 חיזוי אצווה":
    st.title("🔮 חיזוי משך הפרחה")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        greenhouse = st.selectbox("בחר חממה", sorted(df['חממה'].unique()))
        strain = st.selectbox("בחר זן", sorted(df['זן'].unique()))
    with col2:
        start_date = st.date_input("תאריך כניסה להפרחה", datetime.today())

    if st.button("🔮 חשב חיזוי", use_container_width=True):
        if gb is not None:
            pred, season = predict_ml(gb, feature_cols, mapping, df, greenhouse, strain, start_date)
            method = "🤖 מודל ML (Gradient Boosting - דיוק 93%)"
        else:
            hist = df[(df['חממה'] == greenhouse) & (df['זן'] == strain)]['סה״כ ימים בהפרחה']
            pred = round(hist.mean() if len(hist) > 0 else df['סה״כ ימים בהפרחה'].mean(), 1)
            season = get_season(start_date.month)
            method = "📊 ממוצע היסטורי"

        end_date = datetime.combine(start_date, datetime.min.time()) + timedelta(days=pred)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ ימי הפרחה צפויים", f"{pred} ימים")
        with col2:
            st.metric("📅 תאריך קציר משוער", end_date.strftime("%d/%m/%Y"))
        with col3:
            st.metric("🌤️ עונה", season)

        st.info(f"שיטת חיזוי: {method}")

        hist = df[(df['חממה'] == greenhouse) & (df['זן'] == strain)]['סה״כ ימים בהפרחה']
        if len(hist) > 0:
            st.markdown(f"**📚 היסטוריה:** {len(hist)} אצוות קודמות | ממוצע: {hist.mean():.1f} ימים | סטיית תקן: {hist.std():.1f} ימים")

            fig = px.histogram(hist, title=f"התפלגות ימי הפרחה - זן {strain} בחממה {greenhouse}",
                             labels={'value': 'ימי הפרחה'}, color_discrete_sequence=['#2d6a4f'])
            fig.add_vline(x=pred, line_dash="dash", line_color="red",
                         annotation_text=f"חיזוי: {pred} ימים")
            st.plotly_chart(fig, use_container_width=True)

elif page == "📊 ניתוח נתונים":
    st.title("📊 ניתוח נתונים")
    st.markdown("---")

    selected_gh = st.multiselect("בחר חממות", sorted(df['חממה'].unique()),
                                  default=sorted(df['חממה'].unique())[:3])
    filtered = df[df['חממה'].isin(selected_gh)] if selected_gh else df

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(filtered, x='חודש_התחלה', y='סה״כ ימים בהפרחה',
                        color='חממה', title="ימי הפרחה לפי חודש התחלה")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        strain_perf = filtered.groupby('זן')['סה״כ ימים בהפרחה'].agg(['mean','std','count']).reset_index()
        strain_perf = strain_perf[strain_perf['count'] >= 3].sort_values('mean')
        fig2 = px.bar(strain_perf, x='זן', y='mean', error_y='std',
                      title="ממוצע ימי הפרחה לפי זן (מינימום 3 אצוות)",
                      color='mean', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 טבלת נתונים")
    cols = ['מספר אצווה','זן','חממה','תאריך תחילת הפרחה','סה״כ ימים בהפרחה']
    if 'עונה' in df.columns:
        cols.append('עונה')
    st.dataframe(filtered[cols], use_container_width=True)

elif page == "📅 גאנט":
    st.title("📅 גאנט הפרחה")
    st.markdown("---")

    df['תאריך תחילת הפרחה'] = pd.to_datetime(df['תאריך תחילת הפרחה'], errors='coerce')
    df['תאריך סיום'] = df['תאריך תחילת הפרחה'] + pd.to_timedelta(df['סה״כ ימים בהפרחה'], unit='D')
    df_gantt = df.dropna(subset=['תאריך תחילת הפרחה','תאריך סיום']).tail(60)

    fig = px.timeline(df_gantt, x_start='תאריך תחילת הפרחה', x_end='תאריך סיום',
                      y='חממה', color='זן', title="גאנט אצוות אחרונות",
                      hover_data=['מספר אצווה','סה״כ ימים בהפרחה'])
    fig.update_yaxes(categoryorder='category ascending')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
