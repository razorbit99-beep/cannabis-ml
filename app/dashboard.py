import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

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
</style>
""", unsafe_allow_html=True)

# טעינת נתונים - תומך בענן ובמקומי
@st.cache_data
def load_data():
    paths = [
        'app/training_dataset.csv',
        'data/processed/training_dataset.csv',
        'training_dataset.csv'
    ]
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

def get_season(month):
    if month in [12, 1, 2]: return 'חורף'
    if month in [3, 4, 5]: return 'אביב'
    if month in [6, 7, 8]: return 'קיץ'
    return 'סתיו'

df = load_data()

if df is None:
    st.error("לא נמצאו נתונים")
    st.stop()

# תפריט צד
st.sidebar.title("🌿 מערכת חיזוי הפרחה")
st.sidebar.markdown("---")
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
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    if 'עונה' in df.columns:
        st.subheader("📈 ממוצע ימי הפרחה לפי עונה")
        season_avg = df.groupby('עונה')['סה״כ ימים בהפרחה'].mean().reset_index()
        fig3 = px.bar(season_avg, x='עונה', y='סה״כ ימים בהפרחה',
                      color='עונה', title="ממוצע ימי הפרחה לפי עונה")
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
        hist = df[(df['חממה'] == greenhouse) & (df['זן'] == strain)]['סה״כ ימים בהפרחה']
        hist_gh = df[df['חממה'] == greenhouse]['סה״כ ימים בהפרחה']
        
        if len(hist) > 0:
            pred = hist.mean()
            source = f"מבוסס על {len(hist)} אצוות קודמות של זן זה בחממה זו"
        else:
            pred = hist_gh.mean()
            source = f"מבוסס על ממוצע חממה {greenhouse} ({len(hist_gh)} אצוות)"

        end_date = datetime.combine(start_date, datetime.min.time()) + timedelta(days=pred)
        season = get_season(start_date.month)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ ימי הפרחה צפויים", f"{pred:.1f} ימים")
        with col2:
            st.metric("📅 תאריך קציר משוער", end_date.strftime("%d/%m/%Y"))
        with col3:
            st.metric("🌤️ עונה", season)

        st.info(source)

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
                      title="ממוצע ימי הפרחה לפי זן",
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

    fig = px.timeline(df_gantt,
                      x_start='תאריך תחילת הפרחה',
                      x_end='תאריך סיום',
                      y='חממה',
                      color='זן',
                      title="גאנט אצוות אחרונות",
                      hover_data=['מספר אצווה','סה״כ ימים בהפרחה'])
    fig.update_yaxes(categoryorder='category ascending')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
