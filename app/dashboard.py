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

    # גרף 1 - ממוצע לפי חודש
    month_names = {1:'ינואר',2:'פברואר',3:'מרץ',4:'אפריל',5:'מאי',6:'יוני',
                   7:'יולי',8:'אוגוסט',9:'ספטמבר',10:'אוקטובר',11:'נובמבר',12:'דצמבר'}
    monthly = filtered.groupby('חודש_התחלה')['סה״כ ימים בהפרחה'].mean().reset_index()
    monthly['חודש'] = monthly['חודש_התחלה'].map(month_names)
    fig1 = px.bar(monthly, x='חודש', y='סה״כ ימים בהפרחה',
                  title="ממוצע ימי הפרחה לפי חודש כניסה",
                  color='סה״כ ימים בהפרחה', color_continuous_scale='RdYlGn_r',
                  labels={'סה״כ ימים בהפרחה': 'ממוצע ימי הפרחה', 'חודש': 'חודש כניסה להפרחה'})
    fig1.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)

    # גרף 2 - ממוצע לפי חממה
    with col1:
        gh_perf = filtered.groupby('חממה')['סה״כ ימים בהפרחה'].agg(['mean','count']).reset_index()
        gh_perf.columns = ['חממה', 'ממוצע ימים', 'מספר אצוות']
        gh_perf = gh_perf.sort_values('ממוצע ימים')
        fig2 = px.bar(gh_perf, x='חממה', y='ממוצע ימים',
                      title="ממוצע ימי הפרחה לפי חממה",
                      color='ממוצע ימים', color_continuous_scale='RdYlGn_r',
                      hover_data=['מספר אצוות'],
                      labels={'ממוצע ימים': 'ממוצע ימי הפרחה', 'חממה': 'חממה'})
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # גרף 3 - הזנים עם הכי הרבה אצוות
    with col2:
        strain_perf = filtered.groupby('זן')['סה״כ ימים בהפרחה'].agg(['mean','count']).reset_index()
        strain_perf.columns = ['זן', 'ממוצע ימים', 'מספר אצוות']
        strain_perf = strain_perf[strain_perf['מספר אצוות'] >= 3].sort_values('ממוצע ימים')
        fig3 = px.bar(strain_perf, x='זן', y='ממוצע ימים',
                      title="ממוצע ימי הפרחה לפי זן (מינימום 3 אצוות)",
                      color='ממוצע ימים', color_continuous_scale='RdYlGn_r',
                      hover_data=['מספר אצוות'],
                      labels={'ממוצע ימים': 'ממוצע ימי הפרחה', 'זן': 'זן'})
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    # טבלה נקייה
    st.subheader("📋 טבלת נתונים")
    cols_show = ['מספר אצווה','זן','חממה','תאריך תחילת הפרחה','סה״כ ימים בהפרחה']
    if 'עונה' in filtered.columns:
        cols_show.append('עונה')
    display_df = filtered[cols_show].copy()
    display_df['סה״כ ימים בהפרחה'] = display_df['סה״כ ימים בהפרחה'].round(1)
    display_df.columns = ['מספר אצווה','זן','חממה','תאריך התחלה','ימי הפרחה'] + (['עונה'] if 'עונה' in filtered.columns else [])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif page == "📅 גאנט":
    st.title("📅 גאנט הפרחה")
    st.markdown("---")

    df['תאריך תחילת הפרחה'] = pd.to_datetime(df['תאריך תחילת הפרחה'], errors='coerce')
    df['תאריך סיום'] = df['תאריך תחילת הפרחה'] + pd.to_timedelta(df['סה״כ ימים בהפרחה'], unit='D')
    df_valid = df.dropna(subset=['תאריך תחילת הפרחה','תאריך סיום'])

    # פילטרים
    col1, col2, col3 = st.columns(3)
    with col1:
        all_gh = sorted(df_valid['חממה'].unique())
        selected_gh_gantt = st.multiselect("סנן לפי חממה", all_gh, default=all_gh)
    with col2:
        all_strains = sorted(df_valid['זן'].unique())
        selected_strain = st.multiselect("סנן לפי זן", all_strains, default=[])
    with col3:
        n_batches = st.slider("מספר אצוות להצגה", 10, 100, 50)

    # פילטור
    filtered_gantt = df_valid[df_valid['חממה'].isin(selected_gh_gantt)]
    if selected_strain:
        filtered_gantt = filtered_gantt[filtered_gantt['זן'].isin(selected_strain)]
    filtered_gantt = filtered_gantt.tail(n_batches)

    st.markdown(f"**מציג {len(filtered_gantt)} אצוות**")

    # גאנט
    fig = px.timeline(
        filtered_gantt,
        x_start='תאריך תחילת הפרחה',
        x_end='תאריך סיום',
        y='חממה',
        color='זן',
        title="גאנט אצוות הפרחה",
        hover_data={
            'מספר אצווה': True,
            'זן': True,
            'חממה': True,
            'סה״כ ימים בהפרחה': ':.1f',
            'תאריך תחילת הפרחה': True,
            'תאריך סיום': True
        },
        labels={
            'תאריך תחילת הפרחה': 'תאריך כניסה',
            'תאריך סיום': 'תאריך קציר',
            'סה״כ ימים בהפרחה': 'ימי הפרחה'
        }
    )
    fig.update_yaxes(categoryorder='category ascending')
    fig.update_layout(
        height=550,
        xaxis_title="תאריך",
        yaxis_title="חממה",
        legend_title="זן",
        hoverlabel=dict(bgcolor="white", font_size=13)
    )
    # קו אנכי - היום
    today_str = datetime.today().strftime('%Y-%m-%d')
    fig.add_vline(
        x=today_str,
        line_dash="dash",
        line_color="red"
    )
    # הגבלת ציר X עד היום + חודש
    from datetime import timedelta
    end_range = (datetime.today() + timedelta(days=30)).strftime('%Y-%m-%d')
    start_range = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')
    fig.update_xaxes(range=[start_range, end_range])
    st.plotly_chart(fig, use_container_width=True)

    # סטטיסטיקה
    st.markdown("---")
    st.subheader("📊 סיכום גאנט")
    col1, col2, col3 = st.columns(3)
    with col1:
        active = filtered_gantt[filtered_gantt['תאריך סיום'] >= datetime.today()]
        st.metric("אצוות פעילות", len(active))
    with col2:
        completed = filtered_gantt[filtered_gantt['תאריך סיום'] < datetime.today()]
        st.metric("אצוות שהסתיימו", len(completed))
    with col3:
        st.metric("ממוצע ימי הפרחה", f"{filtered_gantt['סה״כ ימים בהפרחה'].mean():.1f}")
