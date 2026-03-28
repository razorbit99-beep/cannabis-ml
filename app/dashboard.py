import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os
import joblib
import os

# חיבור ל-Supabase
@st.cache_resource
def get_supabase():
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", "https://gcfqucqiyggpaeuuurtl.supabase.co")
        key = st.secrets.get("SUPABASE_KEY", "")
        if key:
            return create_client(url, key)
    except:
        pass
    return None

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
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { text-align: right; color: #1a3a1e; }
    .stMetric { direction: rtl; }
    .stMetric label { color: #2d6a4f !important; }
    .stMetric [data-testid="metric-container"] { 
        background: #ffffff;
        border: 1px solid #c8a951;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(200,169,81,0.15);
    }
    [data-testid="stSidebar"] { 
        background-color: #1a3a1e;
    }
    [data-testid="stSidebar"] * { color: #ffffff; }
    [data-testid="stSidebar"] .stRadio label { color: #ffffff !important; }
    .stButton > button {
        background: linear-gradient(135deg, #2d6a4f, #1a3a1e);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 8px 20px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #40916c, #2d6a4f);
        box-shadow: 0 4px 12px rgba(45,106,79,0.3);
    }
    .stSelectbox label { color: #2d6a4f !important; font-weight: 500; }
    .stMultiSelect label { color: #2d6a4f !important; font-weight: 500; }
    .stMultiSelect span[data-baseweb="tag"] { display: none !important; }
    .stRadio label { color: #2d6a4f !important; font-weight: 500; }
    .stSlider label { color: #2d6a4f !important; font-weight: 500; }
    .stTextInput label { color: #2d6a4f !important; font-weight: 500; }
    .stDateInput label { color: #2d6a4f !important; font-weight: 500; }
    .stCheckbox label { color: #2d6a4f !important; }
    .stDataFrame { border: 1px solid #c8a951; border-radius: 8px; }
    .stAlert { border-radius: 8px; }
    hr { border-color: #c8a951; opacity: 0.5; }
    .stInfo { background: #e8f5e9; border-left: 4px solid #2d6a4f; }
    .stSuccess { background: #e8f5e9; border-left: 4px solid #2d6a4f; }
</style>
""", unsafe_allow_html=True)

# לוגו וכותרת
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try:
        st.image('app/logo_white.png', width=120)
    except:
        st.write("🌿")
with col_title:
    st.markdown("""
    <div style="padding-top:15px">
        <h2 style="color:#c8a951;margin:0;font-family:serif;">My Green Fields</h2>
        <p style="color:#a8d5a2;margin:0;font-size:14px;">מערכת ניהול וחיזוי הפרחה</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<hr style='border-color:#2d6a4f;margin:10px 0 20px 0'>", unsafe_allow_html=True)

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


# בחירת שפה
TRANSLATIONS = {
    "he": {
        "title": "מערכת חיזוי הפרחה",
        "nav": "ניווט",
        "pages": ["🏠 דשבורד", "🔮 חיזוי אצווה", "📋 שיבוץ אצוות", "📊 ניתוח נתונים", "📅 גאנט"],
        "page_names": {"🏠 דשבורד": "דשבורד", "🔮 חיזוי אצווה": "חיזוי אצווה", "📋 שיבוץ אצוות": "שיבוץ אצוות", "📊 ניתוח נתונים": "ניתוח נתונים", "📅 גאנט": "גאנט"},
        "ml_active": "✅ מודל ML פעיל",
        "ml_inactive": "⚠️ מצב ממוצע היסטורי",
        "greenhouse_count": "מספר חממות",
        "strain_count": "מספר זנים",
        "avg_days": "ממוצע ימי הפרחה",
        "total_batches": "סה״כ אצוות",
        "language": "שפה / Language",
    },
    "en": {
        "title": "Flowering Prediction System",
        "nav": "Navigation",
        "pages": ["🏠 Dashboard", "🔮 Batch Prediction", "📋 Batch Assignment", "📊 Data Analysis", "📅 Gantt"],
        "page_names": {"🏠 Dashboard": "Dashboard", "🔮 Batch Prediction": "Batch Prediction", "📋 Batch Assignment": "Batch Assignment", "📊 Data Analysis": "Data Analysis", "📅 Gantt": "Gantt"},
        "ml_active": "✅ ML Model Active",
        "ml_inactive": "⚠️ Historical Average Mode",
        "greenhouse_count": "Greenhouses",
        "strain_count": "Strains",
        "avg_days": "Avg. Flowering Days",
        "total_batches": "Total Batches",
        "language": "שפה / Language",
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "he"

col_he, col_en = st.sidebar.columns(2)
with col_he:
    if st.button("🇮🇱 עברית", use_container_width=True):
        st.session_state.lang = "he"
        st.rerun()
with col_en:
    if st.button("🇺🇸 English", use_container_width=True):
        st.session_state.lang = "en"
        st.rerun()

lang = "עברית" if st.session_state.lang == "he" else "English"
lang_key = "he" if lang == "עברית" else "en"
T = TRANSLATIONS[lang_key]

st.sidebar.title("🌿 " + T["title"])
st.sidebar.markdown("---")
if gb is not None:
    st.sidebar.success(T["ml_active"])
else:
    st.sidebar.warning(T["ml_inactive"])

page_display = st.sidebar.radio(T["nav"], T["pages"])
# המר לעברית לצורך הלוגיקה
en_to_he = {
    "🏠 Dashboard": "🏠 דשבורד",
    "🔮 Batch Prediction": "🔮 חיזוי אצווה",
    "📋 Batch Assignment": "📋 שיבוץ אצוות",
    "📊 Data Analysis": "📊 ניתוח נתונים",
    "📅 Gantt": "📅 גאנט"
}
page = en_to_he.get(page_display, page_display)

if page == "📋 שיבוץ אצוות":
    st.subheader("שיבוץ אצוות")
    st.markdown("---")
    
    supabase = get_supabase()
    
    # טעינת אצוות מ-Supabase
    @st.cache_data(ttl=30)
    def load_batches_db():
        if supabase:
            try:
                res = supabase.table('batches').select('*').order('start_date', desc=True).execute()
                return pd.DataFrame(res.data)
            except:
                pass
        return df[['מספר אצווה','זן','חממה','תאריך תחילת הפרחה','תאריך סיום הפרחה','סה״כ ימים בהפרחה']].rename(
            columns={'מספר אצווה':'batch_id','זן':'strain','חממה':'greenhouse',
                     'תאריך תחילת הפרחה':'start_date','תאריך סיום הפרחה':'end_date','סה״כ ימים בהפרחה':'total_days'})

    tab1, tab2, tab3 = st.tabs(["➕ הוספת אצווה", "📋 אצוות קיימות", "🔄 עדכון/מחיקה"])
    
    with tab1:
        st.subheader("הוספת אצווה חדשה")
        col1, col2, col3 = st.columns(3)
        with col1:
            all_strains_list = sorted(df['זן'].unique().tolist())
            strain_search = st.text_input("🔍 חיפוש זן", placeholder="הקלד שם זן...")
            if strain_search:
                filtered_strains = [s for s in all_strains_list if strain_search.upper() in s.upper()]
            else:
                filtered_strains = all_strains_list
            
            if filtered_strains:
                new_strain = st.selectbox("בחר זן", filtered_strains, key='new_strain')
            else:
                st.warning("לא נמצא זן - תוכל להוסיף זן חדש למטה")
                new_strain = strain_search.upper()
            
            # הוספת זן חדש
            with st.expander("➕ הוסף זן חדש"):
                new_strain_name = st.text_input("שם הזן החדש (עד 5 תווים)", max_chars=5).upper()
                if new_strain_name and st.button("הוסף זן"):
                    new_strain = new_strain_name
                    st.success(f"✅ זן {new_strain_name} יתווסף עם האצווה")
        with col2:
            new_gh = st.selectbox("חממה", sorted(df['חממה'].unique()), key='new_gh')
        with col3:
            new_date = st.date_input("תאריך כניסה", datetime.today(), key='new_date')
        
        # בדיקת זמינות
        batches_db = load_batches_db()
        if len(batches_db) > 0 and 'start_date' in batches_db.columns:
            batches_db['start_date'] = pd.to_datetime(batches_db['start_date'], errors='coerce')
            batches_db['end_date'] = pd.to_datetime(batches_db['end_date'], errors='coerce')
            target_dt = pd.Timestamp(new_date)
            active = batches_db[
                (batches_db['greenhouse'] == new_gh) &
                (batches_db['start_date'] <= target_dt) &
                (batches_db['end_date'] >= target_dt)
            ]
            if len(active) > 0:
                st.warning(f"⚠️ חממה {new_gh} תפוסה בתאריך זה! יש {len(active)} אצוות פעילות.")
                st.markdown("**💡 חממות חלופיות פנויות:**")
                for gh in sorted(df['חממה'].unique()):
                    if gh == new_gh: continue
                    busy = batches_db[
                        (batches_db['greenhouse'] == gh) &
                        (batches_db['start_date'] <= target_dt) &
                        (batches_db['end_date'] >= target_dt)
                    ]
                    if len(busy) == 0:
                        hist = df[(df['חממה']==gh)&(df['זן']==new_strain)]
                        exp = f" | {len(hist)} אצוות עם הזן" if len(hist)>0 else " | אין ניסיון עם הזן"
                        st.success(f"✅ חממה {gh} פנויה{exp}")
            else:
                st.success(f"✅ חממה {new_gh} פנויה בתאריך זה!")
        
        # חיזוי ימי הפרחה
        hist_match = df[(df['חממה']==new_gh)&(df['זן']==new_strain)]['סה״כ ימים בהפרחה']
        predicted_days = round(hist_match.mean() if len(hist_match)>0 else df['סה״כ ימים בהפרחה'].mean(), 1)
        end_date_pred = datetime.combine(new_date, datetime.min.time()) + timedelta(days=predicted_days)
        
        st.info(f"⏱️ חיזוי: {predicted_days} ימי הפרחה | תאריך קציר משוער: {end_date_pred.strftime('%d/%m/%Y')}")
        
        # המלצת חממות
        st.markdown("---")
        st.subheader("חממות מומלצות לזן זה")
        rec_results = []
        for gh_opt in sorted(df["חממה"].unique()):
            hist_opt = df[(df["חממה"]==gh_opt)&(df["זן"]==new_strain)]
            all_gh_opt = df[df["חממה"]==gh_opt]
            n = len(hist_opt)
            target_col = [c for c in df.columns if "ימים" in c][0]
            avg = hist_opt[target_col].mean() if n>0 else all_gh_opt[target_col].mean()
            std = hist_opt[target_col].std() if n>0 else all_gh_opt[target_col].std()
            if pd.isna(avg): avg=46
            if pd.isna(std): std=5
            exp_score = min(n*15, 40)
            stab_score = max(0, 30-std*2)
            batches_check = load_batches_db()
            if len(batches_check)>0 and "start_date" in batches_check.columns:
                batches_check["start_date"] = pd.to_datetime(batches_check["start_date"], errors="coerce")
                batches_check["end_date"] = pd.to_datetime(batches_check["end_date"], errors="coerce")
                busy = batches_check[
                    (batches_check["greenhouse"]==gh_opt) &
                    (batches_check["start_date"]<=pd.Timestamp(new_date)) &
                    (batches_check["end_date"]>=pd.Timestamp(new_date))
                ]
                avail_score = 30 if len(busy)==0 else 0
                avail_txt = "✅ פנויה" if len(busy)==0 else "❌ תפוסה"
            else:
                avail_score = 30
                avail_txt = "✅ פנויה"
            total = round(exp_score + stab_score + avail_score)
            rec_results.append({"חממה":gh_opt,"ניסיון":n,"ממוצע":round(avg,1),"זמינות":avail_txt,"ציון":total})
        rec_df = pd.DataFrame(rec_results).sort_values("ציון", ascending=False)
        for _, row in rec_df.head(5).iterrows():
            color = "#b8ddb8" if row["ציון"]>=70 else "#f5e6a0" if row["ציון"]>=40 else "#f5c0b8"
            mark = " ← נבחרה" if row["חממה"]==new_gh else ""
            st.markdown(f'''<div style="background:{color};padding:8px 15px;border-radius:8px;color:#1a3a1e;margin:4px 0;font-size:0.9em;">
            <b>חממה {row["חממה"]}{mark}</b> | {row["זמינות"]} | ניסיון: {row["ניסיון"]} | ממוצע: {row["ממוצע"]} ימים | ציון: {row["ציון"]}/100
            </div>''', unsafe_allow_html=True)
        st.markdown("---")
        week_num = new_date.isocalendar()[1]
        year_2 = str(new_date.year)[2:]
        strain_code = new_strain[:3].upper()
        batches_now = load_batches_db()
        existing = [b for b in (batches_now["batch_id"].tolist() if len(batches_now)>0 else []) 
                   if f"Z{new_gh}" in str(b) and year_2 in str(b)]
        next_num = len(existing) + 1
        auto_batch_id = f"G{strain_code}{year_2}{week_num:02d}Z{new_gh}{next_num}"
        st.info(f"מספר אצווה: **{auto_batch_id}**")
        st.caption(f"G=חווה | {strain_code}=זן | {year_2}=שנה | {week_num:02d}=שבוע | Z{new_gh}=חממה | {next_num}=סידורי")
        new_batch_id = st.text_input("שינוי ידני (אופציונלי)", value=auto_batch_id)
        
        if st.button("➕ שבץ אצווה", use_container_width=True):
            if supabase:
                try:
                    record = {
                        'batch_id': new_batch_id,
                        'strain': new_strain,
                        'greenhouse': new_gh,
                        'start_date': str(new_date),
                        'end_date': str(end_date_pred.date()),
                        'total_days': predicted_days,
                        'season': get_season(new_date.month),
                        'is_planned': True
                    }
                    supabase.table('batches').upsert(record, on_conflict='batch_id').execute()
                    st.success(f"✅ האצווה שובצה בהצלחה! חממה {new_gh} | {new_date} → {end_date_pred.strftime('%d/%m/%Y')}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"שגיאה: {e}")
            else:
                st.error("אין חיבור למסד נתונים")
    
    with tab2:
        st.subheader("כל האצוות")
        batches_db = load_batches_db()
        if len(batches_db) > 0:
            show_planned = st.checkbox("הצג רק מתוכננות", value=False)
            if show_planned and 'is_planned' in batches_db.columns:
                display = batches_db[batches_db['is_planned']==True]
            else:
                display = batches_db
            st.dataframe(display[['batch_id','strain','greenhouse','start_date','end_date','total_days']].head(50),
                        use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("עדכון או מחיקת אצווה")
        batches_db = load_batches_db()
        if len(batches_db) > 0:
            batch_ids = batches_db['batch_id'].tolist()
            selected_batch = st.selectbox(
                "חיפוש אצווה",
                options=[""] + batch_ids,
                index=0,
                format_func=lambda x: "הכנס מספר אצווה..." if x == "" else x
            )
            action = st.radio("פעולה", ["מחיקה", "עדכון תאריך סיום"])
            
            if action == "מחיקה":
                if st.button("מחק אצווה", type="primary"):
                    if supabase:
                        try:
                            supabase.table('batches').delete().eq('batch_id', selected_batch).execute()
                            st.success(f"✅ האצווה {selected_batch} נמחקה!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"שגיאה: {e}")
            else:
                new_end = st.date_input("תאריך סיום חדש", datetime.today())
                if st.button("✏️ עדכן", type="primary"):
                    if supabase:
                        try:
                            supabase.table('batches').update({'end_date': str(new_end)}).eq('batch_id', selected_batch).execute()
                            st.success(f"✅ עודכן!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"שגיאה: {e}")

if page == "🏆 המלצת חממה":
    st.title("🏆 המלצת חממה חכמה")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        strain_rec = st.selectbox("בחר זן", sorted(df['זן'].unique()), key='rec_strain')
    with col2:
        target_date = st.date_input("תאריך כניסה מתוכנן", datetime.today(), key='rec_date')

    if st.button("🏆 מצא חממה מומלצת", use_container_width=True):
        st.markdown("---")

        # חישוב ביצועי כל חממה עם הזן הזה
        results = []
        all_gh = sorted(df['חממה'].unique())

        for gh in all_gh:
            # ביצועי הזן בחממה זו
            strain_in_gh = df[(df['חממה'] == gh) & (df['זן'] == strain_rec)]
            all_in_gh = df[df['חממה'] == gh]

            # מספר אצוות של הזן בחממה
            n_strain = len(strain_in_gh)

            # ממוצע ימי הפרחה של הזן בחממה
            if n_strain > 0:
                avg_days = strain_in_gh['סה״כ ימים בהפרחה'].mean()
                std_days = strain_in_gh['סה״כ ימים בהפרחה'].std()
                if pd.isna(std_days): std_days = 0
                experience_score = min(n_strain * 15, 40)  # ניסיון - עד 40 נקודות
                stability_score = max(0, 30 - std_days * 2)  # יציבות - עד 30 נקודות
            else:
                # אין ניסיון עם הזן - משתמשים בממוצע החממה
                avg_days = all_in_gh['סה״כ ימים בהפרחה'].mean() if len(all_in_gh) > 0 else 46
                std_days = all_in_gh['סה״כ ימים בהפרחה'].std() if len(all_in_gh) > 0 else 5
                if pd.isna(std_days): std_days = 0
                experience_score = 0
                stability_score = max(0, 20 - std_days * 2)

            # בדיקת זמינות בגאנט
            df_gantt_check = df.copy()
            df_gantt_check['תאריך תחילת הפרחה'] = pd.to_datetime(df_gantt_check['תאריך תחילת הפרחה'], errors='coerce')
            df_gantt_check['תאריך סיום'] = df_gantt_check['תאריך תחילת הפרחה'] + pd.to_timedelta(df_gantt_check['סה״כ ימים בהפרחה'], unit='D')
            target_dt = pd.Timestamp(target_date)
            active_in_gh = df_gantt_check[
                (df_gantt_check['חממה'] == gh) &
                (df_gantt_check['תאריך תחילת הפרחה'] <= target_dt) &
                (df_gantt_check['תאריך סיום'] >= target_dt)
            ]
            is_available = len(active_in_gh) == 0
            availability_score = 30 if is_available else 0

            # ציון כולל
            total_score = experience_score + stability_score + availability_score

            results.append({
                'חממה': gh,
                'ניסיון עם הזן': n_strain,
                'ממוצע ימים': round(avg_days, 1) if not pd.isna(avg_days) else 46,
                'יציבות': round(std_days, 1) if not pd.isna(std_days) else 5,
                'פנויה בתאריך': '✅ כן' if is_available else '❌ תפוסה',
                'ציון התאמה': round(total_score),
            })

        results_df = pd.DataFrame(results).sort_values('ציון התאמה', ascending=False)

        # הצגת המלצה ראשית
        best = results_df.iloc[0]
        st.markdown(f"""
        <div style="background: #e8f5e9; padding: 20px; border-radius: 12px; border: 1px solid #2d6a4f; text-align: center; margin-bottom: 20px;">
            <h2 style="color:#1a3a1e;font-size:1.3em;">חממה מומלצת: {best['חממה']}</h2>
            <h3 style="color:#2d6a4f;font-size:1.1em;">ציון התאמה: {best['ציון התאמה']}/100</h3>
            <p style="color:#333;font-size:0.9em;">ניסיון: {best['ניסיון עם הזן']} אצוות | ממוצע: {best['ממוצע ימים']} ימים | {best['פנויה בתאריך']}</p>
        </div>
        """, unsafe_allow_html=True)

        # גרף ציונים
        fig = px.bar(results_df, x='חממה', y='ציון התאמה',
                     color='ציון התאמה', color_continuous_scale=['#f5c0b8','#f5e6a0','#b8ddb8'],
                     title=f"ציון התאמה לפי חממה - זן {strain_rec}",
                     text='ציון התאמה')
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 105], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=1.0, title_xanchor='right')
        st.plotly_chart(fig, use_container_width=True)

        # טבלה מפורטת
        st.subheader("פירוט לפי חממה")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

if page == "🏠 דשבורד":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(T.get("total_batches", "סה״כ אצוות"), len(df), "היסטוריה")
    with col2:
        st.metric(T.get("avg_days", "ממוצע ימי הפרחה"), f"{df['סה״כ ימים בהפרחה'].mean():.1f}", "ימים")
    with col3:
        st.metric(T.get("strain_count", "מספר זנים"), df['זן'].nunique(), "זנים שונים")
    with col4:
        st.metric(T.get("greenhouse_count", "מספר חממות"), df['חממה'].nunique(), "חממות פעילות")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Flowering Days by Greenhouse" if lang_key == "en" else "ימי הפרחה לפי חממה")
        fig = px.box(df, x='חממה', y='סה״כ ימים בהפרחה', color_discrete_sequence=['#a8c8e8','#b8ddb8','#f5c8a0','#d4a8b8','#c8c8e8','#f5e0a0','#a8d4d0','#e8c0b8','#c0d4a8','#d4c0e0'], color='חממה',
                     title="Flowering Days Distribution by Greenhouse" if lang_key=="en" else "התפלגות ימי הפרחה לפי חממה")
        fig.update_layout(showlegend=False, height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=1.0, title_xanchor='right')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Most Common Strains" if lang_key == "en" else "הזנים הנפוצים ביותר")
        top_strains = df['זן'].value_counts().head(10)
        fig2 = px.bar(x=top_strains.values, y=top_strains.index, orientation='h',
                      title="Top 10 Strains" if lang_key=="en" else "זנים נפוצים - 10 הראשונים", color=top_strains.values,
                      color_continuous_scale=['#d4edd4','#74b474'], labels={'x': 'Batches' if lang_key=='en' else 'מספר אצוות', 'y': 'Strain' if lang_key=='en' else 'זן'})
        fig2.update_layout(height=350, coloraxis_showscale=False, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=1.0, title_xanchor='right')
        st.plotly_chart(fig2, use_container_width=True)

    if 'עונה' in df.columns:
        season_avg = df.groupby('עונה')['סה״כ ימים בהפרחה'].mean().reset_index()
        fig3 = px.bar(season_avg, x='עונה', y='סה״כ ימים בהפרחה', color='עונה',
                      title="Avg Flowering Days by Season" if lang_key=="en" else "ממוצע ימי הפרחה לפי עונה",
                      color_discrete_map={'חורף':'#a8c8e8','אביב':'#b8ddb8','קיץ':'#f5c8a0','סתיו':'#d4a8b8'})
        fig3.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=1.0, title_xanchor='right')
        st.plotly_chart(fig3, use_container_width=True)

elif page == "🔮 חיזוי אצווה":
    st.header("חיזוי משך הפרחה")
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
    st.subheader("ניתוח נתונים")
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
                  color='סה״כ ימים בהפרחה', color_continuous_scale=['#d4edd4','#2d6a4f'],
                  labels={'סה״כ ימים בהפרחה': 'ממוצע ימי הפרחה', 'חודש': 'חודש כניסה להפרחה'})
    fig1.update_layout(coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=1.0, title_xanchor='right')
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)

    # גרף 2 - ממוצע לפי חממה
    with col1:
        gh_perf = filtered.groupby('חממה')['סה״כ ימים בהפרחה'].agg(['mean','count']).reset_index()
        gh_perf.columns = ['חממה', 'ממוצע ימים', 'מספר אצוות']
        gh_perf = gh_perf.sort_values('ממוצע ימים')
        fig2 = px.bar(gh_perf, x='חממה', y='ממוצע ימים',
                      title="ממוצע ימי הפרחה לפי חממה",
                      color='ממוצע ימים', color_continuous_scale=['#d4edd4','#2d6a4f'],
                      hover_data=['מספר אצוות'],
                      labels={'ממוצע ימים': 'ממוצע ימי הפרחה', 'חממה': 'חממה'})
        fig2.update_layout(coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=1.0, title_xanchor='right')
        st.plotly_chart(fig2, use_container_width=True)

    # גרף 3 - הזנים עם הכי הרבה אצוות
    with col2:
        strain_perf = filtered.groupby('זן')['סה״כ ימים בהפרחה'].agg(['mean','count']).reset_index()
        strain_perf.columns = ['זן', 'ממוצע ימים', 'מספר אצוות']
        strain_perf = strain_perf[strain_perf['מספר אצוות'] >= 3].sort_values('ממוצע ימים')
        fig3 = px.bar(strain_perf, x='זן', y='ממוצע ימים',
                      title="ממוצע ימי הפרחה לפי זן (מינימום 3 אצוות)",
                      color='ממוצע ימים', color_continuous_scale=['#d4edd4','#2d6a4f'],
                      hover_data=['מספר אצוות'],
                      labels={'ממוצע ימים': 'ממוצע ימי הפרחה', 'זן': 'זן'})
        fig3.update_layout(coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=1.0, title_xanchor='right')
        st.plotly_chart(fig3, use_container_width=True)

    # טבלה נקייה
    st.subheader("טבלת נתונים")
    cols_show = ['מספר אצווה','זן','חממה','תאריך תחילת הפרחה','סה״כ ימים בהפרחה']
    if 'עונה' in filtered.columns:
        cols_show.append('עונה')
    display_df = filtered[cols_show].copy()
    display_df['סה״כ ימים בהפרחה'] = display_df['סה״כ ימים בהפרחה'].round(1)
    display_df.columns = ['מספר אצווה','זן','חממה','תאריך התחלה','ימי הפרחה'] + (['עונה'] if 'עונה' in filtered.columns else [])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif page == "📅 גאנט":
    st.subheader("גאנט הפרחה")
    st.markdown("---")

    supabase_gantt = get_supabase()
    if supabase_gantt:
        try:
            res = supabase_gantt.table('batches').select('*').order('start_date', desc=False).limit(500).execute()
            raw = pd.DataFrame(res.data)
            raw['start'] = pd.to_datetime(raw['start_date'], errors='coerce')
            raw['end'] = pd.to_datetime(raw['end_date'].astype(str).str[:10], format='%Y-%m-%d', errors='coerce')
            raw['זן'] = raw['strain']
            raw['חממה'] = raw['greenhouse']
            raw['מספר אצווה'] = raw['batch_id']
            raw['סה״כ ימים'] = raw['total_days']
            raw['סוג'] = raw['is_planned'].apply(lambda x: '📋 מתוכנן' if x else '✅ היסטורי')
            df_valid = raw.dropna(subset=['start','end']).copy()
            df_valid = df_valid[df_valid['start'] >= '2023-01-01']
            st.info(f"סה״כ {len(df_valid)} אצוות במסד")
        except Exception as e:
            st.error(f"שגיאה: {e}")
            df_valid = pd.DataFrame()
    else:
        df_valid = pd.DataFrame()

    today = pd.Timestamp.today()

    col1, col2, col3 = st.columns(3)
    with col1:
        all_gh = sorted(df_valid['חממה'].unique()) if len(df_valid)>0 else []
        selected_gh_gantt = st.multiselect("סנן לפי חממה", all_gh, default=all_gh)
    with col2:
        view_mode = st.radio("תצוגה", ["פעיל + עתידי", "הכל", "עבר בלבד"], horizontal=True, index=0)
    with col3:
        all_strains_g = sorted(df_valid['זן'].unique()) if len(df_valid)>0 else []
        selected_strain = st.multiselect("סנן לפי זן", all_strains_g, default=[])

    filtered_gantt = df_valid[df_valid['חממה'].isin(selected_gh_gantt)] if selected_gh_gantt else df_valid.copy()
    if selected_strain:
        filtered_gantt = filtered_gantt[filtered_gantt['זן'].isin(selected_strain)]

    if view_mode == "פעיל + עתידי":
        filtered_gantt = filtered_gantt[filtered_gantt['end'] >= today]
    elif view_mode == "עבר בלבד":
        filtered_gantt = filtered_gantt[filtered_gantt['end'] < today]

    st.markdown(f"**מציג {len(filtered_gantt)} אצוות**")

    if len(filtered_gantt) == 0:
        st.warning("אין אצוות להצגה בטווח זה")
    else:
        # יצירת שורות נפרדות לאצוות חופפות
        filtered_gantt = filtered_gantt.sort_values(['חממה','start'])
        filtered_gantt['שורה'] = ''
        for gh in filtered_gantt['חממה'].unique():
            mask = filtered_gantt['חממה'] == gh
            gh_batches = filtered_gantt[mask].copy()
            rows = []
            row_ends = []
            for _, batch in gh_batches.iterrows():
                placed = False
                for i, end in enumerate(row_ends):
                    if batch['start'] >= end:
                        rows.append(f"{gh}-{i+1}")
                        row_ends[i] = batch['end']
                        placed = True
                        break
                if not placed:
                    rows.append(f"{gh}-{len(row_ends)+1}")
                    row_ends.append(batch['end'])
            filtered_gantt.loc[mask, 'שורה'] = rows

        fig = px.timeline(
            filtered_gantt,
            x_start='start',
            x_end='end',
            y='שורה',
            color='זן',
            title="גאנט אצוות הפרחה",
            hover_data=['מספר אצווה','סוג'],
            color_discrete_sequence=['#a8c8e8','#b8ddb8','#f5c8a0','#d4a8b8','#c8c8e8','#f5e0a0','#a8d4d0','#e8c0b8','#c0d4a8','#d4c0e0']
        )
        today_str = datetime.today().strftime('%Y-%m-%d')
        fig.add_vline(x=today_str, line_dash="dash", line_color="#2d6a4f", line_width=1.5)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=1.0, title_xanchor='right', margin=dict(l=10,r=10,t=40,b=10))
        fig.update_yaxes(categoryorder='category ascending')
        fig.update_layout(height=550, xaxis_title='Date' if lang_key=='en' else 'תאריך', yaxis_title='Greenhouse' if lang_key=='en' else 'חממה', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=1.0, title_xanchor='right')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("סיכום")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("פעיל/עתידי", len(filtered_gantt[filtered_gantt['end'] >= today]))
        with col2:
            st.metric("הסתיים", len(filtered_gantt[filtered_gantt['end'] < today]))
        with col3:
            days_col = [c for c in filtered_gantt.columns if 'ימים' in str(c) or 'days' in str(c).lower()]
            if days_col:
                st.metric("ממוצע ימים", f"{filtered_gantt[days_col[0]].mean():.1f}")
            else:
                st.metric("ממוצע ימים", "N/A")