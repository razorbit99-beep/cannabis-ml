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

page = st.sidebar.radio("ניווט", ["🏠 דשבורד", "🔮 חיזוי אצווה", "🏆 המלצת חממה", "📋 שיבוץ אצוות", "📊 ניתוח נתונים", "📅 גאנט"])

if page == "📋 שיבוץ אצוות":
    st.title("📋 שיבוץ אצוות")
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
        st.subheader("➕ הוספת אצווה חדשה")
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
        st.subheader("🏆 חממות מומלצות לזן זה")
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
            color = "#2d6a4f" if row["ציון"]>=70 else "#e6a817" if row["ציון"]>=40 else "#c0392b"
            mark = " ← נבחרה" if row["חממה"]==new_gh else ""
            st.markdown(f'''<div style="background:{color};padding:8px 15px;border-radius:8px;color:white;margin:4px 0;">
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
        st.subheader("📋 כל האצוות")
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
        st.subheader("🔄 עדכון או מחיקת אצווה")
        batches_db = load_batches_db()
        if len(batches_db) > 0:
            batch_ids = batches_db['batch_id'].tolist()
            selected_batch = st.selectbox("בחר אצווה", batch_ids)
            action = st.radio("פעולה", ["מחיקה", "עדכון תאריך סיום"])
            
            if action == "מחיקה":
                if st.button("🗑️ מחק אצווה", type="primary"):
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
        <div style="background: linear-gradient(135deg, #1a472a, #2d6a4f); padding: 20px; 
                    border-radius: 12px; color: white; text-align: center; margin-bottom: 20px;">
            <h2>🏆 חממה מומלצת: {best['חממה']}</h2>
            <h3>ציון התאמה: {best['ציון התאמה']}/100</h3>
            <p>ניסיון: {best['ניסיון עם הזן']} אצוות | ממוצע: {best['ממוצע ימים']} ימים | {best['פנויה בתאריך']}</p>
        </div>
        """, unsafe_allow_html=True)

        # גרף ציונים
        fig = px.bar(results_df, x='חממה', y='ציון התאמה',
                     color='ציון התאמה', color_continuous_scale='RdYlGn',
                     title=f"ציון התאמה לפי חממה - זן {strain_rec}",
                     text='ציון התאמה')
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

        # טבלה מפורטת
        st.subheader("📋 פירוט לפי חממה")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

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

    # טעינת נתונים מ-Supabase + היסטוריה
    supabase_gantt = get_supabase()
    if supabase_gantt:
        try:
            res = supabase_gantt.table('batches').select('*').execute()
            db_df = pd.DataFrame(res.data)
            db_df = db_df.rename(columns={
                'batch_id':'מספר אצווה','strain':'זן','greenhouse':'חממה',
                'start_date':'תאריך תחילת הפרחה','end_date':'תאריך סיום',
                'total_days':'סה״כ ימים בהפרחה'
            })
            db_df['תאריך תחילת הפרחה'] = pd.to_datetime(db_df['תאריך תחילת הפרחה'], errors='coerce')
            db_df['תאריך סיום'] = pd.to_datetime(db_df['תאריך סיום'], errors='coerce')
            db_df['סוג'] = db_df.get('is_planned', False).apply(lambda x: '📋 מתוכנן' if x else '✅ היסטורי')
            df_valid = db_df.dropna(subset=['תאריך תחילת הפרחה','תאריך סיום'])
            st.success(f"✅ נטענו {len(df_valid)} אצוות מהמסד")
        except Exception as e:
            df['תאריך תחילת הפרחה'] = pd.to_datetime(df['תאריך תחילת הפרחה'], errors='coerce')
            df['תאריך סיום'] = df['תאריך תחילת הפרחה'] + pd.to_timedelta(df['סה״כ ימים בהפרחה'], unit='D')
            df_valid = df.dropna(subset=['תאריך תחילת הפרחה','תאריך סיום'])
    else:
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
    # אצוות מתוכננות תמיד מוצגות
    if 'is_planned' in filtered_gantt.columns:
        planned = filtered_gantt[filtered_gantt['is_planned']==True]
        historical = filtered_gantt[filtered_gantt['is_planned']!=True].tail(n_batches)
        filtered_gantt = pd.concat([historical, planned]).drop_duplicates()
    else:
        filtered_gantt = filtered_gantt.tail(n_batches)

    show_future = st.checkbox("הצג אצוות עתידיות/מתוכננות", value=True)
    if not show_future:
        filtered_gantt = filtered_gantt[filtered_gantt['תאריך תחילת הפרחה'] <= pd.Timestamp.today()]
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
