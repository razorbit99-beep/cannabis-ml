import streamlit as st
from supabase import create_client
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os
import joblib
import streamlit.components.v1 as components

st.set_page_config(
    page_title="מערכת חיזוי הפרחה - קנאביס",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global: hide Streamlit's top bar & footer on every screen ───────────────
st.markdown("""
<style>
[data-testid="stHeader"],
[data-testid="stToolbar"],
header { display: none !important; height: 0 !important; }
#MainMenu, footer { display: none !important; }
input { border-radius: 8px !important; }
.main .block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "org_verified" not in st.session_state:
    st.session_state.org_verified = False
if "org_id" not in st.session_state:
    st.session_state.org_id = None
if "org_name" not in st.session_state:
    st.session_state.org_name = None
if "user" not in st.session_state:
    st.session_state.user = None
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"   # "login" | "signup"
if "signup_step" not in st.session_state:
    st.session_state.signup_step = "org"   # "org" | "details"
if "signup_org" not in st.session_state:
    st.session_state.signup_org = None     # {"id":…, "name":…}

@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )

supabase = get_supabase()

# ─── Shared CSS for auth screens (org + login) ───────────────────────────────
_POPUP_CSS = """
<style>
.stApp {
    background: linear-gradient(160deg, #0a1f10 0%, #1a4429 55%, #0a1f10 100%) !important;
}
/* hide sidebar on auth screens */
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none !important;
}
/* target block container across all Streamlit versions */
.main .block-container,
[data-testid="stMain"] .block-container,
[data-testid="stMainBlockContainer"],
section.main > div.block-container,
div.block-container {
    max-width: 440px !important;
    width: 440px !important;
    padding: 10vh 16px 40px !important;
    margin: 0 auto !important;
}
[data-testid="stForm"] {
    background: #ffffff !important;
    border-radius: 20px !important;
    padding: 32px 32px 26px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12),
                0 12px 32px rgba(0,0,0,0.25),
                0 24px 80px rgba(0,0,0,0.45) !important;
    border: none !important;
    outline: none !important;
}
[data-testid="stForm"] input,
[data-testid="stForm"] .stTextInput,
[data-testid="stForm"] .stTextInput > div,
[data-testid="stForm"] .stTextInput > div > div {
    pointer-events: auto !important;
    cursor: text !important;
}
[data-testid="stForm"] .stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #d8e8de !important;
    padding: 10px 14px !important;
    font-size: 15px !important;
    background: #f8fcfa !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stForm"] .stTextInput > div > div > input:focus {
    border-color: #2d6a4f !important;
    box-shadow: 0 0 0 3px rgba(45,106,79,0.14) !important;
    background: #fff !important;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2d6a4f 0%, #1a3a1e 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    height: 46px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    width: 100% !important;
    margin-top: 8px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 6px 20px rgba(45,106,79,0.45) !important;
    transform: translateY(-1px) !important;
}
/* Lang toggle + secondary buttons on auth screens */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"] {
    height: 46px !important;
    font-size: 14px !important;
    border-radius: 12px !important;
    padding: 0 14px !important;
    font-weight: 600 !important;
    background: #1a4429 !important;
    color: #ffffff !important;
    border: none !important;
    transition: opacity 0.2s !important;
}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    opacity: 0.85 !important;
    background: #2d6a4f !important;
}
/* secondary action buttons (signup / switch org) */
.auth-secondary .stButton > button {
    height: 40px !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    background: #f0f7f3 !important;
    color: #2d6a4f !important;
    border: 1.5px solid #c0ddd0 !important;
    transition: all 0.2s ease !important;
}
.auth-secondary .stButton > button:hover {
    background: #e0f0e8 !important;
    border-color: #2d6a4f !important;
}
</style>
"""

def _lang_toggle():
    """Render a small EN / HE toggle row above the popup card."""
    active = st.session_state.lang
    st.markdown(f"""
    <div class="lang-row">
        <style>
        div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {{}}
        .lang-row .stButton:first-child > button {{
            {"background:#2d6a4f !important; color:#fff !important; border-color:#2d6a4f !important;" if active=="en" else ""}
        }}
        .lang-row .stButton:last-child > button {{
            {"background:#2d6a4f !important; color:#fff !important; border-color:#2d6a4f !important;" if active=="he" else ""}
        }}
        </style>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="lang-row">', unsafe_allow_html=True)
    _, c1, c2 = st.columns([3, 1.5, 1.5])
    with c1:
        if st.button("🇺🇸 English", use_container_width=True, key="lt_en"):
            st.session_state.lang = "en"
            st.rerun()
    with c2:
        if st.button("🇮🇱 עברית", use_container_width=True, key="lt_he"):
            st.session_state.lang = "he"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ─── DEMO BYPASS ─────────────────────────────────────────────────────────────
if st.query_params.get("demo") == "1" and st.session_state.user is None:
    st.session_state.user         = type("U", (), {"email": "demo@mygreenfieldss.com"})()
    st.session_state.org_verified = True
    st.session_state.org_id       = "demo"
    st.session_state.org_name     = "My Green Fields"
    st.rerun()

# ─── UNIFIED AUTH SCREEN ─────────────────────────────────────────────────────
if st.session_state.user is None:
    st.markdown(_POPUP_CSS, unsafe_allow_html=True)
    _lang_toggle()

    _lk  = st.session_state.lang
    _he  = _lk == "he"
    _dir = "rtl" if _he else "ltr"

    # ── Texts ────────────────────────────────────────────────────────────────
    T_AUTH = {
        "he": {
            "title":        "מערכת לחיזוי קנאביס",
            "email":        "אימייל",
            "password":     "סיסמה",
            "login_btn":    "כניסה",
            "to_signup":    "משתמש חדש? צור חשבון",
            "to_login":     "כבר יש לך חשבון? כניסה",
            "login_err":    "שגיאת התחברות",
            "org_code":     "קוד ארגון",
            "org_verify":   "אמת קוד ארגון",
            "org_bad":      "קוד ארגון לא תקין",
            "org_ok":       "אומת בהצלחה",
            "first_name":   "שם פרטי",
            "last_name":    "שם משפחה",
            "role":         "תפקיד בחברה",
            "signup_btn":   "צור חשבון",
            "signup_err":   "שגיאה ביצירת חשבון",
            "back":         "חזרה",
        },
        "en": {
            "title":        "Cannabis Prediction System",
            "email":        "Email",
            "password":     "Password",
            "login_btn":    "Login",
            "to_signup":    "New user? Create account",
            "to_login":     "Already have an account? Login",
            "login_err":    "Login failed",
            "org_code":     "Organization code",
            "org_verify":   "Verify org code",
            "org_bad":      "Invalid organization code",
            "org_ok":       "Verified successfully",
            "first_name":   "First name",
            "last_name":    "Last name",
            "role":         "Role in company",
            "signup_btn":   "Create account",
            "signup_err":   "Signup failed",
            "back":         "Back",
        }
    }
    t = T_AUTH[_lk]

    # ══ LOGIN MODE ════════════════════════════════════════════════════════════
    if st.session_state.auth_mode == "login":
        with st.form("login_form"):
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:18px; direction:{_dir};">
                <h2 style="color:#1a3a1e; font-weight:700; font-size:1.4em; margin:0;">
                    {t['title']}
                </h2>
            </div>
            """, unsafe_allow_html=True)
            email    = st.text_input(t["email"],    placeholder=t["email"],    label_visibility="collapsed")
            password = st.text_input(t["password"], placeholder=t["password"], type="password", label_visibility="collapsed")
            login_ok = st.form_submit_button(t["login_btn"], use_container_width=True)

            if login_ok:
                try:
                    res  = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    user = res.user
                    # Restore org from Supabase user metadata (set at signup)
                    meta   = user.user_metadata or {}
                    org_id = meta.get("org_id")
                    if org_id:
                        org_rows = supabase.table("organizations").select("*").eq("id", org_id).execute()
                        if org_rows.data:
                            st.session_state.org_verified = True
                            st.session_state.org_id       = org_id
                            st.session_state.org_name     = org_rows.data[0]["name"]
                    st.session_state.user = user
                    st.rerun()
                except Exception as e:
                    st.error(f"{t['login_err']}: {e}")

        if st.button(t["to_signup"], use_container_width=True, key="go_signup"):
            st.session_state.auth_mode  = "signup"
            st.session_state.signup_step = "org"
            st.session_state.signup_org  = None
            st.rerun()

    # ══ SIGNUP MODE ══════════════════════════════════════════════════════════
    else:
        # ── Step 1: verify org code ──────────────────────────────────────────
        if st.session_state.signup_step == "org":
            with st.form("signup_org_form"):
                st.markdown(f"""
                <div style="text-align:center; margin-bottom:18px; direction:{_dir};">
                    <h2 style="color:#1a3a1e; font-weight:700; font-size:1.4em; margin:0;">
                        {t['title']}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                org_input = st.text_input(t["org_code"], placeholder=t["org_code"], label_visibility="collapsed")
                verify_ok = st.form_submit_button(t["org_verify"], use_container_width=True)

                if verify_ok:
                    if not org_input.strip():
                        st.error(t["org_bad"])
                    else:
                        org_rows = (supabase.table("organizations")
                                    .select("*")
                                    .eq("org_code", org_input.strip())
                                    .execute())
                        if not org_rows.data:
                            st.error(t["org_bad"])
                        else:
                            st.session_state.signup_org  = {"id": org_rows.data[0]["id"],
                                                             "name": org_rows.data[0]["name"]}
                            st.session_state.signup_step = "details"
                            st.rerun()

            if st.button(t["to_login"], use_container_width=True, key="go_login_from_org"):
                st.session_state.auth_mode = "login"
                st.rerun()

        # ── Step 2: fill in user details ────────────────────────────────────
        else:
            org_name = st.session_state.signup_org["name"]
            with st.form("signup_details_form"):
                # Logo
                try:
                    _logo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo_white.png')
                    _, logo_mid, _ = st.columns([1, 2, 1])
                    with logo_mid:
                        st.image(_logo, use_container_width=True)
                except:
                    pass

                st.markdown(f"""
                <div style="text-align:center; margin-bottom:8px; direction:{_dir};">
                    <h2 style="color:#1a3a1e; font-weight:700; font-size:1.4em; margin:0 0 4px;">
                        {t['title']}
                    </h2>
                    <div style="display:inline-block; background:#e8f5ee; color:#2d6a4f;
                                border-radius:8px; padding:4px 14px; font-size:13px;
                                font-weight:600; margin-top:6px;">
                        ✓ {org_name}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                first_name = st.text_input(t["first_name"], placeholder=t["first_name"], label_visibility="collapsed")
                last_name  = st.text_input(t["last_name"],  placeholder=t["last_name"],  label_visibility="collapsed")
                role       = st.text_input(t["role"],       placeholder=t["role"],       label_visibility="collapsed")
                email      = st.text_input(t["email"],      placeholder=t["email"],      label_visibility="collapsed")
                password   = st.text_input(t["password"],   placeholder=t["password"],   type="password", label_visibility="collapsed")
                signup_ok  = st.form_submit_button(t["signup_btn"], use_container_width=True)

                if signup_ok:
                    if not all([first_name, last_name, email, password]):
                        st.error("יש למלא את כל השדות" if _he else "Please fill in all fields")
                    else:
                        try:
                            meta = {
                                "org_id":     st.session_state.signup_org["id"],
                                "org_name":   org_name,
                                "first_name": first_name,
                                "last_name":  last_name,
                                "role":       role,
                            }
                            res = supabase.auth.sign_up({
                                "email":    email,
                                "password": password,
                                "options":  {"data": meta}
                            })
                            st.session_state.user         = res.user
                            st.session_state.org_verified = True
                            st.session_state.org_id       = meta["org_id"]
                            st.session_state.org_name     = org_name
                            st.rerun()
                        except Exception as e:
                            st.error(f"{t['signup_err']}: {e}")

            if st.button(t["back"], use_container_width=True, key="back_to_org"):
                st.session_state.signup_step = "org"
                st.session_state.signup_org  = None
                st.rerun()

    st.stop()

def find_file(filename, folders=['app', 'models', '.']):
    # First: look relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        return path
    # Then: look relative to cwd
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
def load_models(_version=1):
    try:
        p1 = find_file('gb_model.pkl')
        p2 = find_file('feature_cols.pkl')
        p3 = find_file('mapping.pkl')
        if not p1 or not p2 or not p3:
            return None, None, None, None, None, None
        gb = joblib.load(p1)
        feature_cols = joblib.load(p2)
        mapping = joblib.load(p3)
        gb_thca, thca_feature_cols = None, None
        if mapping.get('has_thca_model'):
            p4 = find_file('gb_thca_model.pkl')
            p5 = find_file('thca_feature_cols.pkl')
            if p4 and p5:
                gb_thca = joblib.load(p4)
                thca_feature_cols = joblib.load(p5)
        # ממוצעים עונתיים (חממה × חודש) לניבוי מדויק
        seasonal = None
        p6 = find_file('seasonal_averages.pkl')
        if p6:
            seasonal = joblib.load(p6)
        return gb, feature_cols, mapping, gb_thca, thca_feature_cols, seasonal
    except Exception as e:
        return None, None, None, None, None, None

def get_season(month, lk='he'):
    if month in [12, 1, 2]: return 'Winter' if lk=='en' else 'חורף'
    if month in [3, 4, 5]: return 'Spring' if lk=='en' else 'אביב'
    if month in [6, 7, 8]: return 'Summer' if lk=='en' else 'קיץ'
    return 'Fall' if lk=='en' else 'סתיו'

def _get_sensor_means_for_prediction(feature_cols, df, greenhouse, start_date, seasonal=None):
    """
    מחזיר dict של ערכי חיישנים לניבוי.
    מעדיף ממוצעים עונתיים (seasonal) על פני ממוצע כללי מהדאטאסט.
    """
    sensor_cols = [c for c in feature_cols if '_mean' in c or '_std' in c]
    sensor_means = {}
    gh_letter = greenhouse[0].upper() if greenhouse else ''
    month = start_date.month

    if seasonal and (gh_letter, month) in seasonal:
        # ממוצעים עונתיים מדויקים (חממה × חודש)
        seas_data = seasonal[(gh_letter, month)]
        for col in sensor_cols:
            sensor_means[col] = seas_data.get(col, np.nan)
    else:
        # fallback: ממוצע כללי לפי חממה מנתוני אימון
        for col in sensor_cols:
            if col in df.columns:
                gh_data = df[df['חממה'] == greenhouse][col].dropna()
                sensor_means[col] = float(gh_data.mean()) if not gh_data.empty else np.nan
            else:
                sensor_means[col] = np.nan

    return sensor_means


def predict_ml(model, feature_cols, mapping, df, greenhouse, strain, start_date, seasonal=None):
    season = get_season(start_date.month, st.session_state.lang)
    greenhouses = mapping['חממות']
    strains = mapping['זנים']
    seasons = mapping['עונות']

    gh_code     = greenhouses.index(greenhouse) if greenhouse in greenhouses else 0
    strain_code = strains.index(strain)         if strain in strains         else 0
    season_code = seasons.index(season)         if season in seasons         else 0

    sensor_means = _get_sensor_means_for_prediction(feature_cols, df, greenhouse, start_date, seasonal)

    row = {
        'חממה_קוד':    gh_code,
        'זן_קוד':      strain_code,
        'עונה_קוד':    season_code,
        'חודש_התחלה': start_date.month,
        **sensor_means
    }
    # אם המודל אומן עם THCA כפיצ'ר — ממלאים עם ממוצע הזן
    if 'THCA' in feature_cols:
        if 'THCA' in df.columns:
            strain_thca = df[df['זן'] == strain]['THCA'].dropna()
            row['THCA'] = float(strain_thca.mean()) if not strain_thca.empty else float(df['THCA'].dropna().mean() if df['THCA'].notna().any() else 21.0)
        else:
            row['THCA'] = 21.0
    X = pd.DataFrame([row])[feature_cols]
    return round(float(model.predict(X)[0]), 1), season


def predict_thca(model, thca_feature_cols, mapping, df, greenhouse, strain, start_date, seasonal=None):
    season = get_season(start_date.month, st.session_state.lang)
    greenhouses = mapping['חממות']
    strains = mapping['זנים']
    seasons = mapping['עונות']

    gh_code     = greenhouses.index(greenhouse) if greenhouse in greenhouses else 0
    strain_code = strains.index(strain)         if strain in strains         else 0
    season_code = seasons.index(season)         if season in seasons         else 0

    sensor_means = _get_sensor_means_for_prediction(thca_feature_cols, df, greenhouse, start_date, seasonal)

    row = {
        'חממה_קוד':    gh_code,
        'זן_קוד':      strain_code,
        'עונה_קוד':    season_code,
        'חודש_התחלה': start_date.month,
        **sensor_means
    }
    X = pd.DataFrame([row])[thca_feature_cols]
    return round(float(model.predict(X)[0]), 1)

df = load_data()
gb, feature_cols, mapping, gb_thca, thca_feature_cols, seasonal = load_models(_version=1)
if gb is None:
    _dbg = find_file('gb_model.pkl')
    st.sidebar.error(f"model path: {_dbg}")

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
    params = st.query_params
    st.session_state.lang = params.get("lang", "en")

# ─── Main app CSS + JS (sidebar background + button colors) ──────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2b16 0%, #1a4429 100%) !important;
}
[data-testid="stSidebar"] * {
    color: #e8f5ee !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2d6a4f !important;
}
</style>
""", unsafe_allow_html=True)

# JS approach for sidebar buttons — CSS selectors alone don't override Streamlit 1.56 styles
_is_he = (st.session_state.lang == "he")
components.html(f"""
<script>
var isHebrew = {'true' if _is_he else 'false'};

function applyGreenButtons() {{
    try {{
        var doc = window.parent.document;
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (!sidebar) return;
        sidebar.querySelectorAll('button').forEach(function(btn) {{
            btn.style.setProperty('background-color', '#2d6a4f', 'important');
            btn.style.setProperty('color', '#ffffff', 'important');
            btn.style.setProperty('border', '1px solid #3d8a65', 'important');
            btn.style.setProperty('border-radius', '10px', 'important');
            btn.style.setProperty('font-weight', '600', 'important');
        }});
    }} catch(e) {{}}
}}

function moveSidebarRight() {{
    if (!isHebrew) return;
    try {{
        var doc  = window.parent.document;
        var app  = doc.querySelector('.stApp');
        var side = doc.querySelector('[data-testid="stSidebar"]');
        if (!app || !side) return;
        app.style.setProperty('flex-direction', 'row-reverse', 'important');
    }} catch(e) {{}}
}}

applyGreenButtons();
moveSidebarRight();
setTimeout(function(){{ applyGreenButtons(); moveSidebarRight(); }}, 500);
setTimeout(function(){{ applyGreenButtons(); moveSidebarRight(); }}, 1500);
var obs = new MutationObserver(function(){{ applyGreenButtons(); moveSidebarRight(); }});
obs.observe(window.parent.document.body, {{childList: true, subtree: true}});
</script>
""", height=1)

col_he, col_en = st.sidebar.columns(2)
with col_he:
    if st.button("🇮🇱 עברית", use_container_width=True):
        st.session_state.lang = "he"
        st.query_params["lang"] = "he"
        st.rerun()
with col_en:
    if st.button("🇺🇸 English", use_container_width=True):
        st.session_state.lang = "en"
        st.query_params["lang"] = "en"
        st.rerun()

lang = "עברית" if st.session_state.lang == "he" else "English"
lang_key = "he" if lang == "עברית" else "en"
T = TRANSLATIONS[lang_key]
if lang_key == "en":
    st.markdown("""<style>
    [data-testid="stHeadingWithActionElements"] { text-align: left !important; direction: ltr !important; }
    [data-testid="stHeadingWithActionElements"] > div { justify-content: flex-start !important; }
    .stMetric { direction: ltr !important; }
    .stMetric label { text-align: left !important; }
    label[data-testid="stWidgetLabel"] { text-align: left !important; direction: ltr !important; }
    .stRadio { direction: ltr !important; text-align: left !important; }
    .stRadio label { text-align: left !important; }
    .stSelectbox label { direction: ltr !important; text-align: left !important; }
    .stTextInput label { direction: ltr !important; text-align: left !important; }
    .stDateInput label { direction: ltr !important; text-align: left !important; }
    .stMultiSelect label { direction: ltr !important; text-align: left !important; }
    .stCheckbox label { direction: ltr !important; text-align: left !important; }
    p, li { direction: ltr !important; text-align: left !important; }
    input { direction: ltr !important; text-align: left !important; }
    [data-baseweb="select"] { direction: ltr !important; }
    [data-baseweb="input"] { direction: ltr !important; text-align: left !important; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
    /* ── RTL layout: sidebar on right ── */
    .stApp { flex-direction: row-reverse !important; }
    [data-testid="stSidebar"] { order: 2 !important; }
    [data-testid="stMain"]    { order: 1 !important; }

    /* ── Main content area RTL ── */
    [data-testid="stMainBlockContainer"],
    .main .block-container { direction: rtl !important; }

    /* ── Headings ── */
    [data-testid="stHeadingWithActionElements"] { text-align: right !important; direction: rtl !important; }
    [data-testid="stHeadingWithActionElements"] > div { justify-content: flex-end !important; }
    h1, h2, h3, h4, h5, h6 { direction: rtl !important; text-align: right !important; }

    /* ── Metric widgets ── */
    .stMetric { direction: rtl !important; }
    .stMetric label { text-align: right !important; }
    [data-testid="stMetricValue"] { text-align: right !important; }
    [data-testid="stMetricLabel"] { text-align: right !important; }

    /* ── Form widgets ── */
    label[data-testid="stWidgetLabel"] { text-align: right !important; direction: rtl !important; }
    .stRadio { direction: rtl !important; text-align: right !important; }
    .stRadio label { text-align: right !important; }
    .stSelectbox label { direction: rtl !important; text-align: right !important; }
    .stTextInput label { direction: rtl !important; text-align: right !important; }
    .stDateInput label { direction: rtl !important; text-align: right !important; }
    .stMultiSelect label { direction: rtl !important; text-align: right !important; }
    .stCheckbox label { direction: rtl !important; text-align: right !important; }
    .stSlider label { direction: rtl !important; text-align: right !important; }
    .stSelectSlider label { direction: rtl !important; text-align: right !important; }

    /* ── Paragraphs, lists, captions ── */
    p, li, caption, figcaption { direction: rtl !important; text-align: right !important; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] div { direction: rtl !important; text-align: right !important; }

    /* ── Inputs / base-web ── */
    input { direction: rtl !important; text-align: right !important; }
    [data-baseweb="select"] { direction: rtl !important; }
    [data-baseweb="input"] { direction: rtl !important; text-align: right !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { direction: rtl !important; }
    [data-testid="stSidebar"] * { text-align: right !important; }

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"] { direction: rtl !important; }
    table { direction: rtl !important; }
    th, td { text-align: right !important; }

    /* ── Tabs ── */
    [data-testid="stTabs"] button { direction: rtl !important; }

    /* ── Alerts / info boxes ── */
    [data-testid="stAlert"] { direction: rtl !important; text-align: right !important; }
    [data-testid="stAlert"] p { direction: rtl !important; text-align: right !important; }

    /* ── Columns: align content to right ── */
    [data-testid="column"] { direction: rtl !important; }
    </style>""", unsafe_allow_html=True)

# ─── Top-of-page header ──────────────────────────────────────────────────────
st.markdown("""
<style>
.mgf-header {
    background: #f0f2f6;
    border-radius: 12px;
    padding: 18px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}
.mgf-header-title {
    font-size: 2em;
    font-weight: 700;
    color: #c8a951;
    line-height: 1.2;
    margin: 0;
}
.mgf-header-sub {
    font-size: 13px;
    color: #2d6a4f;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

hdr_left, hdr_right = st.columns([5, 1])
with hdr_left:
    st.markdown("""
    <div class="mgf-header">
        <div>
            <div class="mgf-header-title">My Green Fields</div>
            <div class="mgf-header-sub">מערכת ניהול וחיזוי הפרחה</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with hdr_right:
    try:
        logo_path = find_file('logo_white.png')
        if logo_path:
            st.image(logo_path, width=90)
    except:
        pass

st.sidebar.markdown("---")
if gb is not None:
    st.sidebar.success(T["ml_active"])
else:
    st.sidebar.warning(T["ml_inactive"])

# שמור דף נוכחי ב-session
if "current_page_he" not in st.session_state:
    params = st.query_params
    st.session_state.current_page_he = params.get("page", "🏠 דשבורד")

he_to_en = {
    "🏠 דשבורד": "🏠 Dashboard",
    "🔮 חיזוי אצווה": "🔮 Batch Prediction",
    "📋 שיבוץ אצוות": "📋 Batch Assignment",
    "📊 ניתוח נתונים": "📊 Data Analysis",
    "📅 גאנט": "📅 Gantt"
}
en_to_he = {v: k for k, v in he_to_en.items()}

current_display = he_to_en.get(st.session_state.current_page_he, st.session_state.current_page_he) if lang_key=="en" else st.session_state.current_page_he
default_idx = T["pages"].index(current_display) if current_display in T["pages"] else 0

page_display = st.sidebar.radio(T["nav"], T["pages"], index=default_idx)
page = en_to_he.get(page_display, page_display)
st.session_state.current_page_he = page
st.query_params["page"] = page

if page == "📋 שיבוץ אצוות":
    if lang_key=="en":
        st.markdown('<h3 style="text-align:left">Batch Assignment</h3>', unsafe_allow_html=True)
    else:
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

    tab1, tab2, tab3 = st.tabs(["➕ Add Batch", "📋 Existing Batches", "🔄 Update/Delete"] if lang_key=="en" else ["➕ הוספת אצווה", "📋 אצוות קיימות", "🔄 עדכון/מחיקה"])

    with tab1:
        if lang_key=="en":
            st.markdown('<h3 style="text-align:left">Add New Batch</h3>', unsafe_allow_html=True)
        else:
            st.subheader("הוספת אצווה חדשה")
        col1, col2, col3 = st.columns(3)
        with col1:
            all_strains_list = sorted(df['זן'].unique().tolist())
            strain_search = st.text_input("🔍 Search Strain" if lang_key=="en" else "🔍 חיפוש זן", placeholder="Type strain name..." if lang_key=="en" else "הקלד שם זן...")
            if strain_search:
                filtered_strains = [s for s in all_strains_list if strain_search.upper() in s.upper()]
            else:
                filtered_strains = all_strains_list

            if filtered_strains:
                new_strain = st.selectbox("Select Strain" if lang_key=="en" else "בחר זן", filtered_strains, key='new_strain')
            else:
                st.warning("לא נמצא זן - תוכל להוסיף זן חדש למטה")
                new_strain = strain_search.upper()

            # הוספת זן חדש
            with st.expander("➕ Add New Strain" if lang_key=="en" else "➕ הוסף זן חדש"):
                new_strain_name = st.text_input("שם הזן החדש (עד 5 תווים)", max_chars=5).upper()
                if new_strain_name and st.button("הוסף זן"):
                    new_strain = new_strain_name
                    st.success(f"✅ זן {new_strain_name} יתווסף עם האצווה")
        with col2:
            new_gh = st.selectbox("Greenhouse" if lang_key=="en" else "חממה", sorted(df['חממה'].unique()), key='new_gh')
        with col3:
            new_date = st.date_input("Entry Date" if lang_key=="en" else "תאריך כניסה", datetime.today(), key='new_date')

        # ─── תהליכי מעבר בין אצוות ───────────────────────────────────────────────
        st.markdown("##### ⏱️ " + ("Transition periods between batches" if lang_key=="en" else "תקופות מעבר בין אצוות"))
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown("🌾 " + ("Harvest" if lang_key=="en" else "קציר"), help="תמיד יום אחד" if lang_key=="he" else "Always 1 day")
            st.info("1 " + ("day (fixed)" if lang_key=="en" else "יום (קבוע)"))
            harvest_days = 1
        with tc2:
            cleaning_days = st.select_slider(
                "🧹 " + ("Cleaning / Sanitation" if lang_key=="en" else "ניקוי / ביון"),
                options=[1, 2, 3], value=3, key='cleaning_days'
            )
        with tc3:
            transfer_days = st.select_slider(
                "🌱 " + ("Seedling Transfer" if lang_key=="en" else "העברת שתילים"),
                options=[1, 2, 3], value=3, key='transfer_days'
            )

        total_transition = harvest_days + cleaning_days + transfer_days
        st.caption(
            f"סה״כ מעבר: {harvest_days} קציר + {cleaning_days} ניקוי + {transfer_days} העברה = **{total_transition} ימים**"
            if lang_key=="he" else
            f"Total transition: {harvest_days} harvest + {cleaning_days} cleaning + {transfer_days} transfer = **{total_transition} days**"
        )

        # בדיקת זמינות
        batches_db = load_batches_db()
        cleaning_error = False
        if len(batches_db) > 0 and 'start_date' in batches_db.columns:
            batches_db['start_date'] = pd.to_datetime(batches_db['start_date'], errors='coerce')
            batches_db['end_date'] = pd.to_datetime(batches_db['end_date'], errors='coerce')
            target_dt = pd.Timestamp(new_date)

            # בדיקת חפיפה עם אצוות פעילות
            active = batches_db[
                (batches_db['greenhouse'] == new_gh) &
                (batches_db['start_date'] <= target_dt) &
                (batches_db['end_date'] >= target_dt)
            ]

            # בדיקת תקופת מעבר מלאה — קציר + ניקוי + העברה
            # שימוש ב-actual_end_date אם קיים (סיום מוקדם)
            gh_batches = batches_db[batches_db['greenhouse'] == new_gh].dropna(subset=['end_date']).copy()
            if 'actual_end_date' in gh_batches.columns:
                gh_batches['effective_end'] = gh_batches['actual_end_date'].combine_first(gh_batches['end_date'])
                gh_batches['effective_end'] = pd.to_datetime(gh_batches['effective_end'], errors='coerce')
            else:
                gh_batches['effective_end'] = pd.to_datetime(gh_batches['end_date'], errors='coerce')
            if 'actual_cleaning_days' in gh_batches.columns:
                gh_batches['eff_cleaning'] = gh_batches['actual_cleaning_days'].combine_first(gh_batches.get('cleaning_days', pd.Series([cleaning_days]*len(gh_batches))))
            else:
                gh_batches['eff_cleaning'] = cleaning_days
            if 'actual_transfer_days' in gh_batches.columns:
                gh_batches['eff_transfer'] = gh_batches['actual_transfer_days'].combine_first(gh_batches.get('transfer_days', pd.Series([transfer_days]*len(gh_batches))))
            else:
                gh_batches['eff_transfer'] = transfer_days
            past_batches = gh_batches[gh_batches['effective_end'] < target_dt]
            transition_blocked = False
            earliest_allowed = None
            last_end = None
            if len(past_batches) > 0:
                latest_idx = past_batches['effective_end'].idxmax()
                last_end = past_batches.loc[latest_idx, 'effective_end']
                eff_c = int(past_batches.loc[latest_idx, 'eff_cleaning']) if pd.notna(past_batches.loc[latest_idx, 'eff_cleaning']) else cleaning_days
                eff_t = int(past_batches.loc[latest_idx, 'eff_transfer']) if pd.notna(past_batches.loc[latest_idx, 'eff_transfer']) else transfer_days
                harvest_end   = last_end + timedelta(days=harvest_days)
                cleaning_end  = harvest_end + timedelta(days=eff_c)
                earliest_allowed = cleaning_end + timedelta(days=eff_t)
                if target_dt < earliest_allowed:
                    transition_blocked = True
                    # זיהוי איזה שלב חסום
                    if target_dt <= last_end:
                        blocked_stage = "🌾 עדיין בקציר" if lang_key=="he" else "🌾 Still in harvest"
                    elif target_dt <= harvest_end:
                        blocked_stage = "🌾 יום קציר" if lang_key=="he" else "🌾 Harvest day"
                    elif target_dt <= cleaning_end:
                        blocked_stage = "🧹 ניקוי / ביון" if lang_key=="he" else "🧹 Cleaning / Sanitation"
                    else:
                        blocked_stage = "🌱 העברת שתילים" if lang_key=="he" else "🌱 Seedling Transfer"

            if len(active) > 0:
                cleaning_error = True
                st.error(f"❌ חממה {new_gh} תפוסה בתאריך זה! יש {len(active)} אצוות פעילות.")
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
                        st.success(f"✅ Greenhouse {gh} available{exp}" if lang_key=="en" else f"✅ חממה {gh} פנויה{exp}")
            elif transition_blocked:
                cleaning_error = True
                st.error(
                    f"⛔ חממה {new_gh} בתהליך: **{blocked_stage}**\n\n"
                    f"סיום הפרחה הקודמת: {last_end.strftime('%d/%m/%Y')} | "
                    f"המועד המוקדם ביותר לכניסה: **{earliest_allowed.strftime('%d/%m/%Y')}**\n\n"
                    f"(קציר {harvest_days}י + ניקוי {cleaning_days}י + העברה {transfer_days}י)"
                    if lang_key=="he" else
                    f"⛔ Greenhouse {new_gh} in process: **{blocked_stage}**\n\n"
                    f"Last batch ended: {last_end.strftime('%d/%m/%Y')} | "
                    f"Earliest allowed entry: **{earliest_allowed.strftime('%d/%m/%Y')}**\n\n"
                    f"(harvest {harvest_days}d + cleaning {cleaning_days}d + transfer {transfer_days}d)"
                )
            else:
                if earliest_allowed is not None:
                    st.success(
                        f"✅ חממה {new_gh} פנויה — כל {total_transition} ימי המעבר הושלמו ✔"
                        if lang_key=="he" else
                        f"✅ Greenhouse {new_gh} available — all {total_transition} transition days satisfied ✔"
                    )
                else:
                    st.success(f"✅ Greenhouse {new_gh} available on this date!" if lang_key=="en" else f"✅ חממה {new_gh} פנויה בתאריך זה!")

        # חיזוי ימי הפרחה
        hist_match = df[(df['חממה']==new_gh)&(df['זן']==new_strain)]['סה״כ ימים בהפרחה']
        predicted_days = round(hist_match.mean() if len(hist_match)>0 else df['סה״כ ימים בהפרחה'].mean(), 1)
        end_date_pred = datetime.combine(new_date, datetime.min.time()) + timedelta(days=predicted_days)

        st.info(f"⏱️ Prediction: {predicted_days} flowering days | Est. harvest: {end_date_pred.strftime('%d/%m/%Y')}")

        # המלצת חממות
        st.markdown("---")
        if lang_key=="en":
            st.markdown('<h3 style="text-align:left">Recommended Greenhouses</h3>', unsafe_allow_html=True)
        else:
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
                avail_txt = ("✅ Available" if lang_key=="en" else "✅ פנויה") if len(busy)==0 else ("❌ Busy" if lang_key=="en" else "❌ תפוסה")
            else:
                avail_score = 30
                avail_txt = "✅ Available" if lang_key=="en" else "✅ פנויה"
            total = round(exp_score + stab_score + avail_score)
            rec_results.append({"חממה":gh_opt,"ניסיון":n,"ממוצע":round(avg,1),"זמינות":avail_txt,"ציון":total})
        rec_df = pd.DataFrame(rec_results).sort_values("ציון", ascending=False)
        for _, row in rec_df.head(5).iterrows():
            color = "#b8ddb8" if row["ציון"]>=70 else "#f5e6a0" if row["ציון"]>=40 else "#f5c0b8"
            mark = " ← Selected" if lang_key=="en" else " ← נבחרה" if row["חממה"]==new_gh else ""
            st.markdown(f'''<div style="background:{color};padding:8px 15px;border-radius:8px;color:#1a3a1e;margin:4px 0;font-size:0.9em;">
            <b>{"Greenhouse" if lang_key=="en" else "חממה"} {row["חממה"]}{mark}</b> | {row["זמינות"]} | {"Experience" if lang_key=="en" else "ניסיון"}: {row["ניסיון"]} | Avg: {row["ממוצע"]} days | Score: {row["ציון"]}/100
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
        st.info(f"Batch ID: **{auto_batch_id}**")
        st.caption(f"G=Farm | {strain_code}=Strain | {year_2}=Yr | {week_num:02d}=Wk | Z{new_gh}=GH | {next_num}=Seq" if lang_key=="en" else f"G=חווה | {strain_code}=זן | {year_2}=שנה | {week_num:02d}=שבוע | Z{new_gh}=חממה | {next_num}=סידורי")
        new_batch_id = st.text_input("Manual override (optional)" if lang_key=="en" else "שינוי ידני (אופציונלי)", value=auto_batch_id)

        if cleaning_error:
            st.button("➕ Assign Batch" if lang_key=="en" else "➕ שבץ אצווה", use_container_width=True, disabled=True)
            st.caption("⛔ לא ניתן לשבץ — יש לתקן את שגיאת הזמינות/ניקוי למעלה" if lang_key=="he" else "⛔ Cannot assign — fix the availability/cleaning error above")
        elif st.button("➕ Assign Batch" if lang_key=="en" else "➕ שבץ אצווה", use_container_width=True):
            if supabase:
                try:
                    record = {
                        'batch_id': new_batch_id,
                        'strain': new_strain,
                        'greenhouse': new_gh,
                        'start_date': str(new_date),
                        'end_date': str(end_date_pred.date()),
                        'total_days': predicted_days,
                        'season': get_season(new_date.month, lang_key),
                        'is_planned': True,
                        'harvest_days': int(harvest_days),
                        'cleaning_days': int(cleaning_days),
                        'transfer_days': int(transfer_days)
                    }
                    supabase.table('batches').upsert(record, on_conflict='batch_id').execute()
                    st.success(
                        f"✅ האצווה שובצה! חממה {new_gh} | {new_date} → {end_date_pred.strftime('%d/%m/%Y')} | "
                        f"מעבר: 🌾{harvest_days} 🧹{cleaning_days} 🌱{transfer_days} ימים"
                    )
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection error: {e}" if lang_key=="en" else f"שגיאה: {e}")
            else:
                st.error("אין חיבור למסד נתונים")

    with tab2:
        if lang_key=="en":
            st.markdown('<h3 style="text-align:left">All Batches</h3>', unsafe_allow_html=True)
        else:
            st.subheader("כל האצוות")
        batches_db = load_batches_db()
        if len(batches_db) > 0:
            show_planned = st.checkbox("Show planned only" if lang_key=="en" else "הצג רק מתוכננות", value=False)
            if show_planned and 'is_planned' in batches_db.columns:
                display = batches_db[batches_db['is_planned']==True]
            else:
                display = batches_db
            st.dataframe(display[['batch_id','strain','greenhouse','start_date','end_date','total_days']].head(50),
                        use_container_width=True, hide_index=True)

    with tab3:
        if lang_key=="en":
            st.markdown('<h3 style="text-align:left">Update or Delete Batch</h3>', unsafe_allow_html=True)
        else:
            st.subheader("עדכון או מחיקת אצווה")
        batches_db = load_batches_db()
        if len(batches_db) > 0:
            batch_ids = batches_db['batch_id'].tolist()
            selected_batch = st.selectbox(
                "Search Batch" if lang_key=="en" else "חיפוש אצווה",
                options=[""] + batch_ids,
                index=0,
                format_func=lambda x: ("Enter batch ID..." if lang_key=="en" else "הכנס מספר אצווה...") if x == "" else x
            )

            action_opts = ["מחיקה", "עדכון תאריך סיום", "✅ סיום מוקדם"] if lang_key!="en" else ["Delete", "Update End Date", "✅ Early Completion"]
            action = st.radio("Action" if lang_key=="en" else "פעולה", action_opts, horizontal=False)

            if action in ["מחיקה", "Delete"]:
                if st.button("Delete Batch" if lang_key=="en" else "מחק אצווה", type="primary"):
                    if supabase:
                        try:
                            supabase.table('batches').delete().eq('batch_id', selected_batch).execute()
                            st.success(f"✅ האצווה {selected_batch} נמחקה!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Connection error: {e}" if lang_key=="en" else f"שגיאה: {e}")

            elif action in ["עדכון תאריך סיום", "Update End Date"]:
                new_end = st.date_input("New End Date" if lang_key=="en" else "תאריך סיום חדש", datetime.today())
                if st.button("✏️ עדכן", type="primary"):
                    if supabase:
                        try:
                            supabase.table('batches').update({'end_date': str(new_end)}).eq('batch_id', selected_batch).execute()
                            st.success(f"✅ עודכן!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Connection error: {e}" if lang_key=="en" else f"שגיאה: {e}")

            elif action in ["✅ סיום מוקדם", "✅ Early Completion"] and selected_batch:
                # שליפת נתוני האצווה הנוכחית
                batch_row = batches_db[batches_db['batch_id'] == selected_batch]
                if len(batch_row) > 0:
                    b = batch_row.iloc[0]
                    planned_end  = pd.to_datetime(b.get('end_date'), errors='coerce')
                    planned_h    = int(b['harvest_days'])  if 'harvest_days'  in b.index and pd.notna(b.get('harvest_days'))  else 1
                    planned_c    = int(b['cleaning_days']) if 'cleaning_days' in b.index and pd.notna(b.get('cleaning_days')) else 1
                    planned_t    = int(b['transfer_days']) if 'transfer_days' in b.index and pd.notna(b.get('transfer_days')) else 1

                    st.markdown(f"**אצווה:** `{selected_batch}` | **סיום מתוכנן:** {planned_end.strftime('%d/%m/%Y') if pd.notna(planned_end) else '—'}")
                    st.markdown("---")
                    st.markdown("##### סמן את השלבים שהסתיימו מוקדם:")

                    ec1, ec2 = st.columns(2)
                    with ec1:
                        early_flowering = st.checkbox(
                            "🌿 הפרחה הסתיימה מוקדם" if lang_key=="he" else "🌿 Flowering ended early"
                        )
                        if early_flowering:
                            actual_end_date = st.date_input(
                                "תאריך סיום הפרחה בפועל" if lang_key=="he" else "Actual flowering end date",
                                value=planned_end.date() if pd.notna(planned_end) else datetime.today().date(),
                                key='actual_end_date_input'
                            )
                        else:
                            actual_end_date = None

                        early_harvest = st.checkbox(
                            f"🌾 קציר הסתיים מוקדם (מתוכנן: {planned_h}י)" if lang_key=="he"
                            else f"🌾 Harvest ended early (planned: {planned_h}d)"
                        )
                        if early_harvest:
                            actual_harvest = st.number_input(
                                "ימי קציר בפועל" if lang_key=="he" else "Actual harvest days",
                                min_value=1, max_value=planned_h, value=1, step=1, key='actual_harvest_input'
                            )
                        else:
                            actual_harvest = None

                    with ec2:
                        early_cleaning = st.checkbox(
                            f"🧹 ניקוי הסתיים מוקדם (מתוכנן: {planned_c}י)" if lang_key=="he"
                            else f"🧹 Cleaning ended early (planned: {planned_c}d)"
                        )
                        if early_cleaning:
                            actual_cleaning = st.number_input(
                                "ימי ניקוי בפועל" if lang_key=="he" else "Actual cleaning days",
                                min_value=1, max_value=planned_c, value=1, step=1, key='actual_cleaning_input'
                            )
                        else:
                            actual_cleaning = None

                        early_transfer = st.checkbox(
                            f"🌱 העברת שתילים הסתיימה מוקדם (מתוכנן: {planned_t}י)" if lang_key=="he"
                            else f"🌱 Seedling transfer ended early (planned: {planned_t}d)"
                        )
                        if early_transfer:
                            actual_transfer = st.number_input(
                                "ימי העברה בפועל" if lang_key=="he" else "Actual transfer days",
                                min_value=1, max_value=planned_t, value=1, step=1, key='actual_transfer_input'
                            )
                        else:
                            actual_transfer = None

                    # סיכום חיסכון
                    total_saved = 0
                    if early_flowering and actual_end_date and pd.notna(planned_end):
                        saved_fl = (planned_end.date() - actual_end_date).days
                        if saved_fl > 0:
                            st.info(f"🌿 חיסכון בהפרחה: **{saved_fl} ימים** מוקדם יותר" if lang_key=="he" else f"🌿 Flowering: **{saved_fl} days** earlier")
                            total_saved += saved_fl
                    if early_harvest and actual_harvest:
                        total_saved += planned_h - actual_harvest
                    if early_cleaning and actual_cleaning:
                        total_saved += planned_c - actual_cleaning
                    if early_transfer and actual_transfer:
                        total_saved += planned_t - actual_transfer
                    if total_saved > 0:
                        st.success(f"⚡ סה״כ חיסכון: **{total_saved} ימים** — החממה תתפנה מוקדם יותר!" if lang_key=="he" else f"⚡ Total saved: **{total_saved} days** — greenhouse available sooner!")

                    if st.button("💾 שמור סיום מוקדם" if lang_key=="he" else "💾 Save Early Completion", type="primary"):
                        if supabase:
                            try:
                                update_rec = {}
                                if early_flowering and actual_end_date:
                                    update_rec['actual_end_date'] = str(actual_end_date)
                                if early_harvest and actual_harvest:
                                    update_rec['actual_harvest_days'] = int(actual_harvest)
                                if early_cleaning and actual_cleaning:
                                    update_rec['actual_cleaning_days'] = int(actual_cleaning)
                                if early_transfer and actual_transfer:
                                    update_rec['actual_transfer_days'] = int(actual_transfer)
                                if update_rec:
                                    supabase.table('batches').update(update_rec).eq('batch_id', selected_batch).execute()
                                    st.success(f"✅ עודכן! {len(update_rec)} שדות נשמרו." if lang_key=="he" else f"✅ Saved! {len(update_rec)} fields updated.")
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.warning("לא נבחר אף שלב לעדכון" if lang_key=="he" else "No phase selected for update")
                            except Exception as e:
                                st.error(f"שגיאה: {e}")

if page == "🏆 המלצת חממה":
    st.title("🏆 המלצת חממה חכמה")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        strain_rec = st.selectbox("Select Strain" if lang_key=="en" else "בחר זן", sorted(df['זן'].unique()), key='rec_strain')
    with col2:
        target_date = st.date_input("תאריך כניסה מתוכנן", datetime.today(), key='rec_date')

    if st.button("🏆 מצא חממה מומלצת", use_container_width=True):
        st.markdown("---")

        results = []
        all_gh = sorted(df['חממה'].unique())

        for gh in all_gh:
            strain_in_gh = df[(df['חממה'] == gh) & (df['זן'] == strain_rec)]
            all_in_gh = df[df['חממה'] == gh]

            n_strain = len(strain_in_gh)

            if n_strain > 0:
                avg_days = strain_in_gh['סה״כ ימים בהפרחה'].mean()
                std_days = strain_in_gh['סה״כ ימים בהפרחה'].std()
                if pd.isna(std_days): std_days = 0
                experience_score = min(n_strain * 15, 40)
                stability_score = max(0, 30 - std_days * 2)
            else:
                avg_days = all_in_gh['סה״כ ימים בהפרחה'].mean() if len(all_in_gh) > 0 else 46
                std_days = all_in_gh['סה״כ ימים בהפרחה'].std() if len(all_in_gh) > 0 else 5
                if pd.isna(std_days): std_days = 0
                experience_score = 0
                stability_score = max(0, 20 - std_days * 2)

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

            total_score = experience_score + stability_score + availability_score

            results.append({
                'חממה': gh,
                'ניסיון עם הזן': n_strain,
                'ממוצע ימים': round(avg_days, 1) if not pd.isna(avg_days) else 46,
                'יציבות': round(std_days, 1) if not pd.isna(std_days) else 5,
                'פנויה בתאריך': ('✅ Yes' if lang_key=="en" else '✅ כן') if is_available else ('❌ No' if lang_key=="en" else '❌ תפוסה'),
                'ציון התאמה': round(total_score),
            })

        results_df = pd.DataFrame(results).sort_values('ציון התאמה', ascending=False)

        best = results_df.iloc[0]
        st.markdown(f"""
        <div style="background: #e8f5e9; padding: 20px; border-radius: 12px; border: 1px solid #2d6a4f; text-align: center; margin-bottom: 20px;">
            <h2 style="color:#1a3a1e;font-size:1.3em;">חממה מומלצת: {best['חממה']}</h2>
            <h3 style="color:#2d6a4f;font-size:1.1em;">ציון התאמה: {best['ציון התאמה']}/100</h3>
            <p style="color:#333;font-size:0.9em;">ניסיון: {best['ניסיון עם הזן']} אצוות | Avg: {best['ממוצע ימים']} ימים | {best['פנויה בתאריך']}</p>
        </div>
        """, unsafe_allow_html=True)

        fig = px.bar(results_df, x='חממה', y='ציון התאמה',
                     color='ציון התאמה', color_continuous_scale=['#f5c0b8','#f5e6a0','#b8ddb8'],
                     title=f"ציון התאמה לפי חממה - זן {strain_rec}",
                     text='ציון התאמה')
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 105], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=0.0 if lang_key=='en' else 1.0, title_xanchor='left' if lang_key=='en' else 'right')
        st.plotly_chart(fig, use_container_width=True)

        if lang_key=="en":
            st.markdown('<h3 style="text-align:left">Breakdown by Greenhouse</h3>', unsafe_allow_html=True)
        else:
            st.subheader("פירוט לפי חממה")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

if page == "🏠 דשבורד":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(T.get("total_batches", "סה״כ אצוות"), len(df), "Historical" if lang_key=="en" else "היסטוריה")
    with col2:
        st.metric(T.get("avg_days", "ממוצע ימי הפרחה"), f"{df['סה״כ ימים בהפרחה'].mean():.1f}", "Days" if lang_key=="en" else "ימים")
    with col3:
        st.metric(T.get("strain_count", "מספר זנים"), df['זן'].nunique(), "Different Strains" if lang_key=="en" else "זנים שונים")
    with col4:
        st.metric(T.get("greenhouse_count", "מספר חממות"), df['חממה'].nunique(), "Active Greenhouses" if lang_key=="en" else "חממות פעילות")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<h3 style="text-align:left">Flowering Days by Greenhouse</h3>' if lang_key=="en" else '### ימי הפרחה לפי חממה', unsafe_allow_html=lang_key=="en")
        fig = px.box(df, x='חממה', y='סה״כ ימים בהפרחה', labels={'חממה': 'Greenhouse' if lang_key=='en' else 'חממה', 'סה״כ ימים בהפרחה': 'Flowering Days' if lang_key=='en' else 'סה״כ ימים בהפרחה'}, color_discrete_sequence=['#a8c8e8','#b8ddb8','#f5c8a0','#d4a8b8','#c8c8e8','#f5e0a0','#a8d4d0','#e8c0b8','#c0d4a8','#d4c0e0'], color='חממה',
                     title="Flowering Days Distribution by Greenhouse" if lang_key=="en" else "התפלגות ימי הפרחה לפי חממה")
        fig.update_layout(showlegend=False, height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=0.0 if lang_key=='en' else 1.0, title_xanchor='left' if lang_key=='en' else 'right')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f'<h3 style="text-align:left">Most Common Strains</h3>' if lang_key=="en" else '### הזנים הנפוצים ביותר', unsafe_allow_html=lang_key=="en")
        top_strains = df['זן'].value_counts().head(10)
        fig2 = px.bar(x=top_strains.values, y=top_strains.index, orientation='h',
                      title="Top 10 Strains" if lang_key=="en" else "זנים נפוצים - 10 הראשונים", color=top_strains.values,
                      color_continuous_scale=['#d4edd4','#74b474'], labels={'x': 'Batches' if lang_key=='en' else 'מספר אצוות', 'y': 'Strain' if lang_key=='en' else 'זן'})
        fig2.update_layout(height=350, coloraxis_showscale=False, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=0.0 if lang_key=='en' else 1.0, title_xanchor='left' if lang_key=='en' else 'right')
        st.plotly_chart(fig2, use_container_width=True)

    if 'עונה' in df.columns:
        if lang_key == 'en':
            season_map = {'חורף': 'Winter', 'אביב': 'Spring', 'קיץ': 'Summer', 'סתיו': 'Fall'}
            df_seasons = df.copy()
            df_seasons['עונה'] = df_seasons['עונה'].map(season_map).fillna(df_seasons['עונה'])
        else:
            df_seasons = df
        season_avg = df_seasons.groupby('עונה')['סה״כ ימים בהפרחה'].mean().reset_index()
        fig3 = px.bar(season_avg, x='עונה', y='סה״כ ימים בהפרחה', labels={'עונה': 'Season' if lang_key=='en' else 'עונה', 'סה״כ ימים בהפרחה': 'Flowering Days' if lang_key=='en' else 'סה״כ ימים בהפרחה'}, color='עונה',
                      title="Avg Flowering Days by Season" if lang_key=="en" else "ממוצע ימי הפרחה לפי עונה",
                      color_discrete_map={'חורף':'#a8c8e8','אביב':'#b8ddb8','קיץ':'#f5c8a0','סתיו':'#d4a8b8'})
        fig3.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=0.0 if lang_key=='en' else 1.0, title_xanchor='left' if lang_key=='en' else 'right')
        st.plotly_chart(fig3, use_container_width=True)

elif page == "🔮 חיזוי אצווה":
    if lang_key=="en":
        st.markdown('<h2 style="text-align:left">Batch Duration Prediction</h2>', unsafe_allow_html=True)
    else:
        st.header("חיזוי משך הפרחה")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        greenhouse = st.selectbox("בחר חממה" if lang_key=="he" else "Select Greenhouse", sorted(df['חממה'].unique()))
        strain = st.selectbox("Select Strain" if lang_key=="en" else "בחר זן", sorted(df['זן'].unique()))
    with col2:
        start_date = st.date_input("Entry Date" if lang_key=="en" else "תאריך כניסה להפרחה", datetime.today())

    if st.button("Calculate Prediction" if lang_key=="en" else "חשב חיזוי", use_container_width=True):
        if gb is not None:
            pred, season = predict_ml(gb, feature_cols, mapping, df, greenhouse, strain, start_date, seasonal=seasonal)
            method = "🤖 מודל ML (Gradient Boosting - דיוק 93%)"
        else:
            hist = df[(df['חממה'] == greenhouse) & (df['זן'] == strain)]['סה״כ ימים בהפרחה']
            pred = round(hist.mean() if len(hist) > 0 else df['סה״כ ימים בהפרחה'].mean(), 1)
            season = get_season(start_date.month, lang_key)
            method = "📊 ממוצע היסטורי"

        thca_pred = None
        if gb_thca is not None and thca_feature_cols is not None:
            try:
                thca_pred = predict_thca(gb_thca, thca_feature_cols, mapping, df, greenhouse, strain, start_date, seasonal=seasonal)
            except Exception:
                thca_pred = None

        end_date = datetime.combine(start_date, datetime.min.time()) + timedelta(days=pred)

        st.markdown("---")
        if thca_pred is not None:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ ימי הפרחה צפויים", f"{pred} ימים")
        with col2:
            st.metric("📅 תאריך קציר משוער", end_date.strftime("%d/%m/%Y"))
        with col3:
            st.metric("🌤️ עונה", season)
        if thca_pred is not None:
            with col4:
                st.metric("🧪 THCA צפוי" if lang_key == "he" else "🧪 Predicted THCA", f"{thca_pred:.1f}%")

        st.info(f"שיטת Prediction: {method}")

        hist = df[(df['חממה'] == greenhouse) & (df['זן'] == strain)]['סה״כ ימים בהפרחה']
        if len(hist) > 0:
            st.markdown(f"**📚 היסטוריה:** {len(hist)} אצוות קודמות | Avg: {hist.mean():.1f} ימים | סטיית תקן: {hist.std():.1f} ימים")

            fig = px.histogram(hist, title=f"התפלגות ימי הפרחה - זן {strain} בחממה {greenhouse}",
                             labels={'value': 'ימי הפרחה'}, color_discrete_sequence=['#2d6a4f'])
            fig.add_vline(x=pred, line_dash="dash", line_color="red",
                         annotation_text=f"Prediction: {pred} ימים")
            st.plotly_chart(fig, use_container_width=True)

elif page == "📊 ניתוח נתונים":
    if lang_key=="en":
        st.markdown('<h3 style="text-align:left">Data Analysis</h3>', unsafe_allow_html=True)
    else:
        st.subheader("ניתוח נתונים")
    st.markdown("---")

    selected_gh = st.multiselect("Select Greenhouses" if lang_key=="en" else "בחר חממות", sorted(df['חממה'].unique()),
                                  default=sorted(df['חממה'].unique())[:3])
    filtered = df[df['חממה'].isin(selected_gh)] if selected_gh else df

    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'} if lang_key=='en' else {1:'ינואר',2:'פברואר',3:'מרץ',4:'אפריל',5:'מאי',6:'יוני',7:'יולי',8:'אוגוסט',9:'ספטמבר',10:'אוקטובר',11:'נובמבר',12:'דצמבר'}
    monthly = filtered.groupby('חודש_התחלה')['סה״כ ימים בהפרחה'].mean().reset_index()
    monthly['חודש'] = monthly['חודש_התחלה'].map(month_names)
    fig1 = px.bar(monthly, x='חודש', y='סה״כ ימים בהפרחה',
                  title="Avg Flowering Days by Entry Month" if lang_key=="en" else "ממוצע ימי הפרחה לפי חודש כניסה",
                  color='סה״כ ימים בהפרחה', color_continuous_scale=['#d4edd4','#2d6a4f'],
                  labels={'סה״כ ימים בהפרחה': 'Avg Flowering Days' if lang_key=='en' else 'ממוצע ימי הפרחה', 'חודש': 'Entry Month' if lang_key=='en' else 'חודש כניסה להפרחה'})
    fig1.update_layout(coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=0.0 if lang_key=='en' else 1.0, title_xanchor='left' if lang_key=='en' else 'right')
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        gh_perf = filtered.groupby('חממה')['סה״כ ימים בהפרחה'].agg(['mean','count']).reset_index()
        gh_perf.columns = ['חממה', 'ממוצע ימים', 'מספר אצוות']
        gh_perf = gh_perf.sort_values('ממוצע ימים')
        fig2 = px.bar(gh_perf, x='חממה', y='ממוצע ימים',
                      title="Avg Flowering Days by Greenhouse" if lang_key=="en" else "ממוצע ימי הפרחה לפי חממה",
                      color='ממוצע ימים', color_continuous_scale=['#d4edd4','#2d6a4f'],
                      hover_data=['מספר אצוות'],
                      labels={'ממוצע ימים': 'Avg Days' if lang_key=='en' else 'ממוצע ימי הפרחה', 'חממה': 'Greenhouse' if lang_key=='en' else 'חממה'})
        fig2.update_layout(coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=0.0 if lang_key=='en' else 1.0, title_xanchor='left' if lang_key=='en' else 'right')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        strain_perf = filtered.groupby('זן')['סה״כ ימים בהפרחה'].agg(['mean','count']).reset_index()
        strain_perf.columns = ['זן', 'ממוצע ימים', 'מספר אצוות']
        strain_perf = strain_perf[strain_perf['מספר אצוות'] >= 3].sort_values('ממוצע ימים')
        fig3 = px.bar(strain_perf, x='זן', y='ממוצע ימים',
                      title="Avg Flowering Days by Strain (min 3 batches)" if lang_key=="en" else "ממוצע ימי הפרחה לפי זן (מינימום 3 אצוות)",
                      color='ממוצע ימים', color_continuous_scale=['#d4edd4','#2d6a4f'],
                      hover_data=['מספר אצוות'],
                      labels={'ממוצע ימים': 'Avg Days' if lang_key=='en' else 'ממוצע ימי הפרחה', 'זן': 'Strain' if lang_key=='en' else 'זן'})
        fig3.update_layout(coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.9)', font=dict(color='#1a3a1e'), title_x=0.0 if lang_key=='en' else 1.0, title_xanchor='left' if lang_key=='en' else 'right')
        st.plotly_chart(fig3, use_container_width=True)

    if lang_key=="en":
        st.markdown('<h3 style="text-align:left">Data Table</h3>', unsafe_allow_html=True)
    else:
        st.subheader("טבלת נתונים")
    cols_show = ['מספר אצווה','זן','חממה','תאריך תחילת הפרחה','סה״כ ימים בהפרחה']
    if 'עונה' in filtered.columns:
        cols_show.append('עונה')
    display_df = filtered[cols_show].copy()
    display_df['סה״כ ימים בהפרחה'] = display_df['סה״כ ימים בהפרחה'].round(1)
    if lang_key=='en' and 'עונה' in display_df.columns:
        season_map = {'חורף':'Winter','אביב':'Spring','קיץ':'Summer','סתיו':'Fall'}
        display_df['עונה'] = display_df['עונה'].map(season_map).fillna(display_df['עונה'])
    if lang_key=='en':
        display_df.columns = ['Batch ID','Strain','Greenhouse','Start Date','Flowering Days'] + (['Season'] if 'עונה' in filtered.columns else [])
    else:
        display_df.columns = ['מספר אצווה','זן','חממה','תאריך התחלה','ימי הפרחה'] + (['עונה'] if 'עונה' in filtered.columns else [])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif page == "📅 גאנט":
    # ─── כותרת עמוד מקצועית ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a3a1e 0%,#2d6a4f 60%,#1a3a1e 100%);
                border-radius:14px;padding:20px 28px 16px;margin-bottom:18px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-size:2.2em;">📅</span>
        <div>
          <h2 style="color:#fff;margin:0;font-size:1.55em;letter-spacing:0.5px;">
            """ + ("Flowering Schedule — Gantt View" if lang_key=="en" else "לוח אצוות — תצוגת גאנט") + """
          </h2>
          <p style="color:#a8d8b8;margin:2px 0 0;font-size:0.9em;">
            """ + ("Live timeline · hover for details" if lang_key=="en" else "ציר זמן חי · עמידה על בלוק = פירוט מלא") + """
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    supabase_gantt = get_supabase()
    if supabase_gantt:
        try:
            res = supabase_gantt.table('batches').select('*').order('start_date', desc=False).limit(500).execute()
            raw = pd.DataFrame(res.data)
            raw['start'] = pd.to_datetime(raw['start_date'], errors='coerce')
            raw['planned_end'] = pd.to_datetime(raw['end_date'].astype(str).str[:10], format='%Y-%m-%d', errors='coerce')
            if 'actual_end_date' in raw.columns:
                raw['actual_end_dt'] = pd.to_datetime(raw['actual_end_date'], errors='coerce')
                raw['end'] = raw['actual_end_dt'].combine_first(raw['planned_end'])
                raw['ended_early'] = raw['actual_end_dt'].notna()
            else:
                raw['end'] = raw['planned_end']
                raw['ended_early'] = False
            raw['זן'] = raw['strain']
            raw['חממה'] = raw['greenhouse']
            raw['מספר אצווה'] = raw['batch_id']
            raw['סה״כ ימים'] = raw['total_days']
            def _batch_status(row, today_ts, lk):
                if row.get('ended_early', False):
                    return '⚡ סיים מוקדם' if lk=='he' else '⚡ Early finish'
                s = pd.to_datetime(row.get('start_date'), errors='coerce')
                e = pd.to_datetime(str(row.get('end_date',''))[:10], errors='coerce')
                if pd.notna(s) and pd.notna(e) and s <= today_ts <= e:
                    return '🌿 פעיל' if lk=='he' else '🌿 Active'
                if pd.notna(e) and e < today_ts:
                    return '✅ הסתיים' if lk=='he' else '✅ Done'
                return '📋 מתוכנן' if lk=='he' else '📋 Planned'
            raw['סוג'] = raw.apply(lambda r: _batch_status(r, pd.Timestamp.today(), lang_key), axis=1)
            df_valid = raw.dropna(subset=['start','end']).copy()
            df_valid = df_valid[df_valid['start'] >= '2023-01-01']
        except Exception as e:
            st.error(f"שגיאה: {e}")
            df_valid = pd.DataFrame()
    else:
        df_valid = pd.DataFrame()

    today = pd.Timestamp.today()

    # ─── פילטרים מקצועיים ────────────────────────────────────────────────────────
    st.markdown("""<div style="background:#f8fafc;border-radius:10px;padding:10px 16px 6px;
                    margin-bottom:10px;border:1px solid #cbd5e1;">""", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    with fc1:
        all_gh = sorted(df_valid['חממה'].unique()) if len(df_valid)>0 else []
        selected_gh_gantt = st.multiselect(
            ("Greenhouse" if lang_key=="en" else "חממה"), all_gh, default=all_gh, key='gantt_gh')
    with fc2:
        view_mode = st.radio(("View" if lang_key=="en" else "תצוגה"),
            ["Active + Future", "All", "Past Only"] if lang_key=="en" else ["פעיל + עתידי", "הכל", "עבר בלבד"],
            horizontal=True, index=0, key='gantt_view')
    with fc3:
        all_strains_g = sorted(df_valid['זן'].unique()) if len(df_valid)>0 else []
        selected_strain = st.multiselect(
            ("Strain" if lang_key=="en" else "זן"), all_strains_g, default=[], key='gantt_strain')
    st.markdown("</div>", unsafe_allow_html=True)

    filtered_gantt = df_valid[df_valid['חממה'].isin(selected_gh_gantt)] if selected_gh_gantt else df_valid.copy()
    if selected_strain:
        filtered_gantt = filtered_gantt[filtered_gantt['זן'].isin(selected_strain)]
    if view_mode in ["פעיל + עתידי", "Active + Future"] and 'end' in filtered_gantt.columns:
        filtered_gantt = filtered_gantt[filtered_gantt['end'] >= today]
    elif view_mode in ["עבר בלבד", "Past Only"]:
        filtered_gantt = filtered_gantt[filtered_gantt['end'] < today]

    # ─── פס מדדים מקצועי ─────────────────────────────────────────────────────────
    active_count  = len(filtered_gantt[(filtered_gantt['start'] <= today) & (filtered_gantt['end'] >= today)])
    future_count  = len(filtered_gantt[filtered_gantt['start'] > today])
    done_count    = len(filtered_gantt[filtered_gantt['end'] < today])
    total_count   = len(filtered_gantt)
    days_col_g    = [c for c in filtered_gantt.columns if 'ימים' in str(c) or ('days' in str(c).lower() and 'actual' not in str(c).lower())]
    avg_days_g    = round(filtered_gantt[days_col_g[0]].mean(), 1) if days_col_g else "—"

    _lbl_active  = "Active"    if lang_key=="en" else "פעיל כרגע"
    _lbl_future  = "Upcoming"  if lang_key=="en" else "עתידי"
    _lbl_done    = "Completed" if lang_key=="en" else "הסתיים"
    _lbl_total   = "Total"     if lang_key=="en" else "סה\"כ"
    _lbl_avg     = "Avg days"  if lang_key=="en" else "ממוצע ימים"
    _dir         = "ltr"       if lang_key=="en" else "rtl"

    st.markdown(f"""
    <div style="display:flex;gap:0;background:#fff;border-radius:12px;overflow:hidden;
                border:1px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin:10px 0 16px;
                direction:{_dir};">
      <div style="flex:1;padding:14px 22px;border-right:1px solid #e2e8f0;text-align:center;">
        <div style="font-size:0.72em;text-transform:uppercase;color:#64748b;letter-spacing:0.06em;font-weight:600;">{_lbl_active}</div>
        <div style="font-size:2em;font-weight:800;color:#15803d;line-height:1.15;">{active_count}</div>
        <div style="width:32px;height:3px;background:#15803d;border-radius:2px;margin:4px auto 0;"></div>
      </div>
      <div style="flex:1;padding:14px 22px;border-right:1px solid #e2e8f0;text-align:center;">
        <div style="font-size:0.72em;text-transform:uppercase;color:#64748b;letter-spacing:0.06em;font-weight:600;">{_lbl_future}</div>
        <div style="font-size:2em;font-weight:800;color:#2563eb;line-height:1.15;">{future_count}</div>
        <div style="width:32px;height:3px;background:#2563eb;border-radius:2px;margin:4px auto 0;"></div>
      </div>
      <div style="flex:1;padding:14px 22px;border-right:1px solid #e2e8f0;text-align:center;">
        <div style="font-size:0.72em;text-transform:uppercase;color:#64748b;letter-spacing:0.06em;font-weight:600;">{_lbl_done}</div>
        <div style="font-size:2em;font-weight:800;color:#64748b;line-height:1.15;">{done_count}</div>
        <div style="width:32px;height:3px;background:#94a3b8;border-radius:2px;margin:4px auto 0;"></div>
      </div>
      <div style="flex:1;padding:14px 22px;border-right:1px solid #e2e8f0;text-align:center;">
        <div style="font-size:0.72em;text-transform:uppercase;color:#64748b;letter-spacing:0.06em;font-weight:600;">{_lbl_total}</div>
        <div style="font-size:2em;font-weight:800;color:#d97706;line-height:1.15;">{total_count}</div>
        <div style="width:32px;height:3px;background:#d97706;border-radius:2px;margin:4px auto 0;"></div>
      </div>
      <div style="flex:1;padding:14px 22px;text-align:center;">
        <div style="font-size:0.72em;text-transform:uppercase;color:#64748b;letter-spacing:0.06em;font-weight:600;">{_lbl_avg}</div>
        <div style="font-size:2em;font-weight:800;color:#7c3aed;line-height:1.15;">{avg_days_g}</div>
        <div style="width:32px;height:3px;background:#7c3aed;border-radius:2px;margin:4px auto 0;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if len(filtered_gantt) == 0:
        st.warning("אין אצוות להצגה בטווח זה")
    else:
        filtered_gantt = filtered_gantt.sort_values(['חממה','start']).copy()
        TRANSITION_TYPES = ['🌾 קציר','🌾 Harvest','🧹 ניקוי','🧹 Cleaning','🌱 העברה','🌱 Transfer']

        # ─── כל אצווה = שורה משלה ────────────────────────────────────────────────
        # תווית: "GH | BatchID" — מבטיחה שאצוות מקבילות לא יסתירו אחת את השנייה
        filtered_gantt['שורה'] = filtered_gantt.apply(
            lambda r: f"{r['חממה']}  |  {r['מספר אצווה']}", axis=1
        )
        # סדר שורות: לפי חממה קודם, אחר כך לפי תאריך התחלה
        y_order = (filtered_gantt.sort_values(['חממה','start'])['שורה'].unique()).tolist()

        # ─── חישוב THCA + טווח קציר ───────────────────────────────────────────────
        thca_list, harv_early, harv_late, days_rem, harv_planned = [], [], [], [], []
        for _, row_g in filtered_gantt.iterrows():
            try:
                t = predict_thca(gb_thca, thca_feature_cols, mapping, df,
                                 row_g['חממה'], row_g['זן'], row_g['start'].date(),
                                 seasonal=seasonal) if gb_thca is not None else None
                thca_list.append(f"{t:.1f}" if t is not None else "—")
            except Exception:
                thca_list.append("—")
            hist_d = df[(df['חממה']==row_g['חממה']) & (df['זן']==row_g['זן'])]['סה״כ ימים בהפרחה'] if 'זן' in df.columns else pd.Series(dtype=float)
            if len(hist_d) < 3:
                hist_d = df.get('סה״כ ימים בהפרחה', pd.Series(dtype=float))
            std_d = round(hist_d.std()) if len(hist_d) > 1 else 3
            he = row_g.get('planned_end', row_g['end'])
            if pd.notna(he):
                harv_early.append((he - timedelta(days=std_d)).strftime('%d/%m/%Y'))
                harv_late.append((he + timedelta(days=std_d)).strftime('%d/%m/%Y'))
                harv_planned.append(he.strftime('%d/%m/%Y'))
            else:
                harv_early.append("—"); harv_late.append("—"); harv_planned.append("—")
            dl = (row_g['end'] - today).days
            days_rem.append(f"{dl} {'ימים' if lang_key=='he' else 'd'}" if dl >= 0 else ("הסתיים" if lang_key=="he" else "Done"))

        filtered_gantt['THCA צפוי']   = thca_list
        filtered_gantt['קציר מוקדם']  = harv_early
        filtered_gantt['קציר מאוחר']  = harv_late
        filtered_gantt['סיום מתוכנן'] = harv_planned
        filtered_gantt['נותר']        = days_rem

        # ─── בלוקי מעבר — קציר אחרי כל אצווה, ניקוי+העברה רק אחרי הקציר האחרון בחממה ─
        # last_end_per_gh: תאריך סיום מקסימלי לכל חממה
        last_end_per_gh = filtered_gantt.groupby('חממה')['end'].max()
        transition_rows = []
        for _, row_g in filtered_gantt.iterrows():
            h_d = int(row_g.get('actual_harvest_days') or row_g.get('harvest_days') or 1)
            c_d = int(row_g.get('actual_cleaning_days') or row_g.get('cleaning_days') or 3)
            t_d = int(row_g.get('actual_transfer_days') or row_g.get('transfer_days') or 3)
            batch_end  = row_g['end']
            h_end = batch_end  + timedelta(days=h_d)
            c_end = h_end      + timedelta(days=c_d)
            t_end = c_end      + timedelta(days=t_d)
            row_lbl = row_g['שורה']
            gh      = row_g['חממה']
            is_last_in_gh = (batch_end == last_end_per_gh.get(gh))
            base = {'שורה':row_lbl,'חממה':gh,'סוג':'🔄','THCA צפוי':'','קציר מוקדם':'','קציר מאוחר':'','סיום מתוכנן':'','נותר':'','מספר אצווה':''}
            # קציר — אחרי כל אצווה
            transition_rows.append(
                {**base,'start':batch_end,'end':h_end,
                 'זן':'🌾 קציר' if lang_key=='he' else '🌾 Harvest',
                 'מספר אצווה':f"קציר {h_d}י" if lang_key=='he' else f"Harvest {h_d}d"}
            )
            # ניקוי + העברה — רק אחרי האצווה האחרונה בחממה
            if is_last_in_gh:
                transition_rows.append(
                    {**base,'start':h_end,'end':c_end,
                     'זן':'🧹 ניקוי' if lang_key=='he' else '🧹 Cleaning',
                     'מספר אצווה':f"ניקוי {c_d}י" if lang_key=='he' else f"Cleaning {c_d}d"}
                )
                transition_rows.append(
                    {**base,'start':c_end,'end':t_end,
                     'זן':'🌱 העברה' if lang_key=='he' else '🌱 Transfer',
                     'מספר אצווה':f"העברה {t_d}י" if lang_key=='he' else f"Transfer {t_d}d"}
                )

        all_gantt = pd.concat(
            [filtered_gantt, pd.DataFrame(transition_rows)], ignore_index=True
        ) if transition_rows else filtered_gantt.copy()

        # ─── צבעים ───────────────────────────────────────────────────────────────
        STRAIN_COLORS = ['#1d4ed8','#15803d','#7e22ce','#b91c1c','#0369a1',
                         '#b45309','#be185d','#065f46','#6d28d9','#92400e']
        all_strains_uniq = [s for s in all_gantt['זן'].unique() if s not in TRANSITION_TYPES]
        color_map = {s: STRAIN_COLORS[i % len(STRAIN_COLORS)] for i,s in enumerate(all_strains_uniq)}
        color_map.update({'🌾 קציר':'#ca8a04','🌾 Harvest':'#ca8a04',
                          '🧹 ניקוי':'#64748b','🧹 Cleaning':'#64748b',
                          '🌱 העברה':'#16a34a','🌱 Transfer':'#16a34a'})

        # ─── בניית גרף ───────────────────────────────────────────────────────────
        fig = px.timeline(
            all_gantt, x_start='start', x_end='end', y='שורה', color='זן',
            color_discrete_map=color_map,
            custom_data=['מספר אצווה','זן','חממה','סוג','THCA צפוי',
                         'קציר מוקדם','קציר מאוחר','סיום מתוכנן','נותר']
        )

        # ─── Hover template ───────────────────────────────────────────────────────
        lbl = {
            'strain':  "Strain"      if lang_key=="en" else "זן",
            'gh':      "Greenhouse"  if lang_key=="en" else "חממה",
            'start':   "Start"       if lang_key=="en" else "תחילת הפרחה",
            'harvest': "Est. Harvest" if lang_key=="en" else "קציר משוער",
            'range':   "Range ±σ"    if lang_key=="en" else "טווח ±σ",
            'remain':  "Remaining"   if lang_key=="en" else "נותר",
            'status':  "Status"      if lang_key=="en" else "סטטוס",
        }
        for trace in fig.data:
            if trace.name in TRANSITION_TYPES:
                trace.hovertemplate = "<b>%{customdata[0]}</b><extra></extra>"
                trace.marker.opacity = 0.75
                trace.marker.line = dict(width=0)
                trace.width = 0.35
            else:
                trace.hovertemplate = (
                    "<b>📦 %{customdata[0]}</b><br>"
                    "──────────────────────<br>"
                    f"🌿 {lbl['strain']}: <b>%{{customdata[1]}}</b><br>"
                    f"🏠 {lbl['gh']}: <b>%{{customdata[2]}}</b><br>"
                    f"📅 {lbl['start']}: <b>%{{base|%d/%m/%Y}}</b><br>"
                    f"🌾 {lbl['harvest']}: <b>%{{customdata[7]}}</b><br>"
                    f"📊 {lbl['range']}: %{{customdata[5]}} → %{{customdata[6]}}<br>"
                    "🧪 THCA: <b>%{customdata[4]}%</b><br>"
                    f"⏳ {lbl['remain']}: %{{customdata[8]}}<br>"
                    f"🏷 {lbl['status']}: %{{customdata[3]}}<br>"
                    "<extra></extra>"
                )
                trace.marker.opacity = 0.90
                trace.marker.line = dict(color='rgba(255,255,255,0.6)', width=1.5)
                trace.width = 0.55

        # ─── רקעים מתחלפים + הפרדות חממות ───────────────────────────────────────
        prev_gh = None
        for i, row_lbl in enumerate(y_order):
            gh_of_row = row_lbl.split('  |  ')[0].strip()
            fill = '#f1f5f9' if i % 2 == 0 else '#ffffff'
            fig.add_hrect(y0=i-0.5, y1=i+0.5, fillcolor=fill,
                          layer='below', line_width=0, opacity=1)
            if prev_gh is not None and gh_of_row != prev_gh:
                # קו הפרדה בין חממות
                fig.add_hrect(y0=i-0.52, y1=i-0.48, fillcolor='#334155',
                              layer='above', line_width=0, opacity=1)
            prev_gh = gh_of_row

        today_str = datetime.today().strftime('%Y-%m-%d')

        # ─── layout ──────────────────────────────────────────────────────────────
        n_rows = max(len(y_order), 1)
        fig.update_layout(
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            font=dict(family="'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
                      size=12, color='#1e293b'),
            title=dict(text=''),
            margin=dict(l=160, r=160, t=48, b=20),
            height=max(340, n_rows * 56 + 120),
            xaxis=dict(
                side='top', title=dict(text=''),
                showgrid=True, gridcolor='#e9eef3', gridwidth=1,
                tickfont=dict(size=11, color='#475569', family="'Segoe UI',Arial,sans-serif"),
                tickformat='%d %b', zeroline=False,
                showline=True, linecolor='#94a3b8', linewidth=1,
            ),
            yaxis=dict(
                title=dict(text=''),
                categoryorder='array', categoryarray=list(reversed(y_order)),
                tickfont=dict(size=11.5, color='#1e293b',
                              family="'Segoe UI',Arial,sans-serif"),
                showgrid=False, showline=True,
                linecolor='#cbd5e1', linewidth=1,
                ticklabelposition='outside',
            ),
            legend=dict(
                title=dict(
                    text=("Strain / Phase" if lang_key=="en" else "זן / שלב"),
                    font=dict(size=11, color='#475569')),
                bgcolor='rgba(255,255,255,0.97)',
                bordercolor='#e2e8f0', borderwidth=1,
                font=dict(size=10.5), orientation='v',
                x=1.01, y=1, xanchor='left',
                tracegroupgap=4,
            ),
            hoverlabel=dict(
                bgcolor='#0f172a', font_color='#f1f5f9',
                font_size=12.5, bordercolor='#22d3ee', namelength=0,
                font_family="'Segoe UI',Arial,sans-serif",
            ),
            bargap=0.22,
        )

        # ─── קו "היום" + תווית ───────────────────────────────────────────────────
        fig.add_vline(x=today_str, line_dash="dot", line_color="#dc2626", line_width=2)
        try:
            fig.add_annotation(
                x=today_str, y=1.0, xref='x', yref='paper',
                text=("◀ Today" if lang_key=="en" else "היום ▶"),
                showarrow=False, yanchor='bottom',
                font=dict(size=10, color='#dc2626'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#dc2626', borderwidth=1,
                borderpad=3,
            )
        except Exception:
            pass

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # ─── מקרא ────────────────────────────────────────────────────────────────
        _legend_dir = "ltr" if lang_key=="en" else "rtl"
        st.markdown(
            f"""<div style="display:flex;gap:22px;align-items:center;padding:8px 16px;
                background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;
                font-size:0.8em;color:#475569;flex-wrap:wrap;
                margin-top:6px;direction:{_legend_dir};">
              <span style="font-weight:700;color:#334155;font-size:0.9em;">"""
              + ("Legend:" if lang_key=="en" else "מקרא:") + """</span>
              <span style="display:flex;align-items:center;gap:5px;">
                <span style="display:inline-block;width:14px;height:10px;background:#ca8a04;border-radius:2px;"></span>"""
              + ("Harvest" if lang_key=="en" else "קציר") + """</span>
              <span style="display:flex;align-items:center;gap:5px;">
                <span style="display:inline-block;width:14px;height:10px;background:#64748b;border-radius:2px;"></span>"""
              + ("Cleaning" if lang_key=="en" else "ניקוי") + """</span>
              <span style="display:flex;align-items:center;gap:5px;">
                <span style="display:inline-block;width:14px;height:10px;background:#16a34a;border-radius:2px;"></span>"""
              + ("Transfer" if lang_key=="en" else "העברת שתילים") + """</span>
              <span style="display:flex;align-items:center;gap:5px;border-left:1px solid #cbd5e1;padding-left:16px;">
                <span style="display:inline-block;width:2px;height:14px;background:#dc2626;border-radius:1px;"></span>"""
              + ("Today" if lang_key=="en" else "היום") + """</span>
            </div>""", unsafe_allow_html=True
        )

        # ─── טבלת אצוות פעילות ───────────────────────────────────────────────────
        active_now = filtered_gantt[(filtered_gantt['start'] <= today) & (filtered_gantt['end'] >= today)].copy()
        if len(active_now) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div style='background:#0f172a;border-radius:8px 8px 0 0;"
                "padding:10px 20px;color:#e2e8f0;font-weight:600;font-size:0.95em;"
                "letter-spacing:0.02em;border-bottom:2px solid #22c55e;'>"
                + ("Active Batches — Predicted Data" if lang_key=="en" else "אצוות פעילות — נתוני חיזוי") +
                "</div>", unsafe_allow_html=True
            )
            show_cols = [c for c in ['מספר אצווה','חממה','זן','start','end',
                                     'THCA צפוי','קציר מוקדם','קציר מאוחר','נותר','סוג']
                         if c in active_now.columns]
            show_active = active_now[show_cols].copy()
            if 'start' in show_active.columns:
                show_active['start'] = show_active['start'].dt.strftime('%d/%m/%Y')
            if 'end' in show_active.columns:
                show_active['end'] = show_active['end'].dt.strftime('%d/%m/%Y')
            show_active = show_active.rename(columns={
                'start':'תחילת הפרחה','end':'סיום צפוי',
                'קציר מוקדם':'קציר מוקדם (−σ)','קציר מאוחר':'קציר מאוחר (+σ)',
                'נותר':'ימים נותרים','סוג':'סטטוס'
            })
            st.dataframe(show_active, use_container_width=True, hide_index=True)
