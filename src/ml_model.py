"""
ml_model.py
-----------
אימון מודל חיזוי ימי הפרחה + THCA עם משתנים סביבתיים.
המודל מתחשב בטמפ', לחות, קרינה, DLI ועוד.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR  = os.path.join(BASE_DIR, '..', 'app')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')


# =============================================================================
# טעינת נתונים
# =============================================================================

def load_data(seasonal=None):
    """
    טוען דאטאסט אימון + תוצאות מעבדה.
    אם אין seasonal מוכן, יש לקרוא build_training_dataset() קודם.
    """
    from data_processor import build_training_dataset

    # ניסיון לטעון מקובץ מעובד
    processed_path = os.path.join(BASE_DIR, '..', 'data', 'processed', 'training_dataset.csv')
    app_path = os.path.join(APP_DIR, 'training_dataset.csv')

    if os.path.exists(processed_path):
        print("טוען דאטאסט מעובד מ-data/processed/...")
        df = pd.read_csv(processed_path)
        if seasonal is None:
            # בנה seasonal בנפרד
            _, seasonal = build_training_dataset(save=False)
    elif os.path.exists(app_path):
        print("טוען דאטאסט מ-app/...")
        df = pd.read_csv(app_path)
    else:
        print("בונה דאטאסט חדש...")
        df, seasonal = build_training_dataset(save=True)

    # מיזוג תוצאות מעבדה (THCA)
    lab_path = os.path.join(DATA_DIR, 'תוצאות מעבדה.xlsx')
    if os.path.exists(lab_path):
        print("ממזג תוצאות מעבדה (THCA)...")
        try:
            lab_raw = pd.read_excel(lab_path, header=7)
            lab_raw = lab_raw.dropna(subset=[lab_raw.columns[1]])
            lab_raw = lab_raw.rename(columns={
                lab_raw.columns[1]:  'מספר אצווה',
                lab_raw.columns[13]: 'THCA',
                lab_raw.columns[14]: 'THC_val',
                lab_raw.columns[15]: 'Total_THC_as_is',
            })
            lab_raw['מספר אצווה'] = lab_raw['מספר אצווה'].astype(str).str.strip()
            lab_raw['THCA']       = pd.to_numeric(lab_raw['THCA'], errors='coerce')
            lab_slim = lab_raw[['מספר אצווה', 'THCA', 'THC_val', 'Total_THC_as_is']].drop_duplicates('מספר אצווה')

            df['מספר אצווה'] = df['מספר אצווה'].astype(str).str.strip()
            df = df.merge(lab_slim, on='מספר אצווה', how='left')
            print(f"  {df['THCA'].notna().sum()} שורות עם THCA מתוך {len(df)}")
        except Exception as e:
            print(f"  שגיאה בטעינת מעבדה: {e}")
            df['THCA'] = np.nan

    return df, seasonal


# =============================================================================
# בניית רשימת פיצ'רים
# =============================================================================

# פיצ'רים סביבתיים שנרצה לכלול (אם קיימים בדאטאסט)
ENV_FEATURES = [
    'temp_int_mean', 'temp_int_std',
    'humidity_mean', 'humidity_std',
    'radiation_mean', 'radiation_std',
    'temp_ext_mean', 'temp_ext_std',
    'humidity_ext_mean',
    'dli_mean', 'dli_std',
]

# פיצ'רים ישנים (תאימות אחורה עם training_dataset.csv ישן)
LEGACY_ENV_FEATURES = [
    'Humidity (%)_mean', 'Humidity (%)_std',
    'Radiation (J/m2)_mean', 'Radiation (J/m2)_std',
]


def build_features(df, include_thca=False):
    """רשימת פיצ'רים זמינים בדאטאסט"""
    base = ['חממה_קוד', 'זן_קוד', 'עונה_קוד', 'חודש_התחלה']
    env  = [c for c in ENV_FEATURES + LEGACY_ENV_FEATURES if c in df.columns]
    feat = base + env
    if include_thca and 'THCA' in df.columns:
        feat.append('THCA')
    return feat


# =============================================================================
# אימון מודלים
# =============================================================================

def train_flowering_model(df):
    """מודל 1: חיזוי ימי הפרחה"""
    print("\n" + "=" * 55)
    print("מודל 1: חיזוי ימי הפרחה")

    # מציאת עמודת יעד
    target = None
    for col in df.columns:
        if 'ימים' in str(col) and 'הפרחה' in str(col):
            target = col
            break
    if target is None:
        raise ValueError("לא נמצאה עמודת 'ימים בהפרחה'")

    feature_cols = build_features(df, include_thca=True)
    df_model = df[feature_cols + [target]].copy()
    df_model[target] = pd.to_numeric(df_model[target], errors='coerce')
    df_model = df_model.dropna(subset=[target])

    X = df_model[feature_cols]
    y = df_model[target]

    print(f"שורות לאימון: {len(df_model)}")
    print(f"פיצ'רים ({len(feature_cols)}): {feature_cols}")

    env_cols = [c for c in feature_cols if c not in ['חממה_קוד', 'זן_קוד', 'עונה_קוד', 'חודש_התחלה', 'THCA']]
    non_null_env = X[env_cols].notna().any(axis=1).sum() if env_cols else 0
    print(f"  אצוות עם נתוני חיישנים: {non_null_env}/{len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, min_samples_leaf=5,
            subsample=0.8, random_state=42,
        ))
    ])
    gb.fit(X_train, y_train)

    rf = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_train, y_train)

    print("\n  תוצאות:")
    for name, model in [("Gradient Boosting", gb), ("Random Forest", rf)]:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        cv  = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        print(f"  {name}: MAE={mae:.2f} ימים | R²={r2:.3f} | CV MAE={-cv.mean():.2f}±{cv.std():.2f}")

    importances = pd.Series(
        gb.named_steps['model'].feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n  חשיבות פיצ'רים:")
    print(importances.head(10).to_string())

    return gb, rf, feature_cols


def train_thca_model(df):
    """מודל 2: חיזוי THCA"""
    print("\n" + "=" * 55)
    print("מודל 2: חיזוי THCA")

    df_thca = df[df['THCA'].notna()].copy() if 'THCA' in df.columns else pd.DataFrame()
    if len(df_thca) < 20:
        print(f"  רק {len(df_thca)} שורות עם THCA - לא מאמן")
        return None, None

    feature_cols = build_features(df_thca, include_thca=False)
    X = df_thca[feature_cols]
    y = df_thca['THCA']

    print(f"שורות לאימון: {len(df_thca)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb_thca = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=3, min_samples_leaf=3,
            random_state=42,
        ))
    ])
    gb_thca.fit(X_train, y_train)

    y_pred = gb_thca.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    cv  = cross_val_score(gb_thca, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"  MAE={mae:.2f}% | R²={r2:.3f} | CV MAE={-cv.mean():.2f}±{cv.std():.2f}")

    importances = pd.Series(
        gb_thca.named_steps['model'].feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n  חשיבות פיצ'רים:")
    print(importances.head(8).to_string())

    return gb_thca, feature_cols


# =============================================================================
# שמירה
# =============================================================================

def save_models(gb, rf, feature_cols, gb_thca, thca_feature_cols, df, seasonal):
    """שמירת מודלים + מיפוי לתיקיית app/"""
    os.makedirs(APP_DIR, exist_ok=True)

    joblib.dump(gb,           os.path.join(APP_DIR, 'gb_model.pkl'))
    joblib.dump(rf,           os.path.join(APP_DIR, 'rf_model.pkl'))
    joblib.dump(feature_cols, os.path.join(APP_DIR, 'feature_cols.pkl'))

    if gb_thca is not None:
        joblib.dump(gb_thca,           os.path.join(APP_DIR, 'gb_thca_model.pkl'))
        joblib.dump(thca_feature_cols, os.path.join(APP_DIR, 'thca_feature_cols.pkl'))

    # seasonal: dict { (gh, month): {feat: val} }
    seasonal_out = seasonal or {}
    joblib.dump(seasonal_out, os.path.join(APP_DIR, 'seasonal_averages.pkl'))

    mapping = {
        'חממות':              sorted(df['חממה'].dropna().unique().tolist()),
        'זנים':               sorted(df['זן'].dropna().unique().tolist()),
        'עונות':              ['אביב', 'קיץ', 'סתיו', 'חורף'],
        'feature_cols':       feature_cols,
        'has_thca_model':     gb_thca is not None,
        'thca_feature_cols':  thca_feature_cols or [],
        'has_seasonal':       len(seasonal_out) > 0,
    }
    joblib.dump(mapping, os.path.join(APP_DIR, 'mapping.pkl'))
    print(f"\n✅ המודלים נשמרו ב-{os.path.abspath(APP_DIR)}")
    print(f"   seasonal_averages.pkl: {len(seasonal_out)} רשומות")


# =============================================================================
# נקודת כניסה
# =============================================================================

def train_model():
    print("מתחיל אימון מודלים...")
    df, seasonal = load_data()

    gb, rf, feature_cols       = train_flowering_model(df)
    gb_thca, thca_feature_cols = train_thca_model(df)

    save_models(gb, rf, feature_cols, gb_thca, thca_feature_cols, df, seasonal)
    return gb, feature_cols


if __name__ == '__main__':
    train_model()
