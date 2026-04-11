import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR  = os.path.join(BASE_DIR, '..', 'app')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')


def load_data():
    """טעינת דאטאסט האימון + מיזוג עם תוצאות מעבדה (THCA)"""
    train_path = os.path.join(APP_DIR, 'training_dataset.csv')
    lab_path   = os.path.join(DATA_DIR, 'תוצאות מעבדה.xlsx')

    df = pd.read_csv(train_path)

    if os.path.exists(lab_path):
        print("מוצא קובץ תוצאות מעבדה, ממזג THCA...")
        lab_raw = pd.read_excel(lab_path, header=7)
        lab_raw = lab_raw.dropna(subset=[lab_raw.columns[1]])
        lab_raw = lab_raw.rename(columns={
            lab_raw.columns[1]:  'מספר אצווה',
            lab_raw.columns[13]: 'THCA',
            lab_raw.columns[14]: 'THC_val',
            lab_raw.columns[15]: 'Total_THC_as_is',
            lab_raw.columns[16]: 'Total_THC_dry',
        })
        lab_raw['מספר אצווה']      = lab_raw['מספר אצווה'].astype(str).str.strip()
        lab_raw['THCA']            = pd.to_numeric(lab_raw['THCA'],            errors='coerce')
        lab_raw['THC_val']         = pd.to_numeric(lab_raw['THC_val'],         errors='coerce')
        lab_raw['Total_THC_as_is'] = pd.to_numeric(lab_raw['Total_THC_as_is'], errors='coerce')
        lab_slim = lab_raw[['מספר אצווה', 'THCA', 'THC_val', 'Total_THC_as_is']].drop_duplicates('מספר אצווה')

        df['מספר אצווה'] = df['מספר אצווה'].astype(str).str.strip()
        df = df.merge(lab_slim, on='מספר אצווה', how='left')
        merged = df['THCA'].notna().sum()
        print(f"  <- {merged} שורות עם THCA מתוך {len(df)}")
    else:
        print("קובץ תוצאות מעבדה לא נמצא - ממשיך בלי THCA")
        df['THCA']            = np.nan
        df['THC_val']         = np.nan
        df['Total_THC_as_is'] = np.nan

    return df


def build_features(df, include_thca=False):
    """בנה רשימת פיצ'רים"""
    base_cols   = ['חממה_קוד', 'זן_קוד', 'עונה_קוד', 'חודש_התחלה']
    sensor_cols = [c for c in df.columns if '_mean' in c or '_std' in c]
    feature_cols = base_cols + sensor_cols
    if include_thca and 'THCA' in df.columns:
        feature_cols.append('THCA')
    return feature_cols


def train_flowering_model(df):
    """מודל 1: חיזוי ימי הפרחה"""
    print("\n" + "="*50)
    print("מודל 1: חיזוי ימי הפרחה")

    # מצא את עמודת היעד
    target = None
    for col in df.columns:
        if 'ימים' in col and 'הפרחה' in col:
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model',   GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, min_samples_leaf=5,
            subsample=0.8, random_state=42
        ))
    ])
    gb.fit(X_train, y_train)

    rf = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model',   RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_train, y_train)

    for name, model in [("Gradient Boosting", gb), ("Random Forest", rf)]:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        cv  = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        print(f"\n  {name}")
        print(f"  MAE: {mae:.2f} ימים  |  R2: {r2:.3f}  |  CV MAE: {-cv.mean():.2f} +/- {cv.std():.2f}")

    importances = pd.Series(
        gb.named_steps['model'].feature_importances_,
        index=feature_cols
    )
    print(f"\n  חשיבות משתנים (GB):")
    print(importances.sort_values(ascending=False).head(8).to_string())

    return gb, rf, feature_cols


def train_thca_model(df):
    """מודל 2: חיזוי THCA"""
    print("\n" + "="*50)
    print("מודל 2: חיזוי THCA")

    df_thca = df[df['THCA'].notna()].copy()
    if len(df_thca) < 30:
        print(f"  רק {len(df_thca)} שורות עם THCA - דלג על מודל THCA")
        return None, None

    feature_cols = build_features(df_thca, include_thca=False)
    target = 'THCA'

    X = df_thca[feature_cols]
    y = df_thca[target]

    print(f"שורות לאימון: {len(df_thca)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb_thca = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model',   GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=3, min_samples_leaf=3,
            random_state=42
        ))
    ])
    gb_thca.fit(X_train, y_train)

    y_pred = gb_thca.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    cv  = cross_val_score(gb_thca, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"\n  Gradient Boosting (THCA)")
    print(f"  MAE: {mae:.2f}%  |  R2: {r2:.3f}  |  CV MAE: {-cv.mean():.2f} +/- {cv.std():.2f}")

    importances = pd.Series(
        gb_thca.named_steps['model'].feature_importances_,
        index=feature_cols
    )
    print(f"\n  חשיבות משתנים:")
    print(importances.sort_values(ascending=False).head(8).to_string())

    return gb_thca, feature_cols


def save_models(gb, rf, feature_cols, gb_thca, thca_feature_cols, df):
    """שמירת המודלים לתיקיית app/"""
    os.makedirs(APP_DIR, exist_ok=True)

    joblib.dump(gb,           os.path.join(APP_DIR, 'gb_model.pkl'))
    joblib.dump(rf,           os.path.join(APP_DIR, 'rf_model.pkl'))
    joblib.dump(feature_cols, os.path.join(APP_DIR, 'feature_cols.pkl'))

    if gb_thca is not None:
        joblib.dump(gb_thca,           os.path.join(APP_DIR, 'gb_thca_model.pkl'))
        joblib.dump(thca_feature_cols, os.path.join(APP_DIR, 'thca_feature_cols.pkl'))

    mapping = {
        'חממות':             sorted(df['חממה'].dropna().unique().tolist()),
        'זנים':              sorted(df['זן'].dropna().unique().tolist()),
        'עונות':             ['אביב', 'קיץ', 'סתיו', 'חורף'],
        'feature_cols':      feature_cols,
        'has_thca_model':    gb_thca is not None,
        'thca_feature_cols': thca_feature_cols or [],
    }
    joblib.dump(mapping, os.path.join(APP_DIR, 'mapping.pkl'))
    print(f"\n המודלים נשמרו ב-{os.path.abspath(APP_DIR)}")


def train_model():
    print("טוען נתונים...")
    df = load_data()

    gb, rf, feature_cols       = train_flowering_model(df)
    gb_thca, thca_feature_cols = train_thca_model(df)

    save_models(gb, rf, feature_cols, gb_thca, thca_feature_cols, df)
    return gb, feature_cols


if __name__ == '__main__':
    train_model()
