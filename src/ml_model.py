import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_model():
    print("טוען דאטאסט...")
    df = pd.read_csv('data/processed/training_dataset.csv')
    
    # פיצ'רים למודל
    feature_cols = [
        'חממה_קוד', 'זן_קוד', 'עונה_קוד', 'חודש_התחלה'
    ]
    
    # הוספת עמודות חיישנים אם קיימות
    sensor_cols = [c for c in df.columns if '_mean' in c or '_std' in c]
    feature_cols += sensor_cols
    
    target = 'סה״כ ימים בהפרחה'
    
    # ניקוי
    df_clean = df[feature_cols + [target]].dropna()
    print(f"שורות לאימון: {len(df_clean)}")
    print(f"פיצ'רים: {feature_cols}")
    
    X = df_clean[feature_cols]
    y = df_clean[target]
    
    # חלוקה לאימון ובדיקה
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # אימון Random Forest
    print("\nמאמן Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # אימון Gradient Boosting
    print("מאמן Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
    gb.fit(X_train, y_train)
    
    # הערכה
    for name, model in [("Random Forest", rf), ("Gradient Boosting", gb)]:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        print(f"\n{'='*30}")
        print(f"מודל: {name}")
        print(f"MAE (שגיאה ממוצעת): {mae:.2f} ימים")
        print(f"R² (דיוק): {r2:.3f}")
        print(f"Cross-Validation MAE: {-cv.mean():.2f} ± {cv.std():.2f} ימים")
    
    # חשיבות פיצ'רים
    print(f"\n{'='*30}")
    print("חשיבות משתנים (Random Forest):")
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    print(importances.sort_values(ascending=False).head(10))
    
    # שמירת המודל הטוב יותר
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/rf_model.pkl')
    joblib.dump(gb, 'models/gb_model.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    
    # שמירת מיפויים לחיזוי עתידי
    mapping = {
        'חממות': sorted(df['חממה'].dropna().unique().tolist()),
        'זנים': sorted(df['זן'].dropna().unique().tolist()),
        'עונות': ['אביב', 'קיץ', 'סתיו', 'חורף'],
        'feature_cols': feature_cols
    }
    joblib.dump(mapping, 'models/mapping.pkl')
    
    print("\n✅ מודלים נשמרו בתיקיית models/")
    return rf, feature_cols

if __name__ == '__main__':
    train_model()
