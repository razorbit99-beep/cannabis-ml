import pandas as pd
import numpy as np
import glob
import os

def load_batches():
    """טעינת סיכום אצוות"""
    df = pd.read_excel('data/raw/סיכום אצוות כולל.xlsx')
    df['תאריך תחילת הפרחה'] = pd.to_datetime(df['תאריך תחילת הפרחה'], dayfirst=True, errors='coerce')
    df['חודש_התחלה'] = df['תאריך תחילת הפרחה'].dt.month
    df['עונה'] = df['חודש_התחלה'].map({
        12:'חורף', 1:'חורף', 2:'חורף',
        3:'אביב', 4:'אביב', 5:'אביב',
        6:'קיץ', 7:'קיץ', 8:'קיץ',
        9:'סתיו', 10:'סתיו', 11:'סתיו'
    })
    return df

def load_greenhouse_sensors():
    """טעינת נתוני חיישנים מכל החממות"""
    all_dfs = []
    files = glob.glob('data/raw/[A-Z]*.csv') + glob.glob('data/raw/[a-z]*.csv')
    # רק קבצי חממות (לא DLI)
    sensor_files = [f for f in files if 'DLI' not in f and 'ימים' not in f]
    
    for f in sensor_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'GREENHOUSE' in df.columns:
                all_dfs.append(df)
        except:
            pass
    
    if not all_dfs:
        return None
    
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined

def load_dli():
    """טעינת נתוני DLI קרינה"""
    all_dfs = []
    files = glob.glob('data/raw/DLI*.csv') + glob.glob('data/raw/DLI*.CSV')
    
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'GREENHOUSE' in df.columns:
                all_dfs.append(df)
        except:
            pass
    
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)

def compute_sensor_stats_per_greenhouse(sensors_df):
    """חישוב ממוצעים של חיישנים לפי חממה"""
    if sensors_df is None:
        return None
    
    # זיהוי עמודת חממה
    gh_col = 'GREENHOUSE'
    
    # עמודות נומריות בלבד
    num_cols = sensors_df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = sensors_df.groupby(gh_col)[num_cols].agg(['mean','std']).round(3)
    stats.columns = ['_'.join(c) for c in stats.columns]
    stats = stats.reset_index()
    
    # נשמור רק עמודת חממה ראשית (A1 -> A)
    stats['חממה'] = stats[gh_col].astype(str).str[0]
    
    return stats

def build_training_dataset():
    """בניית דאטאסט לאימון"""
    print("טוען אצוות...")
    batches = load_batches()
    
    print("טוען נתוני חיישנים...")
    sensors = load_greenhouse_sensors()
    sensor_stats = compute_sensor_stats_per_greenhouse(sensors)
    
    print("מחשב ממוצעי חיישנים לחממה...")
    if sensor_stats is not None:
        df = batches.merge(sensor_stats, on='חממה', how='left')
    else:
        df = batches.copy()
    
    # פיצ'רים בסיסיים
    df['חממה_קוד'] = pd.Categorical(df['חממה']).codes
    df['זן_קוד'] = pd.Categorical(df['זן']).codes
    df['עונה_קוד'] = pd.Categorical(df['עונה']).codes
    
    # שמירה
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/training_dataset.csv', index=False)
    print(f"✅ נשמר! {len(df)} שורות, {len(df.columns)} עמודות")
    print("עמודות:", list(df.columns)[:15], "...")
    return df

if __name__ == '__main__':
    df = build_training_dataset()
