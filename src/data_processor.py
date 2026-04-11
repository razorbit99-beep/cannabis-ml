"""
data_processor.py
-----------------
בונה דאטאסט אימון עם ממוצעי חיישנים ספציפיים לכל אצווה.
עיבוד חסכוני בזיכרון: כל קובץ חיישן מסוכם לממוצע יומי לפני המיזוג.
"""
import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')

# חיפוש תיקיית הנתונים
_POSSIBLE_DATA_DIRS = [
    os.path.join(ROOT_DIR, 'data', 'raw'),
    os.path.join(ROOT_DIR, '..', 'data', 'raw'),
    os.path.expanduser('~/Desktop/cannabis_ml/data/raw'),
]
DATA_RAW = next(
    (d for d in _POSSIBLE_DATA_DIRS if os.path.isfile(os.path.join(d, 'סיכום אצוות כולל.xlsx'))),
    _POSSIBLE_DATA_DIRS[0]
)

SENSOR_COLS_MAP = {
    'Humidity (%)':     'humidity',
    'Radiation (J/m2)': 'radiation',
    'Ex. Temp (°C)':    'temp_ext',
    'Ex. Humidity (%)': 'humidity_ext',
}
TEMP_PATTERN = 'Avg_Temp'
DLI_ZONE_COLS = ['DLI Zone 1', 'DLI Zone 2', 'DLI Zone 3']


def _parse_dates(series):
    return pd.to_datetime(series, dayfirst=True, errors='coerce')


def load_batches():
    path = os.path.join(DATA_RAW, 'סיכום אצוות כולל.xlsx')
    df = pd.read_excel(path)
    df['תאריך תחילת הפרחה'] = _parse_dates(df['תאריך תחילת הפרחה'])
    df['תאריך סיום הפרחה']  = _parse_dates(df['תאריך סיום הפרחה'])
    df['חודש_התחלה'] = df['תאריך תחילת הפרחה'].dt.month
    df['עונה'] = df['חודש_התחלה'].map({
        12: 'חורף', 1: 'חורף', 2: 'חורף',
        3: 'אביב', 4: 'אביב', 5: 'אביב',
        6: 'קיץ', 7: 'קיץ', 8: 'קיץ',
        9: 'סתיו', 10: 'סתיו', 11: 'סתיו',
    })
    return df


def _daily_from_sensor_file(fpath):
    """
    טוען קובץ חיישן אחד, מחשב ממוצע יומי לכל חיישן.
    מחזיר DataFrame: [date, gh_code, temp_int, humidity, radiation, temp_ext, humidity_ext]
    """
    try:
        df = pd.read_csv(fpath, low_memory=False)
        if 'GREENHOUSE' not in df.columns or 'תאריך' not in df.columns:
            return None

        df['date']    = _parse_dates(df['תאריך'])
        df['gh_code'] = df['GREENHOUSE'].astype(str).str.strip().str[0].str.upper()
        df = df.dropna(subset=['date'])

        # עמודת טמפ' פנימית (שם עם unicode)
        temp_col = next((c for c in df.columns if TEMP_PATTERN in c), None)

        agg_dict = {}
        if temp_col:
            df['temp_int'] = pd.to_numeric(df[temp_col], errors='coerce')
            agg_dict['temp_int'] = 'mean'

        for col_pat, feat_name in SENSOR_COLS_MAP.items():
            matched = next((c for c in df.columns if col_pat in c), None)
            if matched:
                df[feat_name] = pd.to_numeric(df[matched], errors='coerce')
                agg_dict[feat_name] = 'mean'

        if not agg_dict:
            return None

        daily = df.groupby(['date', 'gh_code']).agg(agg_dict).reset_index()
        return daily.round(3)
    except Exception as e:
        return None


def _daily_from_dli_file(fpath):
    """
    טוען קובץ DLI אחד, מחשב ממוצע יומי.
    מחזיר DataFrame: [date, gh_code, dli]
    """
    try:
        df = pd.read_csv(fpath, low_memory=False)
        if 'GREENHOUSE' not in df.columns or 'תאריך' not in df.columns:
            return None

        df['date']    = _parse_dates(df['תאריך'])
        df['gh_code'] = df['GREENHOUSE'].astype(str).str.strip().str[0].str.upper()
        df = df.dropna(subset=['date'])

        existing = [c for c in DLI_ZONE_COLS if c in df.columns]
        if not existing:
            return None

        for c in existing:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        df['dli'] = df[existing].mean(axis=1)
        daily = df.groupby(['date', 'gh_code'])['dli'].mean().reset_index()
        return daily.round(3)
    except Exception as e:
        return None


def build_daily_sensors():
    """
    טוען את כל קבצי החיישנים ומחשב ממוצעים יומיים לכל חממה.
    חסכוני בזיכרון: מעבד קובץ אחד בכל פעם.
    מחזיר: (daily_sensors_df, daily_dli_df)
    """
    print("  טוען קבצי חיישנים (ממוצע יומי)...")
    sensor_dfs = []
    dli_dfs    = []

    all_files = glob.glob(os.path.join(DATA_RAW, '*.csv')) + \
                glob.glob(os.path.join(DATA_RAW, '*.CSV'))

    n_sensor, n_dli = 0, 0
    for f in sorted(all_files):
        fname = os.path.basename(f).upper()
        # דלג על קבצי Excel, אצוות, מעבדה
        if any(x in fname for x in ['ימים', 'סיכום', 'מעקב', 'תוצאות']):
            continue
        if 'DLI' in fname:
            d = _daily_from_dli_file(f)
            if d is not None and len(d) > 0:
                dli_dfs.append(d)
                n_dli += 1
        else:
            d = _daily_from_sensor_file(f)
            if d is not None and len(d) > 0:
                sensor_dfs.append(d)
                n_sensor += 1

    sensors_daily = pd.concat(sensor_dfs, ignore_index=True) if sensor_dfs else None
    dli_daily     = pd.concat(dli_dfs,    ignore_index=True) if dli_dfs     else None

    if sensors_daily is not None:
        # ממוצע מכל הקבצים לאותה חממה + תאריך
        num_cols = [c for c in sensors_daily.columns if c not in ['date', 'gh_code']]
        sensors_daily = sensors_daily.groupby(['date', 'gh_code'])[num_cols].mean().reset_index().round(3)
        print(f"  נטענו {n_sensor} קבצי חיישנים → {len(sensors_daily):,} שורות יומיות")

    if dli_daily is not None:
        dli_daily = dli_daily.groupby(['date', 'gh_code'])['dli'].mean().reset_index().round(3)
        print(f"  נטענו {n_dli} קבצי DLI → {len(dli_daily):,} שורות יומיות")

    return sensors_daily, dli_daily


def compute_batch_features(batches_df, sensors_daily, dli_daily):
    """
    לכל אצווה: מחשב ממוצע חיישנים בפרק הזמן של האצווה.
    """
    # מיפוי חממה לקוד חיישן (אות ראשונה)
    gh_map = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D',
              'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J'}

    sensor_feat_cols = ['temp_int', 'humidity', 'radiation', 'temp_ext', 'humidity_ext']
    result_rows = []
    n_matched = 0

    for _, batch in batches_df.iterrows():
        row = {}
        gh    = str(batch.get('חממה', '')).strip().upper()
        start = batch.get('תאריך תחילת הפרחה')
        end   = batch.get('תאריך סיום הפרחה')

        if pd.isna(start) or pd.isna(end) or gh not in gh_map:
            result_rows.append(row)
            continue

        gh_code = gh_map[gh]

        # חיישנים
        if sensors_daily is not None:
            mask = (
                (sensors_daily['gh_code'] == gh_code) &
                (sensors_daily['date'] >= start) &
                (sensors_daily['date'] <= end)
            )
            sub = sensors_daily[mask]
            if len(sub) > 0:
                n_matched += 1
                for feat in sensor_feat_cols:
                    if feat in sub.columns:
                        vals = sub[feat].dropna()
                        if len(vals) > 0:
                            row[f'{feat}_mean'] = round(float(vals.mean()), 2)
                            row[f'{feat}_std']  = round(float(vals.std()),  2)

        # DLI
        if dli_daily is not None:
            mask_dli = (
                (dli_daily['gh_code'] == gh_code) &
                (dli_daily['date'] >= start) &
                (dli_daily['date'] <= end)
            )
            sub_dli = dli_daily[mask_dli]
            if len(sub_dli) > 0:
                vals = sub_dli['dli'].dropna()
                if len(vals) > 0:
                    row['dli_mean'] = round(float(vals.mean()), 2)
                    row['dli_std']  = round(float(vals.std()),  2)

        result_rows.append(row)

    print(f"  אצוות עם נתוני חיישנים: {n_matched}/{len(batches_df)}")
    return pd.DataFrame(result_rows, index=batches_df.index)


def compute_seasonal_averages(sensors_daily, dli_daily):
    """
    ממוצעים עונתיים לכל חממה × חודש — לשימוש בניבוי.
    מחזיר dict: {(gh_code, month): {feat: val, ...}}
    """
    seasonal = {}

    if sensors_daily is not None:
        s = sensors_daily.copy()
        s['month'] = s['date'].dt.month
        feat_cols = [c for c in s.columns if c not in ['date', 'gh_code', 'month']]
        for (gh, month), grp in s.groupby(['gh_code', 'month']):
            row = {}
            for feat in feat_cols:
                vals = grp[feat].dropna()
                if len(vals) > 0:
                    row[f'{feat}_mean'] = round(float(vals.mean()), 2)
            seasonal[(gh, month)] = row

    if dli_daily is not None:
        d = dli_daily.copy()
        d['month'] = d['date'].dt.month
        for (gh, month), grp in d.groupby(['gh_code', 'month']):
            vals = grp['dli'].dropna()
            if len(vals) > 0:
                if (gh, month) not in seasonal:
                    seasonal[(gh, month)] = {}
                seasonal[(gh, month)]['dli_mean'] = round(float(vals.mean()), 2)

    print(f"  ממוצעים עונתיים: {len(seasonal)} שילובי (חממה × חודש)")
    return seasonal


def build_training_dataset(save=True):
    """בניית דאטאסט אימון מלא."""
    print("=" * 55)
    print("בונה דאטאסט אימון עם נתוני חיישנים...")
    print(f"תיקיית נתונים: {DATA_RAW}")

    print("\n[1] טוען אצוות...")
    batches = load_batches()
    print(f"  {len(batches)} אצוות")

    print("\n[2] בונה ממוצעים יומיים...")
    sensors_daily, dli_daily = build_daily_sensors()

    print("\n[3] מחשב ממוצעים לכל אצווה...")
    sensor_features = compute_batch_features(batches, sensors_daily, dli_daily)

    print("\n[4] מחשב ממוצעים עונתיים לניבוי...")
    seasonal = compute_seasonal_averages(sensors_daily, dli_daily)

    # מיזוג
    df = pd.concat([batches.reset_index(drop=True), sensor_features.reset_index(drop=True)], axis=1)

    # קידוד
    df['חממה_קוד'] = pd.Categorical(df['חממה']).codes
    df['זן_קוד']   = pd.Categorical(df['זן']).codes
    df['עונה_קוד'] = pd.Categorical(df['עונה']).codes

    new_feat_cols = [c for c in df.columns if any(x in c for x in
                    ['temp_int', 'humidity', 'radiation', 'temp_ext', 'dli'])]
    print(f"\n  עמודות חיישנים חדשות: {new_feat_cols}")

    if save:
        # שמור ב-data/processed ו-app/ גם יחד
        for out_dir in [
            os.path.join(ROOT_DIR, 'data', 'processed'),
            os.path.join(ROOT_DIR, 'app'),
        ]:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'training_dataset.csv')
            df.to_csv(out_path, index=False)

        print(f"\n✅ נשמר: {len(df)} שורות, {len(df.columns)} עמודות")

    return df, seasonal


if __name__ == '__main__':
    df, seasonal = build_training_dataset()
    print("\nדוגמה:")
    print(df[[c for c in df.columns if any(x in c for x in ['חממה', 'זן', 'temp_int', 'humidity', 'dli', 'ימים'])]].head(5).to_string())
