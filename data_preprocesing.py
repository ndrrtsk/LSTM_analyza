import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def load_and_clean_data(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday-Morning', 'Thursday-Afternoon',
                 'Friday-Morning', 'Friday-Afternoon-DDos',
                 'Friday-Afternoon-PortScan']
    all_files = sorted(
        all_files,
        key=lambda f: next(
            (i for i, d in enumerate(DAY_ORDER)
             if d in os.path.basename(f)), 99
        )
    )

    df_list = []
    print(f"Loading {len(all_files)} files from '{folder_path}'...")
    for filename in all_files:
        data = pd.read_csv(filename, low_memory=False)
        data['_source_file'] = os.path.basename(filename)  # ✅ зберігаємо день
        df_list.append(data)
        print(f"  Loaded: {os.path.basename(filename)} — {len(data)} rows")

    df = pd.concat(df_list, ignore_index=True)
    print(f"\nTotal rows after merge: {df.shape[0]}")

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_shape = df.shape[0]
    df.dropna(inplace=True)
    print(f"Removed {initial_shape - df.shape[0]} rows with NaN/Inf.")

    cols_to_drop = ['Fwd Header Length.1']
    df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)

    df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)

    print(f"\nClass distribution:")
    print(df['Label'].value_counts())
    print(f"Attack ratio: {df['Label'].mean()*100:.1f}%")
    df = reduce_mem_usage(df)
    return df

def prepare_for_training_ordered(df):
    X = df.drop(['Label', '_source_file'], axis=1)
    y = df['Label']
    source = df['_source_file']

    # Odstránenie konštantných stĺpcov
    non_constant = (X != X.iloc[0]).any()
    dropped = X.columns[~non_constant].tolist()
    if dropped:
        print(f"Dropped constant columns: {dropped}")
    X = X.loc[:, non_constant]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=df.index)

    train_idx, val_idx, test_idx = [], [], []

    for fname in source.unique():
        mask = (source == fname)
        idx = df.index[mask].tolist()
        n = len(idx)
        t_end = int(n * 0.64)
        v_end = int(n * 0.80)

        train_idx.extend(idx[:t_end])
        val_idx.extend(idx[t_end:v_end])
        test_idx.extend(idx[v_end:])

        labels = y[idx].value_counts().to_dict()
        print(f"  {fname[:30]:30s} → "
              f"train={t_end}, val={v_end-t_end}, "
              f"test={n-v_end}  |  labels={labels}")

    X_train = X_scaled.loc[train_idx].values
    X_val   = X_scaled.loc[val_idx].values
    X_test  = X_scaled.loc[test_idx].values
    y_train = y.loc[train_idx].values
    y_val   = y.loc[val_idx].values
    y_test  = y.loc[test_idx].values

    n = len(df)
    print(f"\n[LSTM per-day ordered split]")
    print(f"  Train:      {len(X_train)} ({len(X_train)/n*100:.1f}%)")
    print(f"  Validation: {len(X_val)}   ({len(X_val)/n*100:.1f}%)")
    print(f"  Test:       {len(X_test)}  ({len(X_test)/n*100:.1f}%)")

    # Distribúcia tried v každej množine
    print(f"\n  Train attack ratio:  {y_train.mean()*100:.1f}%")
    print(f"  Val   attack ratio:  {y_val.mean()*100:.1f}%")
    print(f"  Test  attack ratio:  {y_test.mean()*100:.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns


def prepare_for_training(df):
    """
    Видаляє константні ознаки, масштабує дані та ділить на
    Train (64%) / Validation (16%) / Test (20%) з стратифікацією.
    """
    X = df.drop(['Label', '_source_file'], axis=1)
    y = df['Label']

    # Видалення колонок з нульовою дисперсією (константи)
    non_constant = (X != X.iloc[0]).any()
    dropped = X.columns[~non_constant].tolist()
    if dropped:
        print(f"Dropped constant columns: {dropped}")
    X = X.loc[:, non_constant]

    # Масштабування [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Спочатку відділяємо Test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
    )
    # З решти відділяємо Validation (20% від 80% = 16% загалом)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val,
        random_state=42, shuffle=True
    )

    print(f"\nSplit sizes:")
    print(f"  Train:      {X_train.shape[0]} samples ({X_train.shape[0]/len(y)*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(y)*100:.1f}%)")
    print(f"  Test:       {X_test.shape[0]} samples ({X_test.shape[0]/len(y)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns