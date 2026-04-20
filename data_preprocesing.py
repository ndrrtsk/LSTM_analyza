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
    """
    Зчитує всі CSV файли з папки MachineLearningCVE,
    об'єднує їх, очищає та повертає DataFrame з бінарними мітками.
    """
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    df_list = []
    print(f"Loading {len(all_files)} files from '{folder_path}'...")
    for filename in all_files:
        data = pd.read_csv(filename, low_memory=False)
        df_list.append(data)
        print(f"  Loaded: {os.path.basename(filename)} — {len(data)} rows")

    df = pd.concat(df_list, ignore_index=True)
    print(f"\nTotal rows after merge: {df.shape[0]}")

    # Очищення назв колонок (видалення зайвих пробілів)
    df.columns = df.columns.str.strip()

    # Заміна inf → NaN (характерна проблема CIC-IDS2017)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Видалення рядків з NaN
    initial_shape = df.shape[0]
    df.dropna(inplace=True)
    print(f"Removed {initial_shape - df.shape[0]} rows with NaN/Inf.")

    # Видалення дублюючих колонок
    cols_to_drop = ['Fwd Header Length.1']
    df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)

    # Бінарна класифікація: Benign=0, Attack=1
    df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    # df_attacks = df[df['Label'] == 1]
    # df_benign = df[df['Label'] == 0].sample(frac=0.2, random_state=42)

    # df = pd.concat([df_attacks, df_benign]).sample(frac=1).reset_index(drop=True)
    # print(f"New dataset size: {len(df)} rows")

    # FIX: CIC-IDS2017 використовує 'BENIGN' (великі літери), не 'Benign'
    # Перевіряємо баланс класів
    print(f"\nClass distribution:")
    print(df['Label'].value_counts())
    print(f"Attack ratio: {df['Label'].mean()*100:.1f}%")
    df = reduce_mem_usage(df)
    return df


def prepare_for_training(df):
    """
    Видаляє константні ознаки, масштабує дані та ділить на
    Train (64%) / Validation (16%) / Test (20%) з стратифікацією.
    """
    X = df.drop('Label', axis=1)
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
        random_state=42, shuffle=False
    )

    print(f"\nSplit sizes:")
    print(f"  Train:      {X_train.shape[0]} samples ({X_train.shape[0]/len(y)*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(y)*100:.1f}%)")
    print(f"  Test:       {X_test.shape[0]} samples ({X_test.shape[0]/len(y)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns