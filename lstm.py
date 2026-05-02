import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, BatchNormalization
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight


# 1. Завантаження даних

field_names = pd.read_csv('NSL_KDD-master/Field Names.csv',
                          header=None, names=['name', 'type'])
columns = field_names['name'].tolist() + ['label', 'difficulty']

train_df = pd.read_csv('NSL_KDD-master/KDDTrain+.csv', names=columns)
test_df  = pd.read_csv('NSL_KDD-master/KDDTest+.csv',  names=columns)

train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty',  axis=1, inplace=True)


# 2. Binárizácia mítok

train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label']  = test_df['label'].apply( lambda x: 0 if x == 'normal' else 1)


# 3. Kódovanie kategorických príznakov

categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]], axis=0))
    train_df[col] = le.transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])


# 4. Per-class ordered split (analogicky k per-day pre CIC-IDS2017)

def per_type_split(df, train_ratio=0.8):
    train_parts, val_parts = [], []
    for label in sorted(df['label'].unique()):
        subset = df[df['label'] == label].reset_index(drop=True)
        n = len(subset)
        t_end = int(n * train_ratio)
        train_parts.append(subset.iloc[:t_end])
        val_parts.append(subset.iloc[t_end:])
        print(f"  Label={label}: total={n}, train={t_end}, val={n - t_end}")
    train = pd.concat(train_parts).sort_index()
    val   = pd.concat(val_parts).sort_index()
    return train, val

print("\n[NSL-KDD per-class ordered split]")
train_split, val_split = per_type_split(train_df, train_ratio=0.8)
print(f"  Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_df)}")
print(f"  Train attack ratio: {train_split['label'].mean()*100:.1f}%")
print(f"  Val   attack ratio: {val_split['label'].mean()*100:.1f}%")
print(f"  Test  attack ratio: {test_df['label'].mean()*100:.1f}%")


# 5. Normalizácia

feature_cols = [c for c in train_df.columns if c != 'label']
scaler = MinMaxScaler()

X_train_raw = scaler.fit_transform(train_split[feature_cols])
y_train_raw  = train_split['label'].values
X_val_raw    = scaler.transform(val_split[feature_cols])
y_val_raw    = val_split['label'].values
X_test_raw   = scaler.transform(test_df[feature_cols])
y_test_raw   = test_df['label'].values

input_dim = X_train_raw.shape[1]


# 6. Váhy tried

weights = compute_class_weight('balanced',
                               classes=np.unique(y_train_raw),
                               y=y_train_raw)
class_weights = {0: float(weights[0]), 1: float(weights[1])}
print(f"\nClass weights: {class_weights}")


# 7. Generovanie okien pre LSTM

def create_windows(X, y, window_size=5):
    X_win, y_win = [], []
    for i in range(len(X) - window_size):
        X_win.append(X[i:i + window_size])
        y_win.append(y[i + window_size - 1])
    return np.array(X_win), np.array(y_win)

WINDOW_SIZE = 5
X_train_lstm, y_train_lstm = create_windows(X_train_raw, y_train_raw, WINDOW_SIZE)
X_val_lstm,   y_val_lstm   = create_windows(X_val_raw,   y_val_raw,   WINDOW_SIZE)
X_test_lstm,  y_test_lstm  = create_windows(X_test_raw,  y_test_raw,  WINDOW_SIZE)

weights_lstm = compute_class_weight('balanced',
                                    classes=np.unique(y_train_lstm),
                                    y=y_train_lstm)
class_weights_lstm = {0: float(weights_lstm[0]), 1: float(weights_lstm[1])}

print(f"\nLSTM sequence shapes:")
print(f"  Train={X_train_lstm.shape}, Val={X_val_lstm.shape}, Test={X_test_lstm.shape}")


# 8. Pomocná funkcia na evaluáciu

def evaluate(model, X_test, y_test, model_name):
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\n{'='*50}")
    print(f"=== {model_name} — Classification Report ===")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'Attack']))

    roc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    fn   = int(((y_test == 1) & (y_pred == 0)).sum())
    fp   = int(((y_test == 0) & (y_pred == 1)).sum())
    tp   = int(((y_test == 1) & (y_pred == 1)).sum())
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"ROC-AUC:          {roc:.4f}")
    print(f"Avg Precision:    {ap:.4f}")
    print(f"F1 (Attack):      {f1:.4f}")
    print(f"FNR (Miss Rate):  {fnr*100:.2f}%")
    print(f"FP:               {fp}  |  FN: {fn}")

    return {'roc_auc': roc, 'avg_precision': ap, 'f1': f1,
            'fnr': fnr, 'fp': fp, 'fn': fn}


# 9. EXPERIMENT E5: MLP Baseline

print("\n" + "="*60)
print("EXPERIMENT E5: MLP Baseline (64-32, bez BatchNorm)")
print("="*60)

mlp_baseline = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1,  activation='sigmoid')
])
mlp_baseline.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
mlp_baseline.fit(X_train_raw, y_train_raw,
                 validation_data=(X_val_raw, y_val_raw),
                 epochs=10, batch_size=128, verbose=1,
                 class_weight=class_weights)

metrics_mlp = evaluate(mlp_baseline, X_test_raw, y_test_raw, "MLP Baseline (E5)")


# 10. ABLATION E5b: MLP Large (128-64, s Dropout)
# Analogicky k E2 pre CIC-IDS2017

print("\n" + "="*60)
print("ABLATION E5b: MLP Large (128-64, Dropout=0.3)")
print("="*60)

mlp_large = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64,  activation='relu'),
    Dense(1,   activation='sigmoid')
])
mlp_large.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
mlp_large.fit(X_train_raw, y_train_raw,
              validation_data=(X_val_raw, y_val_raw),
              epochs=10, batch_size=128, verbose=1,
              class_weight=class_weights)

metrics_mlp_large = evaluate(mlp_large, X_test_raw, y_test_raw, "MLP Large (E5b)")


# 11. EXPERIMENT E6: LSTM Baseline (window=5, lstm=64)


print("EXPERIMENT E6: LSTM Baseline (window=5, lstm=64)")

lstm_baseline = Sequential([
    Input(shape=(WINDOW_SIZE, input_dim)),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_baseline.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
lstm_baseline.fit(X_train_lstm, y_train_lstm,
                  validation_data=(X_val_lstm, y_val_lstm),
                  epochs=10, batch_size=128, verbose=1,
                  class_weight=class_weights_lstm)

metrics_lstm = evaluate(lstm_baseline, X_test_lstm, y_test_lstm, "LSTM Baseline (E6)")


# 12. ABLATION E6b: LSTM Large (lstm=128, Dense=32)
# Analogicky k E3 vs E2 pre CIC-IDS2017

print("\n" + "="*60)
print("ABLATION E6b: LSTM Large (window=5, lstm=128, Dense=32)")
print("="*60)

lstm_large = Sequential([
    Input(shape=(WINDOW_SIZE, input_dim)),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1,  activation='sigmoid')
])
lstm_large.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
lstm_large.fit(X_train_lstm, y_train_lstm,
               validation_data=(X_val_lstm, y_val_lstm),
               epochs=10, batch_size=128, verbose=1,
               class_weight=class_weights_lstm)

metrics_lstm_large = evaluate(lstm_large, X_test_lstm, y_test_lstm, "LSTM Large (E6b)")


# 13. Súhrnná tabuľka

import pandas as pd

all_results = {
    'E5: MLP Baseline':  metrics_mlp,
    'E5b: MLP Large':    metrics_mlp_large,
    'E6: LSTM Baseline': metrics_lstm,
    'E6b: LSTM Large':   metrics_lstm_large,
}

df_results = pd.DataFrame(all_results).T
print("\n" + "="*60)
print("=== NSL-KDD Model Comparison ===")
print(df_results[['roc_auc', 'avg_precision', 'f1', 'fnr', 'fp', 'fn']].to_string())
df_results.to_csv("nsl_kdd_comparison.csv", index=True)
print("\nResults saved to nsl_kdd_comparison.csv")