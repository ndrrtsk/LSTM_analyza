import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. Завантаження даних
# ============================================================
field_names = pd.read_csv('NSL_KDD-master/Field Names.csv',
                          header=None, names=['name', 'type'])
columns = field_names['name'].tolist() + ['label', 'difficulty']

train_df = pd.read_csv('NSL_KDD-master/KDDTrain+.csv', names=columns)
test_df  = pd.read_csv('NSL_KDD-master/KDDTest+.csv',  names=columns)

train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty',  axis=1, inplace=True)

# ============================================================
# 2. Binárizácia mítok
# ============================================================
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label']  = test_df['label'].apply( lambda x: 0 if x == 'normal' else 1)

# ============================================================
# 3. Kódovanie kategorických príznakov
# ============================================================
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    full_data = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(full_data)
    train_df[col] = le.transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])

# ============================================================
# 4. Normalizácia
# ============================================================
scaler = MinMaxScaler()
X_train_raw = scaler.fit_transform(train_df.drop('label', axis=1))
y_train_raw  = train_df['label'].values
X_test_raw   = scaler.transform(test_df.drop('label', axis=1))
y_test_raw   = test_df['label'].values

# ============================================================
# 5. Váhy tried
# ============================================================
weights = compute_class_weight('balanced',
                               classes=np.unique(y_train_raw),
                               y=y_train_raw)
class_weights = {0: weights[0], 1: weights[1]}

# ============================================================
# 6. Generovanie okien pre LSTM
# ============================================================
def create_windows(X, y, window_size=5):
    X_win, y_win = [], []
    for i in range(len(X) - window_size):
        X_win.append(X[i:i + window_size])
        y_win.append(y[i + window_size - 1])  # ✅ OPRAVENÝ BUG
    return np.array(X_win), np.array(y_win)

WINDOW_SIZE = 5
X_train_lstm, y_train_lstm = create_windows(X_train_raw, y_train_raw, WINDOW_SIZE)
X_test_lstm,  y_test_lstm  = create_windows(X_test_raw,  y_test_raw,  WINDOW_SIZE)

weights_lstm = compute_class_weight('balanced',
                                    classes=np.unique(y_train_lstm),
                                    y=y_train_lstm)
class_weights_lstm = {0: weights_lstm[0], 1: weights_lstm[1]}

# ============================================================
# 7. Modely
# ============================================================
mlp = Sequential([
    Input(shape=(X_train_raw.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

lstm = Sequential([
    Input(shape=(WINDOW_SIZE, X_train_raw.shape[1])),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ============================================================
# 8. Tréning
# ============================================================
print("Training MLP...")
mlp.fit(X_train_raw, y_train_raw,
        epochs=10, batch_size=128, verbose=1,
        class_weight=class_weights)

print("\nTraining LSTM...")
lstm.fit(X_train_lstm, y_train_lstm,
         epochs=10, batch_size=128, verbose=1,
         class_weight=class_weights_lstm)

# ============================================================
# 9. Evaluácia
# ============================================================
y_pred_mlp  = (mlp.predict(X_test_raw)   > 0.5).astype(int)
y_pred_lstm = (lstm.predict(X_test_lstm) > 0.5).astype(int)

print("\n=== MLP Classification Report ===")
print(classification_report(y_test_raw,   y_pred_mlp,
                             target_names=['Normal', 'Attack']))

print("\n=== LSTM Classification Report ===")
print(classification_report(y_test_lstm, y_pred_lstm,
                             target_names=['Normal', 'Attack']))

print("\n[NOTE] LSTM na NSL-KDD má obmedzený potenciál:")
print("  Záznamy v datasete sú i.i.d. — nemajú skutočný časový poriadok.")
print("  Okno 5 susedných riadkov nezachytáva reálnu sieťovú sekvenciu.")