import numpy as np  # FIX: було відсутнє — np.array у create_sequences падало
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, BatchNormalization


# --- Модель 1: MLP Baseline ---
def build_mlp(input_dim, dropout_rate=0.2, hidden_units=(64, 32)):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden_units[0], activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(hidden_units[1], activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- Модель 2: Advanced LSTM ---
def build_lstm(window_size, feature_dim, dropout_rate=0.2, lstm_units=(64, 32)):

    model = Sequential([
        Input(shape=(window_size, feature_dim)),
        LSTM(lstm_units[0], return_sequences=True),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(lstm_units[1], return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- Підготовка часових вікон для LSTM ---
def create_sequences(X, y, window_size=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i: i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)