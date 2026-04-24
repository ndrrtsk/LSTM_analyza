from data_preprocesing import load_and_clean_data, prepare_for_training,prepare_for_training_ordered
from models import build_mlp, build_lstm, create_sequences
from train_eval import train_and_evaluate, analyze_errors, compare_models
import numpy as np

# ============================================================
# 1. Завантаження та очищення даних
# ============================================================
DATA_PATH = 'MachineLearningCVE'
df = load_and_clean_data(DATA_PATH)

# ============================================================
# 2. Попередня обробка та розподіл Train / Val / Test (64/16/20)
# ============================================================
X_train, X_val, X_test, y_train, y_val, y_test, feat_names = prepare_for_training(df)

# LSTM використовує часово впорядковані дані
X_tr_ord, X_va_ord, X_te_ord, y_tr_ord, y_va_ord, y_te_ord, _ = prepare_for_training_ordered(df)
input_dim = X_train.shape[1]
print(f"\nDataset shapes:")
print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"  Features: {input_dim}")
all_results = {}

# ============================================================
# ЕКСПЕРИМЕНТ 1: MLP Baseline (стандартна конфігурація)
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 1: MLP Baseline")
print("="*60)
mlp_model = build_mlp(input_dim, dropout_rate=0.2, hidden_units=(64, 32))
y_pred_mlp, y_prob_mlp, metrics_mlp = train_and_evaluate(
    mlp_model, X_train, y_train, X_val, y_val, X_test, y_test,
    model_name="MLP_Baseline", epochs=15, batch_size=256
)
analyze_errors(y_test, y_pred_mlp, model_name="MLP_Baseline")
all_results['MLP_Baseline'] = metrics_mlp

# ============================================================
# ABLATION STUDY: MLP з більшою мережею та більшим dropout
# Вимога Pokyny: мінімум 2 конфігурації гіперпараметрів
# ============================================================
print("\n" + "="*60)
print("ABLATION STUDY: MLP Large (128-64, dropout=0.4)")
print("="*60)
mlp_large = build_mlp(input_dim, dropout_rate=0.4, hidden_units=(128, 64))
y_pred_mlp_lg, y_prob_mlp_lg, metrics_mlp_lg = train_and_evaluate(
    mlp_large, X_train, y_train, X_val, y_val, X_test, y_test,
    model_name="MLP_Large", epochs=15, batch_size=256
)
analyze_errors(y_test, y_pred_mlp_lg, model_name="MLP_Large")
all_results['MLP_Large'] = metrics_mlp_lg

# ============================================================
# ЕКСПЕРИМЕНТ 2: LSTM Advanced
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 2: LSTM Advanced")
print("="*60)
WINDOW_SIZE = 10

print("Creating sequences for LSTM...")
X_train_seq, y_train_seq = create_sequences(X_tr_ord, y_tr_ord, WINDOW_SIZE)
X_val_seq,   y_val_seq   = create_sequences(X_va_ord,   y_va_ord,   WINDOW_SIZE)
X_test_seq,  y_test_seq  = create_sequences(X_te_ord,  y_te_ord,  WINDOW_SIZE)

print(f"LSTM sequence shapes: Train={X_train_seq.shape}, Test={X_test_seq.shape}")

lstm_model = build_lstm(WINDOW_SIZE, input_dim, dropout_rate=0.2, lstm_units=(64, 32))
y_pred_lstm, y_prob_lstm, metrics_lstm = train_and_evaluate(
    lstm_model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq,
    model_name="LSTM_Advanced", epochs=15, batch_size=256
)
analyze_errors(y_test_seq, y_pred_lstm, model_name="LSTM_Advanced")
all_results['LSTM_Advanced'] = metrics_lstm

# ============================================================
# ABLATION: LSTM з меншим вікном (window=5 vs 10)
# ============================================================
print("\n" + "="*60)
print("ABLATION STUDY: LSTM Window=5")
print("="*60)
WINDOW_SIZE_SMALL = 5

X_train_seq, y_train_seq = create_sequences(X_tr_ord, y_tr_ord, WINDOW_SIZE_SMALL)
X_val_seq,   y_val_seq   = create_sequences(X_va_ord, y_va_ord, WINDOW_SIZE_SMALL)
X_test_seq,  y_test_seq  = create_sequences(X_te_ord, y_te_ord, WINDOW_SIZE_SMALL)
lstm_small = build_lstm(WINDOW_SIZE_SMALL, input_dim, dropout_rate=0.2, lstm_units=(64, 32))
y_pred_lstm_s, y_prob_lstm_s, metrics_lstm_s = train_and_evaluate(
    lstm_small, X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq,
    model_name="LSTM_Window5", epochs=15, batch_size=256
)
analyze_errors(y_test_seq, y_pred_lstm_s, model_name="LSTM_Window5")
all_results['LSTM_Window5'] = metrics_lstm_s

# ============================================================
# 3. Підсумкова таблиця порівняння моделей
# ============================================================
print("\n" + "="*60)
comparison_df = compare_models(all_results)
comparison_df.to_csv("model_comparison.csv", index=True)
print("\nResults saved to model_comparison.csv")
