import numpy as np  # FIX: було відсутнє
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, f1_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test,
                       model_name="Model", epochs=15, batch_size=256):
    """
    Тренує модель та повертає передбачення і метрики.
    epochs та batch_size винесені як параметри для ablation study.
    """
    print(f"\n{'='*50}")
    print(f"Starting training: {model_name}")
    print(f"{'='*50}")

    # Ваги класів для боротьби з дисбалансом датасету CIC-IDS2017
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: float(weights[i]) for i in range(len(weights))}
    print(f"Class weights: {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        verbose=1
    )

    # --- Графіки навчання ---
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_learning_curves.png', dpi=150)
    plt.show()

    # --- Передбачення ---
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # --- Метрики ---
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = ['Benign', 'Attack']
    current_target_names = [target_names[int(i)] for i in unique_labels]

    print(f"\n=== {model_name} — Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=current_target_names))

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    ap = average_precision_score(y_test, y_pred_prob)
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"Average Precision:{ap:.4f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=current_target_names,
                yticklabels=current_target_names)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_confusion_matrix.png', dpi=150)
    plt.show()

    # --- Precision-Recall крива ---
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_pr_curve.png', dpi=150)
    plt.show()

    return y_pred, y_pred_prob, {
        'roc_auc': roc_auc,
        'avg_precision': ap,
        'f1': f1_score(y_test, y_pred)
    }


def analyze_errors(y_test, y_pred, model_name="Model"):
    """
    Детальний аналіз помилок — обов'язковий розділ звіту згідно з Pokyny.
    Виводить кількість FP, FN та їх частку.
    """
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()

    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))  # False Positive: Benign → Attack
    fn = np.sum((y_test == 1) & (y_pred == 0))  # False Negative: Attack → Benign
    tp = np.sum((y_test == 1) & (y_pred == 1))

    total = len(y_test)
    print(f"\n=== {model_name} — Error Analysis ===")
    print(f"Total samples:      {total}")
    print(f"True Positives:     {tp}  ({tp/total*100:.2f}%)")
    print(f"True Negatives:     {tn}  ({tn/total*100:.2f}%)")
    print(f"False Positives:    {fp}  ({fp/total*100:.2f}%)  ← Benign classified as Attack")
    print(f"False Negatives:    {fn}  ({fn/total*100:.2f}%)  ← Attack missed (most dangerous!)")

    print(f"\nFalse Negative Rate (Miss Rate): {fn/(fn+tp)*100:.2f}%")
    print(f"False Positive Rate:             {fp/(fp+tn)*100:.2f}%")

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def compare_models(results: dict):
    """
    Порівняльна таблиця всіх моделей — для розділу Results у звіті.
    results: {'ModelName': {'roc_auc': ..., 'avg_precision': ..., 'f1': ...}}
    """
    df = pd.DataFrame(results).T
    print("\n=== Model Comparison ===")
    print(df.to_string())
    return df