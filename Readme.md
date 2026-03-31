# Network Intrusion Detection — CIC-IDS2017

Detekcia sieťových útokov pomocou neurónových sietí (MLP a LSTM) na datasete CIC-IDS2017.  
Binárna klasifikácia: **Benign (0)** vs **Attack (1)**.

---

## Štruktúra projektu

```
├── data_preprocesing.py   # Načítanie, čistenie, škálovanie, split dát
├── models.py              # Definície MLP a LSTM modelov
├── train_eval.py          # Tréning, metriky, grafy, analýza chýb
├── main.py                # Hlavný skript — spúšťa všetky experimenty
├── requirements.txt       # Závislosti
└── README.md
```

---

## Inštalácia prostredia

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## Dataset

Stiahnite dataset **CIC-IDS2017** z:  
https://www.unb.ca/cic/datasets/ids-2017.html

Rozbaľte do priečinka `MachineLearningCVE/` v koreňovom adresári projektu:

```
MachineLearningCVE/
  Monday-WorkingHours.pcap_ISCX.csv
  Tuesday-WorkingHours.pcap_ISCX.csv
  ...
```

---

## Spustenie experimentov

```bash
python main.py
```

Skript automaticky spustí **4 experimenty**:

| Experiment | Model | Konfigurácia |
|---|---|---|
| 1 | MLP Baseline | dropout=0.2, units=(64,32) |
| 2 | MLP Large (ablation) | dropout=0.4, units=(128,64) |
| 3 | LSTM Advanced | window=10, lstm=(64,32) |
| 4 | LSTM Window=5 (ablation) | window=5, lstm=(64,32) |

---

## Výstupy

Pre každý model sa uložia:
- `{model}_learning_curves.png` — priebeh loss a accuracy
- `{model}_confusion_matrix.png` — matica zámen
- `{model}_pr_curve.png` — Precision-Recall krivka
- `model_comparison.csv` — porovnávacia tabuľka metrík

---

## Experimentálna metodológia

- **Rozdelenie dát:** Train 64% / Validation 16% / Test 20% (so stratifikáciou)
- **Metriky:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Average Precision
- **Nerovnováha tried:** riešená váhami tried (`class_weight='balanced'`)
- **Ablation study:** porovnanie veľkosti MLP a veľkosti okna LSTM

---

## Reprodukovateľnosť

Všetky experimenty používajú `random_state=42`. Výsledky sú plne reprodukovateľné pri rovnakom datasete a prostredí.