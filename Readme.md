# Detekcia sieťových útokov pomocou neurónových sietí

Binárna klasifikácia sieťových tokov (**Benign vs. Attack**) pomocou MLP a LSTM architektúr na datasetoch CIC-IDS2017 a NSL-KDD.

Projekt Neurónové siete 2025/2026 — Sysak · Zemlyanskiy · Synenko · Rytskyi

---

## Štruktúra projektu

```
├── data_preprocesing.py   # Načítanie CSV, čistenie, normalizácia, per-day/per-class split
├── models.py              # Definície MLP Baseline, MLP Large, LSTM Baseline, LSTM Large
├── train_eval.py          # Tréning, metriky, grafy, analýza chýb
├── main.py                # Hlavný skript — CIC-IDS2017 experimenty (E1–E4)
├── lstm.py            # Hlavný skript — NSL-KDD experimenty (E5, E5b, E6, E6b)
├── requirements.txt       # Python závislosti
└── README.md
```

---

## Inštalácia

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Požiadavky

```
tensorflow[and-cuda]>=2.20.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
```

> **HPC Perun:** GPU je dostupná po `pip install tensorflow[and-cuda]` bez `module load cuda`. SLURM skript: `#SBATCH --gres=gpu:1 --partition=GPU`.  
> **Google Colab:** Window=10 spôsobí OOM (peak ~18 GB RAM). Používajte Window=5 s prealokáciou polí (peak ~10 GB).

---

## Datasety

### CIC-IDS2017

Stiahnite z: https://www.unb.ca/cic/datasets/ids-2017.html

Rozbaľte priečinok `MachineLearningCVE/` do koreňového adresára projektu:

```
MachineLearningCVE/
  Monday-WorkingHours.pcap_ISCX.csv
  Tuesday-WorkingHours.pcap_ISCX.csv
  Wednesday-WorkingHours.pcap_ISCX.csv
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  Friday-WorkingHours-Morning.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

| Vlastnosť | Hodnota |
|---|---|
| Vzorky (po čistení) | 2 827 876 |
| Príznaky | 78 → 69 (po odstránení konštantných) |
| Triedy | 80.3 % Benign / 19.7 % Attack |
| Typy útokov | DoS, DDoS, Brute Force, XSS, SQL Injection, Infiltration, Botnet, PortScan |

### NSL-KDD

Stiahnite z: https://www.unb.ca/cic/datasets/nsl.html

Umiestnite do priečinka `NSL-KDD/`:

```
NSL-KDD/
  KDDTrain+.csv
  KDDTest+.csv
  Field Names.csv
```

| Vlastnosť | Hodnota |
|---|---|
| Tréning | 125 973 záznamov → 100 778 train / 25 195 val (per-class ordered split) |
| Test | 22 544 záznamov (obsahuje unseen attack typy!) |
| Príznaky | 41 (38 numerických, 3 kategorické) |
| Typy útokov | DoS, Probe, R2L, U2R |

---

## Spustenie experimentov

### Fáza 1 — CIC-IDS2017 (E1–E4)

```bash
python main.py
```

### Fáza 2 — NSL-KDD (E5, E5b, E6, E6b)

```bash
python lstm.py
```

---

## Experimenty

### Fáza 1: CIC-IDS2017 — GPU NVIDIA H200, per-day stratifikovaný split

| Exp. | Model | Konfigurácia | F1 | ROC-AUC | FNR |
|---|---|---|---|---|---|
| E1 | MLP Baseline | units=(64,32), Dropout=0.2, BatchNorm, epochs=15, batch=256 | 0.9504 | 0.9985 | 0.55 % |
| E2 | MLP Large | units=(128,64), Dropout=0.4, BatchNorm, epochs=15, batch=256 | 0.9490 | 0.9986 | 0.60 % |
| E3 | LSTM Win=10 | lstm=(64,32), window=10, Dropout=0.2, BatchNorm, epochs=15, batch=256 | 0.9477 | 0.9821 | 9.24 % |
| E4 | LSTM Win=5 | lstm=(64,32), window=5, Dropout=0.2, BatchNorm, epochs=15, batch=256 | 0.9455 | 0.9825 | 9.10 % |

**Kľúčové zistenie:** MLP minimalizuje FNR (0.55 %), LSTM minimalizuje FPR — 23× menej falošných poplachov (581 vs. 10 936).

### Fáza 2: NSL-KDD — CPU, per-class ordered split

| Exp. | Model | Konfigurácia | F1 | ROC-AUC | FNR | FN |
|---|---|---|---|---|---|---|
| E5 | MLP Baseline | units=(64,32), epochs=10, batch=128 | 0.7628 | 0.9463 | 36.83 % | 4 727 |
| E5b | MLP Large ★ | units=(128,64), Dropout=0.3, epochs=10, batch=128 | **0.7983** | **0.9503** | **32.27 %** | **4 141** |
| E6 | LSTM Baseline | lstm=(64), window=5, Dropout=0.2, epochs=10, batch=128 | 0.7604 | 0.8651 | 34.72 % | 4 454 |
| E6b | LSTM Large | lstm=(128)+Dense(32), window=5, Dropout=0.2, epochs=10, batch=128 | 0.7350 | 0.8857 | 39.81 % | 5 107 |

**Kľúčové zistenie:** Na NSL-KDD platí opak oproti CIC-IDS2017 — väčší MLP pomáha (underfitting), väčší LSTM škodí (overfit na i.i.d. dátach).

---

## Metodológia

### Rozdelenie datasetu

Voľba správnej metodológie rozdelenia je kritická pre LSTM modely:

| Metóda | Určenie | ROC-AUC (LSTM) |
|---|---|---|
| `shuffle=True` | MLP (oba datasety) | — |
| Globálny chronologický split |  nefunkčné — domain shift | 0.90 |
| **Per-day stratifikovaný split** |  LSTM na CIC-IDS2017 | **0.9821** |
| **Per-class ordered split** |  LSTM/MLP na NSL-KDD | — |

> **Per-day split:** Z každého CSV súboru sa vyčlení 64 % / 16 % / 20 % riadkov v chronologickom poradí. Každá množina tak obsahuje záznamy zo všetkých dní a všetkých typov útokov.

> **Per-class ordered split:** Z každej triedy (Normal / Attack) sa vyčlení 80 % tréning / 20 % validácia v poradí záznamu v CSV súbore.

### Off-by-one bug — create_sequences / create_windows

Pôvodná bugovaná implementácia priradila mítku vzorke *za* oknom:

```python
#  CHYBNÉ — mítka patrí vzorke za oknom
y_win[i] = y[i + window_size]

#  SPRÁVNE — mítka patrí poslednej vzorke okna
y_win[i] = y[i + window_size - 1]
```

Dôsledok bugu: TP = 0, ROC-AUC ≈ 0.5 (model predikoval výhradne Benign).

### Triedy a váhy

Nerovnováha tried je riešená váhami (nie over/undersampling):

```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
# CIC-IDS2017: w0=0.621, w1=2.571  (útok váži 4.1× viac)
# NSL-KDD:     w0=0.935, w1=1.074  (takmer vyvážené)
```

---

## Pamäťová náročnosť (LSTM, CIC-IDS2017)

| Komponent | Window=10 | Window=5 |
|---|---|---|
| Sekvencie X (float32) | 6.80 GB | 3.41 GB |
| Raw arrays + OS + Pandas | ~4.25 GB | ~4.25 GB |
| Spike pri `np.array(list)` | +5.00 GB | +2.50 GB |
| **Peak celkovo** | **~16–18 GB** | **~9–11 GB** |

Spike eliminujte priamou prealokáciou:

```python
N = len(X) - window_size
X_win = np.empty((N, window_size, X.shape[1]), dtype=np.float32)
y_win = np.empty(N, dtype=np.float32)
for i in range(N):
    X_win[i] = X[i:i + window_size]
    y_win[i] = y[i + window_size - 1]   # off-by-one fix
```

---

## Výstupy

Pre každý model sa uložia:

```
{model}_learning_curves.png    # loss a accuracy počas tréningu
{model}_confusion_matrix.png   # matica zámen (TN/FP/FN/TP)
{model}_pr_curve.png           # Precision-Recall krivka (Average Precision)
model_comparison.csv           # porovnávacia tabuľka všetkých metrík
```

---

## Produkčné odporúčania

| Scenár | Odporúčaný model | Dôvod |
|---|---|---|
| Zero-FN politika | **MLP Baseline / CIC** | FNR = 0.55 %, zachytí 99.45 % útokov |
| Alert fatigue | **LSTM Win=5 / CIC** | FPR = 0.11 %, 23× menej falošných poplachov |
| NSL-KDD typ datasetu | **MLP Large / NSL** | Najlepší F1 = 0.7983, FNR = 32.27 % |
| Obmedzená pamäť | **LSTM Win=5** | Úspora ~2.5 GB RAM oproti Win=10, rovnaký výkon |

---

## Reprodukovateľnosť

Všetky experimenty používajú `random_state=42`. Výsledky sú plne reprodukovateľné pri rovnakom datasete a prostredí (HPC Perun, NVIDIA H200, TF 2.20.0, 1. mája 2026).
