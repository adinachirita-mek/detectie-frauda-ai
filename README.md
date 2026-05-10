# Detectia Fraudei pe Carduri de Credit folosind AI

Practica de cercetare pentru tema **"Supravegherea riscurilor financiare prin inteligenta artificiala"**

## Demo live

Acceseaza aplicatia online: `https://detectie-frauda-ai-9lscbrde3bltz9appt3wngg.streamlit.app`

## Despre proiect

Implementare proprie pe directia **detectiei tranzactiilor frauduloase** folosind
3 modele de Machine Learning: Logistic Regression, Random Forest si XGBoost.

### Caracteristici
- Analiza exploratorie a datelor cu grafice interactive
- Demo predictie live (introduci o tranzactie, modelul iti spune daca e frauda)
- Comparatie detaliata intre cele 3 modele (curbe ROC, PR, matrici de confuzie)
- Documentatie completa de 13 pagini

### Dataset
Credit Card Fraud Detection (Kaggle, ULB)
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284.807 tranzactii reale, anonimizate prin PCA
- 492 fraude (0.172% — dezechilibru extrem)

### Tehnologii
- Python 3 + scikit-learn + XGBoost
- SMOTE pentru echilibrarea claselor
- Streamlit + Plotly pentru interfata web
- Deploy pe Streamlit Cloud

## Structura proiectului

```
.
├── app.py                  # Aplicatia Streamlit (interfata web)
├── antrenare_model.py      # Script de antrenare (rulat o data in Colab)
├── requirements.txt        # Pachete Python necesare
├── .streamlit/
│   └── config.toml         # Configurare tema Streamlit
├── model.pkl               # Modelul XGBoost antrenat
├── scaler.pkl              # StandardScaler pentru Time si Amount
├── rezultate.pkl           # Metrici si curbe ROC/PR pre-calculate
├── stats.pkl               # Statistici dataset
└── Documentatie.docx       # Documentatie completa (13 pagini)
```


Practica de cercetare 2025-2026
