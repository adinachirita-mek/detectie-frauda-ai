# 🔍 Detectia Fraudei pe Carduri de Credit folosind AI

Proiect academic pentru tema **"Supravegherea riscurilor financiare prin inteligenta artificiala"**.

## 🌐 Demo live

👉 **[Acceseaza aplicatia online aici](https://numele-tau.streamlit.app)**

*(inlocuieste linkul cu cel real dupa deploy)*

## 📋 Despre proiect

Implementare proprie pe directia **detectiei tranzactiilor frauduloase** folosind
3 modele de Machine Learning: Logistic Regression, Random Forest si XGBoost.

### Caracteristici
- 📊 Analiza exploratorie a datelor cu grafice interactive
- 🤖 Demo predictie live (introduci o tranzactie, modelul iti spune daca e frauda)
- 📈 Comparatie detaliata intre cele 3 modele (curbe ROC, PR, matrici de confuzie)
- 📄 Documentatie completa de 13 pagini

### Dataset
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle, ULB)
- 284.807 tranzactii reale, anonimizate prin PCA
- 492 fraude (0.172% — dezechilibru extrem)

### Tehnologii
- **Python 3** + scikit-learn + XGBoost
- **SMOTE** pentru echilibrarea claselor
- **Streamlit** + **Plotly** pentru interfata web
- Deploy pe **Streamlit Cloud** (gratuit)

## 📁 Structura proiectului

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
├── Documentatie.docx       # Documentatie completa (13 pagini)
├── GHID_DEPLOY.md          # Instructiuni pas-cu-pas pentru deploy
└── README.md               # Acest fisier
```

## 🚀 Deploy

Vezi [GHID_DEPLOY.md](GHID_DEPLOY.md) pentru instructiuni complete.

Pe scurt:
1. Antreneaza modelul in Google Colab → genereaza fisierele `.pkl`
2. Urca tot pe GitHub
3. Conecteaza repository-ul la [Streamlit Cloud](https://share.streamlit.io)
4. Primesti link public live

## 👤 Autor

**[Numele tau]**
Proiect academic 2025-2026

## 📄 Licenta

Proiect academic. Codul este liber pentru uz educational.
