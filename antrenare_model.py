"""
Script de antrenare - se ruleaza O SINGURA DATA pentru a genera fisierele:
  - model.pkl       (modelul XGBoost antrenat)
  - scaler.pkl      (scalerul pentru Time si Amount)
  - rezultate.pkl   (metricile, curbele ROC/PR pentru toate modelele)

Acest script se ruleaza in Google Colab (gratuit, in browser).
Dupa rulare, descarci cele 3 fisiere .pkl si le pui in repository-ul GitHub
alaturi de app.py.
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("1. Incarc datele...")
df = pd.read_csv("creditcard.csv")
print(f"   {len(df):,} tranzactii, {df['Class'].sum()} fraude")

print("2. Preprocesare...")
scaler_amount = StandardScaler()
scaler_time = StandardScaler()
df["Amount"] = scaler_amount.fit_transform(df[["Amount"]])
df["Time"] = scaler_time.fit_transform(df[["Time"]])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("3. Aplic SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("4. Antrenez 3 modele...")
modele = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                             use_label_encoder=False, eval_metric="logloss", n_jobs=-1),
}

rezultate = {}
for nume, model in modele.items():
    print(f"   Antrenez {nume}...")
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    rezultate[nume] = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": prec_curve.tolist(),
        "recall_curve": rec_curve.tolist(),
    }
    print(f"      F1={rezultate[nume]['f1']:.3f}, ROC-AUC={rezultate[nume]['roc_auc']:.3f}")

# Salvam doar modelul XGBoost (cel mai performant) pentru predictii live
print("5. Salvez fisierele...")
with open("model.pkl", "wb") as f:
    pickle.dump(modele["XGBoost"], f)

with open("scaler_amount.pkl", "wb") as f:
    pickle.dump(scaler_amount, f)

with open("scaler_time.pkl", "wb") as f:
    pickle.dump(scaler_time, f)

with open("rezultate.pkl", "wb") as f:
    pickle.dump(rezultate, f)

# Salvam si cateva statistici despre dataset pentru afisare
stats = {
    "total": len(df),
    "fraude": int(df["Class"].sum()),
    "legitime": int((df["Class"] == 0).sum()),
    "procent_fraude": float(df["Class"].mean() * 100),
    "suma_medie_legitima": float(pd.read_csv("creditcard.csv").query("Class==0")["Amount"].mean()),
    "suma_medie_frauda": float(pd.read_csv("creditcard.csv").query("Class==1")["Amount"].mean()),
}
with open("stats.pkl", "wb") as f:
    pickle.dump(stats, f)

print("\nGATA! Fisiere generate:")
print("  - model.pkl")
print("  - scaler_amount.pkl")
print("  - scaler_time.pkl")
print("  - rezultate.pkl")
print("  - stats.pkl")
print("\nDescarca-le si pune-le in repository-ul GitHub alaturi de app.py")
