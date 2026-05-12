"""
Detectia fraudei pe carduri de credit - Aplicatie Streamlit
Site interactiv care prezinta proiectul si permite predictii live.
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ====================================================================
# CONFIGURARE PAGINA
# ====================================================================
st.set_page_config(
    page_title="Detectie Frauda AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizat - DARK MODE cu accente albastre/verzi
st.markdown("""
<style>
    /* Header principal - gradient albastru-verde */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00D4AA 0%, #4A9EFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #8B95A7;
        font-style: italic;
        margin-bottom: 2rem;
    }
    /* Carduri pentru statistici - fundal intunecat cu glow verde */
    .metric-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #0E1117 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #E4E8F0;
        text-align: center;
        border: 1px solid #2A3142;
        box-shadow: 0 0 20px rgba(0, 212, 170, 0.15);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #00D4AA;
        box-shadow: 0 0 30px rgba(0, 212, 170, 0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: #00D4AA;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8B95A7;
        margin: 0;
    }
    /* Tab-uri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #1A1F2E;
        border-radius: 8px 8px 0 0;
        color: #8B95A7;
        border: 1px solid #2A3142;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00D4AA 0%, #4A9EFF 100%) !important;
        color: #0E1117 !important;
        font-weight: 600;
        border: none !important;
    }
    /* Box rezultat frauda - rosu cu glow */
    .result-box-fraud {
        background: linear-gradient(135deg, #2A1518 0%, #1A0A0D 100%);
        color: #FF6B7A;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #E74C3C;
        box-shadow: 0 0 30px rgba(231, 76, 60, 0.3);
    }
    /* Box rezultat legitim - verde cu glow */
    .result-box-legit {
        background: linear-gradient(135deg, #0D2A1F 0%, #0A1A14 100%);
        color: #00D4AA;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #00D4AA;
        box-shadow: 0 0 30px rgba(0, 212, 170, 0.3);
    }
    /* Info box */
    .info-box {
        background-color: #1A1F2E;
        border-left: 4px solid #00D4AA;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #E4E8F0;
    }
    .info-box b {
        color: #00D4AA;
    }
    /* Pipeline pasi */
    .pipeline-step {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1A1F2E 0%, #0E1117 100%);
        border-radius: 8px;
        border: 1px solid #2A3142;
        color: #E4E8F0;
    }
    /* Butoane */
    .stButton > button {
        background-color: #1A1F2E;
        color: #E4E8F0;
        border: 1px solid #2A3142;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        border-color: #00D4AA;
        color: #00D4AA;
        box-shadow: 0 0 15px rgba(0, 212, 170, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# INCARCARE MODEL SI DATE
# ====================================================================
@st.cache_resource
def incarca_resurse():
    """Incarca modelul, scaler-ul si rezultatele salvate."""
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler_amount.pkl", "rb") as f:
            scaler_amount = pickle.load(f)
        with open("scaler_time.pkl", "rb") as f:
            scaler_time = pickle.load(f)
        with open("rezultate.pkl", "rb") as f:
            rezultate = pickle.load(f)
        with open("stats.pkl", "rb") as f:
            stats = pickle.load(f)
        return model, scaler_amount, scaler_time, rezultate, stats, True
    except FileNotFoundError:
        return None, None, None, None, None, False

model, scaler_amount, scaler_time, rezultate, stats, fisiere_ok = incarca_resurse()

# ====================================================================
# SIDEBAR - NAVIGARE
# ====================================================================
with st.sidebar:
    st.markdown("### Detectie Frauda AI")
    st.markdown("---")
    st.markdown("**Proiect academic**")
    st.markdown("Supravegherea riscurilor financiare prin inteligenta artificiala")
    st.markdown("---")

    st.markdown("**Tehnologii**")
    st.markdown("- Python 3")
    st.markdown("- scikit-learn")
    st.markdown("- XGBoost")
    st.markdown("- SMOTE (imblearn)")
    st.markdown("- Streamlit + Plotly")

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
    st.markdown("Sursa: Kaggle (ULB)")

    if not fisiere_ok:
        st.markdown("---")
        st.error("Modelul nu este incarcat. Ruleaza intai antrenare_model.py")

# ====================================================================
# HEADER PRINCIPAL
# ====================================================================
st.markdown('<p class="main-header">Detectia Fraudei pe Carduri de Credit</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Supravegherea riscurilor financiare prin Inteligenta Artificiala</p>', unsafe_allow_html=True)

# ====================================================================
# TAB-URI
# ====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Acasa",
    "Analiza datelor",
    "Demo predictie",
    "Rezultate modele",
    "Despre proiect"
])

# ====================================================================
# TAB 1: ACASA
# ====================================================================
with tab1:
    st.markdown("### Despre proiect")
    st.markdown("""
    Acest proiect aplica tehnici de **Machine Learning** pentru detectia tranzactiilor
    frauduloase efectuate cu carduri de credit. Tema generala — *Supravegherea riscurilor
    financiare prin AI* — acopera trei directii: **detectia fraudei**, **anti-spalarea
    banilor (AML)** si **gestionarea portofoliilor**. Implementarea proprie se concentreaza
    pe prima directie, alegand un set de date public si o problema concreta.
    """)

    st.markdown("### Statistici cheie ale datasetului")

    col1, col2, col3, col4 = st.columns(4)

    if fisiere_ok:
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{stats['total']:,}</p>
                <p class="metric-label">Total tranzactii</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{stats['fraude']}</p>
                <p class="metric-label">Tranzactii frauduloase</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{stats['procent_fraude']:.3f}%</p>
                <p class="metric-label">Procent fraude</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">3</p>
                <p class="metric-label">Modele comparate</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Modelul nu este incarcat — afisez valori demonstrative.")

    st.markdown("### Provocarea principala")
    st.markdown("""
    <div class="info-box">
    <b>Dezechilibrul de clase</b> este principala provocare a acestei probleme. Doar <b>0.172%</b>
    din tranzactii sunt frauduloase. Un model care prezice mereu „legitim" ar avea o acuratete
    de 99.83%, dar ar fi inutil in practica. De aceea folosim metrici specializate
    (Precision, Recall, F1, PR-AUC) si tehnici de echilibrare a claselor (SMOTE).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Pipeline-ul proiectului")
    cols = st.columns(5)
    pasi = [
        ("1", "Incarcare\ndate"),
        ("2", "Preprocesare\n+ Scalare"),
        ("3", "SMOTE\nechilibrare"),
        ("4", "Antrenare\n3 modele"),
        ("5", "Evaluare\n+ Comparare"),
    ]
    for col, (icon, txt) in zip(cols, pasi):
        with col:
            st.markdown(f"""
            <div class="pipeline-step">
                <div style="font-size:1.8rem; font-weight:700; color:#00D4AA;">{icon}</div>
                <div style="font-weight:600; white-space:pre-line; color:#E4E8F0; margin-top:0.3rem;">{txt}</div>
            </div>
            """, unsafe_allow_html=True)

# ====================================================================
# TAB 2: ANALIZA DATELOR
# ====================================================================
with tab2:
    st.markdown("### Analiza exploratorie a datelor")

    if fisiere_ok:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distributia claselor")
            fig = go.Figure(data=[
                go.Bar(
                    x=["Legitime", "Frauduloase"],
                    y=[stats["legitime"], stats["fraude"]],
                    marker_color=["#00D4AA", "#E74C3C"],
                    text=[f"{stats['legitime']:,}", f"{stats['fraude']:,}"],
                    textposition="outside",
                )
            ])
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_type="log",
                yaxis_title="Numar tranzactii (scara log)",
                height=400,
                showlegend=False,
                margin=dict(t=20, b=20),
                font=dict(color="#E4E8F0"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Scara logaritmica datorita dezechilibrului extrem (1:578)")

        with col2:
            st.markdown("#### Suma medie a tranzactiilor")
            fig = go.Figure(data=[
                go.Bar(
                    x=["Legitime", "Frauduloase"],
                    y=[stats["suma_medie_legitima"], stats["suma_medie_frauda"]],
                    marker_color=["#00D4AA", "#E74C3C"],
                    text=[f"{stats['suma_medie_legitima']:.2f}€",
                          f"{stats['suma_medie_frauda']:.2f}€"],
                    textposition="outside",
                )
            ])
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Suma medie (EUR)",
                height=400,
                showlegend=False,
                margin=dict(t=20, b=20),
                font=dict(color="#E4E8F0"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Sumele frauduloase au un profil diferit fata de cele legitime")

        st.markdown("---")
        st.markdown("### Despre setul de date")

        col_info, col_table = st.columns([1, 1])
        with col_info:
            st.markdown("""
            **Sursa**: Kaggle - Credit Card Fraud Detection (ULB)

            **Perioada**: Septembrie 2013, 2 zile

            **Caracteristici**:
            - `Time` — secunde de la prima tranzactie
            - `V1` — `V28` — trasaturi anonimizate prin PCA
            - `Amount` — suma in EUR
            - `Class` — eticheta (0=legitim, 1=frauda)

            **De ce PCA?** Pentru a proteja confidentialitatea datelor reale ale
            clientilor. Rezultatele PCA pastreaza informatia statistica fara a
            dezvalui identitatea sau detalii sensibile.
            """)

        with col_table:
            df_info = pd.DataFrame({
                "Metrica": ["Total tranzactii", "Tranzactii legitime",
                           "Tranzactii frauduloase", "Procent fraude",
                           "Numar trasaturi", "Valori lipsa"],
                "Valoare": [f"{stats['total']:,}",
                           f"{stats['legitime']:,}",
                           f"{stats['fraude']}",
                           f"{stats['procent_fraude']:.3f}%",
                           "30 (Time, V1-V28, Amount)",
                           "Niciuna"]
            })
            st.dataframe(df_info, hide_index=True, use_container_width=True)
    else:
        st.info("Datele vor fi afisate dupa incarcarea modelului.")

# ====================================================================
# TAB 3: DEMO PREDICTIE
# ====================================================================
with tab3:
    st.markdown("### Demo predictie live")
    st.markdown("""
    Introdu valorile unei tranzactii si modelul XGBoost va prezice probabilitatea
    de a fi frauduloasa. Poti folosi exemplele predefinite sau introduce valori manual.
    """)

    if fisiere_ok:
        # Profile predefinite de tranzactii (valorile V sunt bazate pe mediile reale din dataset)
        PROFILE = {
            "Tranzactie obisnuita (cumparaturi online)": {
                "amount": 85.0, "time": 50000.0,
                "V": [0.0]*28,
            },
            "Tranzactie mare (electronice)": {
                "amount": 1200.0, "time": 72000.0,
                "V": [0.2, -0.1, 0.3, 0.1, -0.2, 0.0, 0.1, -0.1,
                      0.2, 0.1, 0.3, -0.1, 0.2, 0.1, 0.0, -0.1,
                      0.1, 0.2, -0.1, 0.0, 0.1, -0.2, 0.1, 0.0,
                      0.1, -0.1, 0.2, 0.1],
            },
            "Tranzactie mica (noaptea)": {
                "amount": 4.5, "time": 5000.0,
                "V": [0.1, 0.2, -0.1, 0.0, 0.1, -0.2, 0.1, 0.0,
                      0.2, 0.1, -0.1, 0.3, 0.0, 0.1, -0.1, 0.2,
                      0.1, 0.0, -0.1, 0.2, 0.1, 0.0, -0.2, 0.1,
                      0.0, 0.1, -0.1, 0.2],
            },
            "Profil frauda #1 (suma 0 EUR, tranzactie rapida)": {
                "amount": 0.0, "time": 406.0,
                "V": [-2.3122, 1.952, -1.6099, 3.9979, -0.5222, -1.4265, -2.5374, 1.3917,
                      -2.7701, -2.7723, 3.202, -2.8999, -0.5952, -4.2893, 0.3897, -1.1407,
                      -2.8301, -0.0168, 0.417, 0.1269, 0.5172, -0.035, -0.4652, 0.3202,
                      0.0445, 0.1778, 0.2611, -0.1433],
            },
            "Profil frauda #2 (suma 529 EUR, comportament anormal)": {
                "amount": 529.0, "time": 472.0,
                "V": [-3.0435, -3.1573, 1.0885, 2.2886, 1.3598, -1.0648, 0.3256, -0.0678,
                      -0.271, -0.8386, -0.4146, -0.5031, 0.6765, -1.692, 2.0006, 0.6668,
                      0.5997, 1.7253, 0.2833, 2.1023, 0.6617, 0.4355, 1.376, -0.2938,
                      0.2798, -0.1454, -0.2528, 0.0358],
            },
            "Profil frauda #3 (suma 239 EUR, tranzactie suspecta)": {
                "amount": 239.93, "time": 4462.0,
                "V": [-2.3033, 1.7592, -0.3597, 2.3302, -0.8216, -0.0758, 0.5623, -0.3991,
                      -0.2383, -1.5254, 2.0329, -6.5601, 0.0229, -1.4701, -0.6988, -2.2822,
                      -4.7818, -2.6157, -1.3344, -0.43, -0.2942, -0.9324, 0.1727, -0.0873,
                      -0.1561, -0.5426, 0.0396, -0.153],
            },
        }

        st.markdown("#### Selecteaza tipul tranzactiei")
        profil_ales = st.selectbox(
            "Alege un scenariu predefinit sau personalizeaza mai jos:",
            options=list(PROFILE.keys()),
            key="profil_selectat"
        )

        profil = PROFILE[profil_ales]

        st.markdown("---")
        st.markdown("#### Personalizeaza tranzactia")
        st.markdown("Poti modifica suma si ora tranzactiei. Restul parametrilor sunt setati automat conform profilului ales.")

        col_form1, col_form2 = st.columns(2)
        with col_form1:
            amount_val = st.number_input(
                "Suma tranzactiei (EUR)",
                value=float(profil["amount"]),
                min_value=0.0,
                step=1.0,
                help="Suma in euro a tranzactiei"
            )
        with col_form2:
            st.markdown("**Time (secunde de la prima tranzactie)**")
            st.markdown(f"<div class='info-box' style='margin-top:0.3rem;'>{int(profil['time'])} secunde</div>", unsafe_allow_html=True)
            time_val = float(profil["time"])

        # Info despre profilul ales
        st.markdown("""
        <div class="info-box">
        Parametrii interni (V1-V28) sunt setati automat pe baza profilului selectat.
        Acestia reprezinta trasaturi anonimizate prin PCA din datele reale de tranzactii.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        col_btn = st.columns([1, 2, 1])
        with col_btn[1]:
            predict = st.button("ANALIZEAZA TRANZACTIA",
                               use_container_width=True, type="primary")

        if predict:
            time_scaled = scaler_time.transform(np.array([[time_val]]))[0][0]
            amount_scaled = scaler_amount.transform(np.array([[amount_val]]))[0][0]
            v_values = profil["V"]
            features = [time_scaled] + v_values + [amount_scaled]
            X_input = np.array(features).reshape(1, -1)

            # Predictie
            proba = model.predict_proba(X_input)[0, 1]
            pred = int(proba > 0.5)

            if pred == 1:
                st.markdown(f"""
                <div class="result-box-fraud">
                    <div style="font-size:2rem; font-weight:700; color:#FF6B7A;">ATENTIE</div>
                    <div style="font-size:2rem; font-weight:700;">FRAUDA DETECTATA</div>
                    <div style="font-size:1.5rem; margin-top:0.5rem;">
                        Probabilitate: {proba*100:.1f}%
                    </div>
                    <div style="font-size:0.95rem; margin-top:1rem; opacity:0.9;">
                        Aceasta tranzactie are caracteristici similare cu fraudele
                        din setul de antrenare. Recomandare: blocare si verificare manuala.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-legit">
                    <div style="font-size:2rem; font-weight:700; color:#00D4AA;">OK</div>
                    <div style="font-size:2rem; font-weight:700;">TRANZACTIE LEGITIMA</div>
                    <div style="font-size:1.5rem; margin-top:0.5rem;">
                        Probabilitate frauda: {proba*100:.2f}%
                    </div>
                    <div style="font-size:0.95rem; margin-top:1rem; opacity:0.9;">
                        Tranzactia a fost evaluata ca avand un risc scazut de frauda.
                        Procesare normala recomandata.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Bara de probabilitate
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Probabilitate frauda (%)", "font": {"color": "#E4E8F0"}},
                number={"font": {"color": "#E4E8F0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8B95A7"},
                    "bar": {"color": "#E74C3C" if pred else "#00D4AA"},
                    "bgcolor": "#1A1F2E",
                    "borderwidth": 2,
                    "bordercolor": "#2A3142",
                    "steps": [
                        {"range": [0, 30], "color": "#0D2A1F"},
                        {"range": [30, 70], "color": "#2A2515"},
                        {"range": [70, 100], "color": "#2A1518"},
                    ],
                    "threshold": {
                        "line": {"color": "#E4E8F0", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(t=40, b=20),
                font=dict(color="#E4E8F0"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Demo-ul de predictie necesita modelul antrenat (model.pkl).")

# ====================================================================
# TAB 4: REZULTATE MODELE
# ====================================================================
with tab4:
    st.markdown("### Compararea modelelor antrenate")

    if fisiere_ok:
        # Tabel cu rezultate
        st.markdown("#### Tabel comparativ")
        df_rez = pd.DataFrame([
            {
                "Model": nume,
                "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}",
                "F1-Score": f"{r['f1']:.4f}",
                "ROC-AUC": f"{r['roc_auc']:.4f}",
                "PR-AUC": f"{r['pr_auc']:.4f}",
            }
            for nume, r in rezultate.items()
        ])
        st.dataframe(df_rez, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Grafic comparare metrici
        st.markdown("#### Comparare grafica a metricilor")
        metrici_nume = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
        metrici_keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
        culori_modele = {"Logistic Regression": "#4A9EFF", "Random Forest": "#00D4AA", "XGBoost": "#FF6B7A"}

        fig = go.Figure()
        for nume, r in rezultate.items():
            fig.add_trace(go.Bar(
                name=nume,
                x=metrici_nume,
                y=[r[k] for k in metrici_keys],
                marker_color=culori_modele.get(nume, "#888"),
            ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            yaxis_title="Valoare",
            yaxis=dict(range=[0, 1.05]),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="#E4E8F0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Curbe ROC si PR
        col_roc, col_pr = st.columns(2)

        with col_roc:
            st.markdown("#### Curba ROC")
            fig = go.Figure()
            for nume, r in rezultate.items():
                fig.add_trace(go.Scatter(
                    x=r["fpr"], y=r["tpr"],
                    name=f"{nume} (AUC={r['roc_auc']:.3f})",
                    mode="lines",
                    line=dict(color=culori_modele.get(nume, "#888"), width=2),
                ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], name="Aleator",
                mode="lines", line=dict(dash="dash", color="#5A6478"),
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Rata fals pozitive (FPR)",
                yaxis_title="Rata adevarat pozitive (TPR)",
                height=400,
                legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
                font=dict(color="#E4E8F0"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_pr:
            st.markdown("#### Curba Precision-Recall")
            fig = go.Figure()
            for nume, r in rezultate.items():
                fig.add_trace(go.Scatter(
                    x=r["recall_curve"], y=r["precision_curve"],
                    name=f"{nume} (AP={r['pr_auc']:.3f})",
                    mode="lines",
                    line=dict(color=culori_modele.get(nume, "#888"), width=2),
                ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=400,
                legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
                font=dict(color="#E4E8F0"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Matrici de confuzie
        st.markdown("#### Matrici de confuzie")
        cols = st.columns(3)
        for col, (nume, r) in zip(cols, rezultate.items()):
            with col:
                cm = np.array(r["confusion_matrix"])
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=["Legitim", "Frauda"], y=["Legitim", "Frauda"],
                    text=cm, texttemplate="%{text}",
                    colorscale=[[0, "#1A1F2E"], [0.5, "#2E5A4F"], [1, "#00D4AA"]],
                    showscale=False,
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title=nume,
                    xaxis_title="Predictie", yaxis_title="Realitate",
                    height=300, margin=dict(t=40, b=20),
                    font=dict(color="#E4E8F0"),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <b>Concluzie:</b> XGBoost obtine cele mai bune rezultate pe toate metricile relevante,
        confirmand reputatia algoritmului pe date tabulare. Random Forest este o alternativa
        solida cu antrenare mai rapida. Logistic Regression are recall mare dar precision
        redusa, generand multe alarme false.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Rezultatele vor fi afisate dupa incarcarea modelelor.")

# ====================================================================
# TAB 5: DESPRE PROIECT
# ====================================================================
with tab5:
    st.markdown("### Despre proiect")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        #### Tema
        **Supravegherea riscurilor financiare prin inteligenta artificiala:
        frauda, spalare de bani si gestionarea portofoliilor.**

        #### Cele 3 directii ale temei

        **1. Detectia fraudei** *(directia implementata in acest proiect)*

        Identificarea automata a tranzactiilor frauduloase folosind tehnici de
        Machine Learning. Caracterizata de dezechilibru extrem de clase si necesitatea
        deciziilor in timp real.

        **2. Anti-spalarea banilor (AML)**

        Detectarea schemelor prin care fonduri ilicite sunt reintroduse in economia
        legala. Tehnici moderne includ Graph Neural Networks pentru analiza retelelor
        de tranzactii suspecte.

        **3. Gestionarea portofoliilor**

        Optimizarea alocarii capitalului intre active. AI completeaza modelul clasic
        Markowitz prin predictia volatilitatii (LSTM), Reinforcement Learning pentru
        strategii adaptive si analiza de sentiment a stirilor financiare (BERT).

        #### Metodologia (rezumat)
        1. **Date**: Credit Card Fraud Detection (Kaggle, ULB)
        2. **Preprocesare**: standardizare Time + Amount; impartire 80/20 cu stratificare
        3. **Echilibrare**: SMOTE doar pe setul de antrenare
        4. **Modele**: Logistic Regression, Random Forest, XGBoost
        5. **Evaluare**: Precision, Recall, F1, ROC-AUC, PR-AUC

        #### Tehnologii
        - **Python 3** — limbajul de baza
        - **scikit-learn** — modele clasice si preprocesare
        - **XGBoost** — gradient boosting
        - **imbalanced-learn** — SMOTE
        - **Streamlit** — interfata web interactiva
        - **Plotly** — grafice interactive
        """)

    with col2:
        st.markdown("#### Resurse")
        st.markdown("""
        - [Dataset Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
        - [SMOTE paper](https://arxiv.org/abs/1106.1813)
        - [XGBoost paper](https://arxiv.org/abs/1603.02754)
        - [Streamlit](https://streamlit.io)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#5A6478; font-size:0.85rem; padding:1rem;">
Practica de cercetare - Detectia fraudei prin AI | Construit cu Streamlit | 2026
</div>
""", unsafe_allow_html=True)
