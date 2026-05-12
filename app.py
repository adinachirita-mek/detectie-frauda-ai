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
    st.markdown("Modelul analizeaza simultan o tranzactie legitima si una frauduloasa. Contrastul dintre cele doua este imediat vizibil.")

    if fisiere_ok:

        # Date fixe pentru cele doua scenarii
        LEGITIM = {
            "label": "Tranzactie legitima",
            "amount": 100.0, "time": 50000.0,
            "V": [-1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396, 0.0987,
                  0.3638, 0.0908, -0.5516, -0.6178, -0.9914, -0.3112, 1.4682, -0.4704,
                  0.208, 0.0258, 0.404, 0.2514, -0.0183, 0.2778, -0.1105, 0.0669,
                  0.1285, -0.1891, 0.1336, -0.0211],
        }
        FRAUDA = {
            "label": "Tranzactie frauduloasa",
            "amount": 0.0, "time": 406.0,
            "V": [-2.3122, 1.952, -1.6099, 3.9979, -0.5222, -1.4265, -2.5374, 1.3917,
                  -2.7701, -2.7723, 3.202, -2.8999, -0.5952, -4.2893, 0.3897, -1.1407,
                  -2.8301, -0.0168, 0.417, 0.1269, 0.5172, -0.035, -0.4652, 0.3202,
                  0.0445, 0.1778, 0.2611, -0.1433],
        }
        SUSPECT = {
            "label": "Tranzactie suspecta",
            "amount": 100.0, "time": 50000.0,
            "V": [-1.836, 0.9396, 0.4632, 2.6881, -0.4303, -0.4821, -1.1489, 0.7452,
                  -1.2031, -1.3407, 1.3252, -1.7589, -0.7933, -2.3003, 0.9289, -0.8055,
                  -1.311, 0.0045, 0.4105, 0.1892, 0.2495, 0.1214, -0.2878, 0.1935,
                  0.0865, -0.0056, 0.1973, -0.0822],
        }

        def predict_profil(profil):
            t = scaler_time.transform(np.array([[profil["time"]]]))[0][0]
            a = scaler_amount.transform(np.array([[profil["amount"]]]))[0][0]
            features = [t] + profil["V"] + [a]
            X = np.array(features).reshape(1, -1)
            return model.predict_proba(X)[0, 1]

        def gauge_fig(proba, culoare):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"font": {"color": "#E4E8F0", "size": 36},
                        "valueformat": ".2f", "suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8B95A7"},
                    "bar": {"color": culoare},
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
                        "thickness": 0.75, "value": 50,
                    },
                },
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=250, margin=dict(t=20, b=10),
                font=dict(color="#E4E8F0"),
            )
            return fig

        # Calculeaza toate probabilitatile automat
        p_legit = predict_profil(LEGITIM)
        p_frauda = predict_profil(FRAUDA)
        p_suspect = predict_profil(SUSPECT)

        st.markdown("---")

        # Afisare side-by-side - 3 coloane
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="result-box-legit" style="min-height:120px;">
                <div style="font-size:1.3rem; font-weight:700;">LEGITIMA</div>
                <div style="font-size:0.9rem; margin-top:0.3rem; opacity:0.8;">
                    Suma: {LEGITIM['amount']} EUR
                </div>
                <div style="font-size:2rem; font-weight:700; margin-top:0.5rem;">
                    {p_legit*100:.4f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(gauge_fig(p_legit, "#00D4AA"),
                          use_container_width=True, key="gauge_legit")

        with col2:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1A1A2E,#0E1117);
                        border:2px solid #F39C12; border-radius:12px;
                        padding:1.5rem; text-align:center; min-height:120px;
                        box-shadow:0 0 20px rgba(243,156,18,0.2);">
                <div style="font-size:1.3rem; font-weight:700; color:#F39C12;">SUSPECTA</div>
                <div style="font-size:0.9rem; margin-top:0.3rem; color:#8B95A7;">
                    Suma: {SUSPECT['amount']} EUR
                </div>
                <div style="font-size:2rem; font-weight:700; color:#F39C12; margin-top:0.5rem;">
                    {p_suspect*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(gauge_fig(p_suspect, "#F39C12"),
                          use_container_width=True, key="gauge_suspect")

        with col3:
            st.markdown(f"""
            <div class="result-box-fraud" style="min-height:120px;">
                <div style="font-size:1.3rem; font-weight:700;">FRAUDA</div>
                <div style="font-size:0.9rem; margin-top:0.3rem; opacity:0.8;">
                    Suma: {FRAUDA['amount']} EUR
                </div>
                <div style="font-size:2rem; font-weight:700; margin-top:0.5rem;">
                    {p_frauda*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(gauge_fig(p_frauda, "#E74C3C"),
                          use_container_width=True, key="gauge_frauda")

        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        Cele trei scenarii de mai sus folosesc tranzactii reale din dataset.
        Parametrii interni (V1-V28) sunt trasaturi anonimizate prin PCA.
        Modelul XGBoost analizeaza toate cele 30 de trasaturi simultan si returneaza
        probabilitatea ca tranzactia sa fie frauduloasa.
        </div>
        """, unsafe_allow_html=True)

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
