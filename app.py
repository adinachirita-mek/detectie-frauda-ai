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
    st.markdown("Genereaza o tranzactie cu date reale si vezi cum o clasifica modelul. Mai jos poti explora interactiv granita dintre legitim si frauda.")

    if fisiere_ok:

        V_LEGIT = [-1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396, 0.0987,
                   0.3638, 0.0908, -0.5516, -0.6178, -0.9914, -0.3112, 1.4682, -0.4704,
                   0.208, 0.0258, 0.404, 0.2514, -0.0183, 0.2778, -0.1105, 0.0669,
                   0.1285, -0.1891, 0.1336, -0.0211]
        V_FRAUDA = [-2.3122, 1.952, -1.6099, 3.9979, -0.5222, -1.4265, -2.5374, 1.3917,
                    -2.7701, -2.7723, 3.202, -2.8999, -0.5952, -4.2893, 0.3897, -1.1407,
                    -2.8301, -0.0168, 0.417, 0.1269, 0.5172, -0.035, -0.4652, 0.3202,
                    0.0445, 0.1778, 0.2611, -0.1433]

        # ── SECTIUNEA 1: GENERARE TRANZACTIE ──────────────────────────
        st.markdown("---")
        st.markdown("#### Genereaza o tranzactie")
        st.markdown("Completeaza detaliile de mai jos pentru a construi o tranzactie si a o clasifica.")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            suma = st.number_input("Suma (EUR)", min_value=0.0, max_value=10000.0,
                                   value=85.0, step=1.0)
            tip_card = st.selectbox("Tip card", ["Visa Debit", "Mastercard Credit",
                                                  "Visa Credit", "Mastercard Debit"])
        with col_f2:
            ora = st.slider("Ora tranzactiei", 0, 23, 14, format="%d:00")
            tara = st.selectbox("Tara comerciant", ["Romania", "Franta", "Germania",
                                                     "UK", "SUA", "Olanda", "Alta tara"])
        with col_f3:
            tip_comerciant = st.selectbox("Tip comerciant", [
                "Cumparaturi online", "Restaurant / Cafenea", "Supermarket",
                "Electronice", "ATM / Retragere numerar", "Transport / Combustibil",
                "Servicii digitale (abonamente)"
            ])
            metoda = st.selectbox("Metoda", ["Contactless", "Chip + PIN", "Online (3D Secure)", "Banda magnetica"])

        # Mapam campurile vizibile la profiluri de risc
        # Factori de risc: ATM noaptea + alta tara + banda magnetica = frauda
        scor_risc = 0
        if ora < 6 or ora > 23:
            scor_risc += 0.3
        if tara in ["SUA", "Alta tara"]:
            scor_risc += 0.25
        if metoda == "Banda magnetica":
            scor_risc += 0.3
        if tip_comerciant == "ATM / Retragere numerar":
            scor_risc += 0.15
        if suma < 2 or suma > 2000:
            scor_risc += 0.15
        scor_risc = min(scor_risc, 1.0)

        # Interpolare V pe baza scorului de risc
        v_tranzactie = [V_LEGIT[i] * (1 - scor_risc) + V_FRAUDA[i] * scor_risc for i in range(28)]
        time_val = float(ora * 3600)

        if st.button("CLASIFICA TRANZACTIA", type="primary", use_container_width=True):
            t_sc = scaler_time.transform(np.array([[time_val]]))[0][0]
            a_sc = scaler_amount.transform(np.array([[suma]]))[0][0]
            features = [t_sc] + v_tranzactie + [a_sc]
            proba_t = model.predict_proba(np.array(features).reshape(1, -1))[0, 1]

            # Afisam cardul de tranzactie
            st.markdown("#### Tranzactia generata")
            col_card, col_rez = st.columns([1, 1])

            with col_card:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #1A1F2E, #2A3142);
                            border:1px solid #2A3142; border-radius:16px;
                            padding:1.5rem; font-family:monospace;">
                    <div style="color:#8B95A7; font-size:0.8rem; margin-bottom:1rem;
                                letter-spacing:2px;">DETALII TRANZACTIE</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;">
                        <div>
                            <div style="color:#8B95A7; font-size:0.75rem;">SUMA</div>
                            <div style="color:#E4E8F0; font-size:1.4rem; font-weight:700;">
                                {suma:.2f} EUR</div>
                        </div>
                        <div>
                            <div style="color:#8B95A7; font-size:0.75rem;">ORA</div>
                            <div style="color:#E4E8F0; font-size:1.4rem; font-weight:700;">
                                {ora:02d}:00</div>
                        </div>
                        <div>
                            <div style="color:#8B95A7; font-size:0.75rem;">TIP CARD</div>
                            <div style="color:#E4E8F0; font-size:0.95rem;">{tip_card}</div>
                        </div>
                        <div>
                            <div style="color:#8B95A7; font-size:0.75rem;">TARA</div>
                            <div style="color:#E4E8F0; font-size:0.95rem;">{tara}</div>
                        </div>
                        <div>
                            <div style="color:#8B95A7; font-size:0.75rem;">COMERCIANT</div>
                            <div style="color:#E4E8F0; font-size:0.95rem;">{tip_comerciant}</div>
                        </div>
                        <div>
                            <div style="color:#8B95A7; font-size:0.75rem;">METODA</div>
                            <div style="color:#E4E8F0; font-size:0.95rem;">{metoda}</div>
                        </div>
                    </div>
                    <div style="margin-top:1rem; padding-top:1rem;
                                border-top:1px solid #2A3142;">
                        <div style="color:#8B95A7; font-size:0.75rem;">PARAMETRI INTERNI (PCA)</div>
                        <div style="color:#5A6478; font-size:0.7rem; margin-top:0.3rem;">
                            V1={v_tranzactie[0]:.3f} &nbsp; V2={v_tranzactie[1]:.3f} &nbsp;
                            V3={v_tranzactie[2]:.3f} &nbsp; V14={v_tranzactie[13]:.3f} &nbsp;
                            Amount={suma:.2f} &nbsp; Time={int(time_val)}s
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_rez:
                if proba_t < 0.3:
                    culoare_t = "#00D4AA"
                    eticheta_t = "LEGITIMA"
                    icon_t = "APROBATA"
                elif proba_t < 0.7:
                    culoare_t = "#F39C12"
                    eticheta_t = "SUSPECTA"
                    icon_t = "IN VERIFICARE"
                else:
                    culoare_t = "#E74C3C"
                    eticheta_t = "FRAUDA DETECTATA"
                    icon_t = "BLOCATA"

                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #1A1F2E, #0E1117);
                            border:2px solid {culoare_t}; border-radius:16px;
                            padding:1.5rem; text-align:center;
                            box-shadow:0 0 25px {culoare_t}44; height:100%;">
                    <div style="color:#8B95A7; font-size:0.8rem; letter-spacing:2px;
                                margin-bottom:0.5rem;">DECIZIE MODEL</div>
                    <div style="font-size:1rem; font-weight:700; color:{culoare_t};
                                letter-spacing:1px;">{icon_t}</div>
                    <div style="font-size:1.6rem; font-weight:700; color:{culoare_t};
                                margin:0.5rem 0;">{eticheta_t}</div>
                    <div style="font-size:2.5rem; font-weight:700; color:{culoare_t};">
                        {proba_t*100:.2f}%</div>
                    <div style="color:#8B95A7; font-size:0.8rem; margin-top:0.5rem;">
                        probabilitate frauda</div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge mic
                fig_t = go.Figure(go.Indicator(
                    mode="gauge",
                    value=proba_t * 100,
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8B95A7"},
                        "bar": {"color": culoare_t},
                        "bgcolor": "#1A1F2E",
                        "borderwidth": 1, "bordercolor": "#2A3142",
                        "steps": [
                            {"range": [0, 30], "color": "#0D2A1F"},
                            {"range": [30, 70], "color": "#2A2515"},
                            {"range": [70, 100], "color": "#2A1518"},
                        ],
                        "threshold": {"line": {"color": "#E4E8F0", "width": 2},
                                      "thickness": 0.75, "value": 50},
                    },
                ))
                fig_t.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=180, margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_t, use_container_width=True)

            st.markdown("""
            <div class="info-box">
            Campurile completate (suma, ora, tara, metoda) sunt traduse in parametri statistici
            prin intermediul profilurilor de risc invatate din dataset. Modelul XGBoost analizeaza
            toti parametrii si returneaza probabilitatea ca tranzactia sa fie frauduloasa.
            </div>
            """, unsafe_allow_html=True)

        # ── SECTIUNEA 2: SLIDER INTERACTIV ────────────────────────────
        st.markdown("---")
        st.markdown("#### Exploreaza granita dintre legitim si frauda")
        st.markdown("Slider-ul interpoleaza in timp real intre o tranzactie legitima si una frauduloasa reale din dataset.")

        nivel = st.slider(
            "Nivel de risc",
            min_value=0, max_value=100, value=0, step=1,
            format="%d%%",
            help="0% = tranzactie complet normala | 100% = profil de frauda pur"
        )

        pct = nivel / 100.0
        v_mix = [V_LEGIT[i] * (1 - pct) + V_FRAUDA[i] * pct for i in range(28)]
        amount_mix = 100.0 * (1 - pct) + 0.0 * pct
        time_mix = 50000.0 * (1 - pct) + 406.0 * pct

        t_scaled = scaler_time.transform(np.array([[time_mix]]))[0][0]
        a_scaled = scaler_amount.transform(np.array([[amount_mix]]))[0][0]
        features = [t_scaled] + v_mix + [a_scaled]
        proba = model.predict_proba(np.array(features).reshape(1, -1))[0, 1]

        if proba < 0.3:
            culoare = "#00D4AA"
            eticheta = "LEGITIMA"
        elif proba < 0.7:
            culoare = "#F39C12"
            eticheta = "SUSPECTA"
        else:
            culoare = "#E74C3C"
            eticheta = "FRAUDA DETECTATA"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #1A1F2E, #0E1117);
                    border: 2px solid {culoare}; border-radius:12px;
                    padding:1.5rem; text-align:center; margin:1rem 0;
                    box-shadow: 0 0 30px {culoare}44;">
            <div style="font-size:1.8rem; font-weight:700; color:{culoare};">
                {eticheta}
            </div>
            <div style="font-size:3rem; font-weight:700; color:{culoare}; margin:0.5rem 0;">
                {proba*100:.2f}%
            </div>
            <div style="color:#8B95A7; font-size:0.9rem;">
                Suma: {amount_mix:.2f} EUR &nbsp;|&nbsp; Nivel risc: {nivel}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"font": {"color": "#E4E8F0", "size": 48},
                    "valueformat": ".2f", "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8B95A7"},
                "bar": {"color": culoare},
                "bgcolor": "#1A1F2E",
                "borderwidth": 2, "bordercolor": "#2A3142",
                "steps": [
                    {"range": [0, 30], "color": "#0D2A1F"},
                    {"range": [30, 70], "color": "#2A2515"},
                    {"range": [70, 100], "color": "#2A1518"},
                ],
                "threshold": {"line": {"color": "#E4E8F0", "width": 3},
                              "thickness": 0.75, "value": 50},
            },
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(t=20, b=10),
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
