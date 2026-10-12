"""Streamlit-App: Churn-Frühwarnsystem für das Retention-Team.

Start lokal:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# src/ in den Suchpfad aufnehmen, damit die App ohne Paketinstallation läuft (z. B. Streamlit Cloud)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import CATEGORY_VALUES, OFFER_COST, SCORES_PATH, SUCCESS_RATE  # noqa: E402
from churn.data import clean_raw_data  # noqa: E402
from churn.explain import contributions, feature_label  # noqa: E402
from churn.features import build_features  # noqa: E402
from churn.score import ChurnModel, load_model, run_batch_scoring, score_customers  # noqa: E402

# Farben aus der validierten Referenzpalette (Status = reserviert, Divergenz = blau/rot)
RISK_COLORS = {"Hoch": "#d03b3b", "Beobachten": "#fab219", "Niedrig": "#0ca30c"}
RISK_ICONS = {"Hoch": "🔴", "Beobachten": "🟠", "Niedrig": "🟢"}
RISK_ORDER = ["Hoch", "Beobachten", "Niedrig"]
COLOR_INCREASE, COLOR_DECREASE, COLOR_SEQ = "#e34948", "#2a78d6", "#2a78d6"

st.set_page_config(page_title="Churn-Frühwarnsystem", page_icon="📉", layout="wide")


@st.cache_resource
def get_model() -> ChurnModel:
    """Modell einmal pro App-Prozess laden."""
    return load_model()


@st.cache_data
def get_scores() -> pd.DataFrame:
    """Score-Liste laden; fehlt sie (z. B. frischer Checkout), wird sie erzeugt."""
    if not SCORES_PATH.exists():
        return run_batch_scoring()
    return pd.read_csv(SCORES_PATH)


def euro(value: float, markdown: bool = False) -> str:
    """Betrag im deutschen Format (Datensatz-Währung ist $).

    In Markdown-Texten muss $ maskiert werden, sonst rendert Streamlit LaTeX.
    """
    text = f"{value:,.0f} $".replace(",", ".")
    return text.replace("$", "\\$") if markdown else text


model = get_model()
scores = get_scores()
meta = model.metadata

# ---------------------------------------------------------------- Kopfbereich
st.title("📉 Churn-Frühwarnsystem")
st.caption(
    f"Telco-Kunden · Logistische Regression · Schwellenwert {model.threshold:.2f} · "
    f"trainiert am {meta['trained_at'][:10]}"
)

tab_overview, tab_customer, tab_model = st.tabs(
    ["Risiko-Übersicht", "Einzelkunde prüfen", "Modell & Annahmen"]
)

# ---------------------------------------------------------------- Tab 1: Übersicht
with tab_overview:
    high = scores[scores["risk_level"] == "Hoch"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aktive Kunden", f"{len(scores):,}".replace(",", "."))
    c2.metric(f"Hochrisiko-Kunden ({len(high) / len(scores):.0%})", f"{len(high):,}".replace(",", "."))
    c3.metric("Gefährdeter Jahresumsatz", euro(high["annual_revenue_at_risk"].sum()))
    c4.metric("Kampagnenbudget (alle Hoch)", euro(OFFER_COST * len(high)))

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Kunden je Risikostufe")
        by_level = (scores.groupby("risk_level")
                    .agg(kunden=("churn_score", "size"),
                         jahresumsatz=("annual_revenue_at_risk", "sum"))
                    .reindex(RISK_ORDER).reset_index())
        by_level["label"] = by_level["risk_level"].map(lambda r: f"{RISK_ICONS[r]} {r}")
        bars = alt.Chart(by_level).mark_bar(cornerRadiusEnd=4, height=28).encode(
            x=alt.X("kunden:Q", title="Anzahl Kunden"),
            y=alt.Y("label:N", sort=[f"{RISK_ICONS[r]} {r}" for r in RISK_ORDER], title=None),
            color=alt.Color("risk_level:N", legend=None,
                            scale=alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values()))),
            tooltip=[alt.Tooltip("risk_level:N", title="Stufe"),
                     alt.Tooltip("kunden:Q", title="Kunden", format=","),
                     alt.Tooltip("jahresumsatz:Q", title="Jahresumsatz ($)", format=",.0f")],
        )
        labels = bars.mark_text(align="left", dx=4).encode(text=alt.Text("kunden:Q", format=","),
                                                         color=alt.value("#52514e"))
        st.altair_chart((bars + labels).properties(height=160), use_container_width=True)

    with right:
        st.subheader("Hauptgründe bei Hochrisiko-Kunden")
        reasons = high["reason_1"].value_counts().rename_axis("grund").reset_index(name="kunden")
        reason_chart = alt.Chart(reasons).mark_bar(cornerRadiusEnd=4, color=COLOR_SEQ).encode(
            x=alt.X("kunden:Q", title="Anzahl Kunden (Grund Nr. 1)"),
            y=alt.Y("grund:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=[alt.Tooltip("grund:N", title="Grund"), alt.Tooltip("kunden:Q", title="Kunden")],
        )
        st.altair_chart(reason_chart.properties(height=34 * len(reasons) + 20), use_container_width=True)

    st.subheader("Kampagnenliste")
    f1, f2, f3 = st.columns([1, 1, 1])
    levels = f1.multiselect("Risikostufe", RISK_ORDER, default=["Hoch"])
    contracts = f2.multiselect("Vertragsart", CATEGORY_VALUES["contract"],
                               default=CATEGORY_VALUES["contract"])
    capacity = f3.number_input("Kapazität des Teams (Kunden)", min_value=10, max_value=len(scores),
                               value=min(300, len(scores)), step=50,
                               help="Wie viele Kunden kann das Retention-Team in dieser Woche ansprechen?")

    filtered = (scores[scores["risk_level"].isin(levels) & scores["contract"].isin(contracts)]
                .sort_values("churn_score", ascending=False)
                .head(int(capacity)))
    st.dataframe(
        filtered[["priority_rank", "customer_id", "churn_score", "risk_level", "reason_1", "reason_2",
                  "recommended_action", "contract", "tenure", "monthly_charges", "annual_revenue_at_risk"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "priority_rank": st.column_config.NumberColumn("Rang"),
            "customer_id": "Kunden-ID",
            "churn_score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "risk_level": "Stufe",
            "reason_1": "Grund 1",
            "reason_2": "Grund 2",
            "recommended_action": "Empfohlene Maßnahme",
            "contract": "Vertrag",
            "tenure": st.column_config.NumberColumn("Monate", format="%d"),
            "monthly_charges": st.column_config.NumberColumn("Monatsbeitrag", format="%.2f $"),
            "annual_revenue_at_risk": st.column_config.NumberColumn("Jahresumsatz", format="%.0f $"),
        },
    )
    expected_savings = (SUCCESS_RATE * filtered["annual_revenue_at_risk"] * filtered["churn_score"]).sum() \
        - OFFER_COST * len(filtered)
    st.caption(
        f"{len(filtered)} Kunden ausgewählt · Kosten {euro(OFFER_COST * len(filtered), markdown=True)} · "
        f"grobe erwartete Ersparnis {euro(expected_savings, markdown=True)} (Score als Wahrscheinlichkeit gelesen – "
        "das Modell überschätzt diese eher, siehe Tab „Modell & Annahmen“)."
    )
    st.download_button("Liste als CSV herunterladen", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="kampagnenliste.csv", mime="text/csv")

# ---------------------------------------------------------------- Tab 2: Einzelkunde
with tab_customer:
    st.subheader("Was-wäre-wenn: Risiko eines Kunden berechnen")
    with st.form("customer_form"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Vertrag**")
            tenure = st.slider("Kundendauer (Monate)", 0, 72, 3)
            contract = st.selectbox("Vertragsart", CATEGORY_VALUES["contract"])
            payment = st.selectbox("Zahlungsart", CATEGORY_VALUES["payment_method"])
            paperless = st.checkbox("Papierlose Rechnung", value=True)
        with b:
            st.markdown("**Dienste**")
            internet = st.selectbox("Internet", CATEGORY_VALUES["internet_service"], index=1)
            phone = st.checkbox("Telefonvertrag", value=True)
            multiple = st.checkbox("Mehrere Leitungen", value=False)
            addons = st.multiselect(
                "Zusatzdienste",
                ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                 "StreamingTV", "StreamingMovies"],
            )
        with c:
            st.markdown("**Kunde & Kosten**")
            monthly = st.number_input("Monatsbeitrag ($)", 18.0, 120.0, 85.0, step=1.0)
            senior = st.checkbox("Seniorenkunde", value=False)
            partner = st.checkbox("Hat Partner", value=False)
            dependents = st.checkbox("Hat Angehörige", value=False)
        submitted = st.form_submit_button("Risiko berechnen", type="primary")

    if submitted:
        yes_no = {True: "Yes", False: "No"}
        no_internet = internet == "No"
        addon_value = {s: ("No internet service" if no_internet else yes_no[s in addons])
                       for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                                 "StreamingTV", "StreamingMovies"]}
        raw = pd.DataFrame([{
            "customerID": "WHAT-IF", "gender": "Female", "SeniorCitizen": int(senior),
            "Partner": yes_no[partner], "Dependents": yes_no[dependents], "tenure": tenure,
            "PhoneService": yes_no[phone],
            "MultipleLines": yes_no[multiple] if phone else "No phone service",
            "InternetService": internet, **addon_value, "Contract": contract,
            "PaperlessBilling": yes_no[paperless], "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": str(round(monthly * tenure, 2)),
        }])
        customer = clean_raw_data(raw)
        result = score_customers(customer, model).iloc[0]

        r1, r2 = st.columns([1, 2])
        with r1:
            level = result["risk_level"]
            st.metric("Churn-Score", f"{result['churn_score']:.2f}")
            st.markdown(f"### {RISK_ICONS[level]} Risikostufe: {level}")
            st.markdown(f"**Empfohlene Maßnahme:** {result['recommended_action']}")
            st.caption("Der Score ist eine Rangfolge, keine exakte Wahrscheinlichkeit.")
        with r2:
            contrib = contributions(model.pipeline, build_features(customer), model.reference).iloc[0]
            contrib_df = (contrib.rename("beitrag").reset_index().rename(columns={"index": "merkmal"}))
            contrib_df["merkmal"] = contrib_df["merkmal"].map(feature_label)
            contrib_df["richtung"] = contrib_df["beitrag"].map(
                lambda v: "erhöht Risiko" if v > 0 else "senkt Risiko")
            contrib_df = contrib_df.reindex(contrib_df["beitrag"].abs().sort_values(ascending=False).index).head(8)
            chart = alt.Chart(contrib_df).mark_bar(cornerRadius=3).encode(
                x=alt.X("beitrag:Q", title="Beitrag zum Risiko (Log-Odds)"),
                y=alt.Y("merkmal:N", sort=alt.EncodingSortField("beitrag", op="sum", order="descending"),
                        title=None, axis=alt.Axis(labelLimit=260)),
                color=alt.Color("richtung:N", title=None,
                                scale=alt.Scale(domain=["erhöht Risiko", "senkt Risiko"],
                                                range=[COLOR_INCREASE, COLOR_DECREASE]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("merkmal:N", title="Merkmal"),
                         alt.Tooltip("beitrag:Q", title="Beitrag", format="+.2f")],
            )
            rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c3c2b7").encode(x="x:Q")
            st.markdown("**Warum dieser Score?** Wichtigste Einflussfaktoren")
            st.altair_chart((chart + rule).properties(height=34 * len(contrib_df) + 20), use_container_width=True)

# ---------------------------------------------------------------- Tab 3: Modell
with tab_model:
    m = meta["test_metrics"]
    st.subheader("Modellgüte auf dem Test-Split")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("PR-AUC", f"{m['pr_auc']:.3f}", help="Zufall ≈ Churn-Rate (0,27)")
    k2.metric("Recall", f"{m['recall']:.0%}", help="Anteil der echten Kündiger, die gefunden werden")
    k3.metric("Precision", f"{m['precision']:.0%}", help="Anteil echter Kündiger unter den Angesprochenen")
    k4.metric("Lift Top-10 %", f"{m['lift_top10']:.1f}×", help="Mehr Kündiger als bei Zufallsauswahl")

    s1, s2 = st.columns(2)
    s1.metric("Netto-Ersparnis mit Modell (Test)", euro(m["net_savings"]))
    s2.metric("Alle Kunden ansprechen (Test)", euro(m["net_savings_all_customers"]))

    st.subheader("Annahmen & Grenzen")
    assumptions = meta["business_assumptions"]
    st.markdown(
        f"""
- **Kampagne:** {assumptions['offer_cost']:.0f} \\$ pro Angebot, {assumptions['success_rate']:.0%} der
  echten Kündiger bleiben, Wert eines geretteten Kunden = {assumptions['revenue_months']} Monatsbeiträge.
- **Schwellenwert {model.threshold:.2f}:** maximiert die Netto-Ersparnis auf Out-of-Fold-Vorhersagen (nur Train).
- **Kalibrierung:** Wegen `class_weight="balanced"` liegen die Scores über den echten Wahrscheinlichkeiten.
- **Snapshot-Daten:** Es gibt keinen Zeitbezug; die aktiven Kunden waren Teil des Trainings (In-Sample-Scoring).
  In der Praxis trainiert man auf einem früheren Zeitraum und bewertet den nächsten.
- **Fairness:** `gender` wird nicht verwendet.
"""
    )
    st.caption(f"scikit-learn {meta['sklearn_version']} · {meta['n_train_rows']} Trainingszeilen")
