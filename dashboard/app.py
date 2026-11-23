"""Streamlit dashboard - six sections for analysts and business users.

Run:  ``streamlit run dashboard/app.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.api.schemas import MONTHS  # noqa: E402
from src.config.settings import get_settings  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.models.predictor import Predictor  # noqa: E402
from src.utils.io import load_json  # noqa: E402

st.set_page_config(page_title="Purchase Intent XAI", page_icon="🛒", layout="wide")
S = get_settings()
REPORTS = S.path("reports_dir")
BLUE, GREEN, RED = "#2b6cb0", "#38a169", "#e53e3e"


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_dataset(S)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> Predictor | None:
    try:
        return Predictor()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def get_report(name: str):
    path = REPORTS / name
    if not path.exists():
        return None
    return pd.read_csv(path) if name.endswith(".csv") else load_json(path)


def need_model(pred: Predictor | None) -> bool:
    if pred is None:
        st.error("Model artifacts not found. Run `python -m src.models.train` first.")
        return False
    return True


# ------------------------------------------------------------------ 1. overview
def page_overview(df: pd.DataFrame, pred: Predictor | None) -> None:
    st.title("🛒 Purchase Intent Prediction with Explainable AI")
    st.write(
        "Predict which browsing sessions will end in a purchase - and **why** - so the business "
        "can act on undecided visitors at the right moment."
    )
    c = st.columns(4)
    c[0].metric("Sessions analysed", f"{len(df):,}")
    c[1].metric("Conversion rate", f"{df[S.target].mean():.1%}")
    c[2].metric("Returning visitors", f"{(df.VisitorType == 'Returning_Visitor').mean():.0%}")
    c[3].metric("Weekend sessions", f"{df.Weekend.mean():.0%}")
    if not need_model(pred):
        return
    m = pred.metadata
    t = m["test_metrics_at_threshold"]
    st.subheader("Model at a glance")
    c = st.columns(4)
    c[0].metric("Model", m["model_name"])
    c[1].metric("PR-AUC (test)", f"{t['pr_auc']:.3f}")
    c[2].metric("Recall (test)", f"{t['recall']:.1%}")
    c[3].metric("Precision (test)", f"{t['precision']:.1%}")
    st.info(
        f"Only **{df[S.target].mean():.1%}** of sessions convert, so a model that always says 'no purchase' "
        f"would be **{1 - df[S.target].mean():.1%} accurate** but useless. That is why we judge the model on "
        "PR-AUC and recall, and choose the decision threshold by expected business value."
    )
    st.subheader("Top drivers of purchase intent")
    imp = pd.DataFrame(m["top_features"]).iloc[::-1]
    fig = px.bar(
        imp, x="mean_abs_shap", y="feature", orientation="h", color_discrete_sequence=[BLUE]
    )
    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Mean |SHAP|", yaxis_title=""
    )
    st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------ 2. live prediction
def page_live(df: pd.DataFrame, pred: Predictor | None) -> None:
    st.header("Live session scoring")
    if not need_model(pred):
        return
    med = df.median(numeric_only=True)
    with st.form("session"):
        st.caption("Describe a browsing session, then press **Predict**.")
        a, b, c = st.columns(3)
        admin = a.number_input("Administrative pages", 0, 100, int(med["Administrative"]))
        admin_d = a.number_input(
            "Administrative seconds", 0.0, 5000.0, float(med["Administrative_Duration"])
        )
        info = a.number_input("Informational pages", 0, 50, int(med["Informational"]))
        info_d = a.number_input(
            "Informational seconds", 0.0, 5000.0, float(med["Informational_Duration"])
        )
        prod = b.number_input("Product pages", 0, 1000, int(med["ProductRelated"]))
        prod_d = b.number_input(
            "Product seconds", 0.0, 50000.0, float(med["ProductRelated_Duration"])
        )
        bounce = b.slider("Bounce rate", 0.0, 1.0, float(round(med["BounceRates"], 3)), 0.001)
        exit_ = b.slider("Exit rate", 0.0, 1.0, float(round(med["ExitRates"], 3)), 0.001)
        pv = c.number_input("Page value", 0.0, 500.0, float(round(med["PageValues"], 2)))
        special = c.select_slider("Special-day closeness", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0)
        month = c.selectbox("Month", MONTHS, index=MONTHS.index("Nov"))
        visitor = c.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        d1, d2, d3, d4, d5 = st.columns(5)
        osys = d1.number_input("OS code", 1, 20, int(med["OperatingSystems"]))
        browser = d2.number_input("Browser code", 1, 30, int(med["Browser"]))
        region = d3.number_input("Region code", 1, 20, int(med["Region"]))
        traffic = d4.number_input("Traffic type code", 1, 30, int(med["TrafficType"]))
        weekend = d5.checkbox("Weekend")
        submitted = st.form_submit_button("Predict", type="primary")
    if not submitted:
        return
    session = dict(
        Administrative=admin,
        Administrative_Duration=admin_d,
        Informational=info,
        Informational_Duration=info_d,
        ProductRelated=prod,
        ProductRelated_Duration=prod_d,
        BounceRates=bounce,
        ExitRates=exit_,
        PageValues=pv,
        SpecialDay=special,
        Month=month,
        OperatingSystems=osys,
        Browser=browser,
        Region=region,
        TrafficType=traffic,
        VisitorType=visitor,
        Weekend=weekend,
    )
    res = pred.predict([session], explain=True, top_k=8)[0]
    p = res["purchase_probability"]
    left, right = st.columns([1, 2])
    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=p * 100,
                number={"suffix": "%"},
                title={"text": "Purchase probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": GREEN if res["prediction"] else RED},
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "value": res["threshold"] * 100,
                    },
                },
            )
        )
        gauge.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(gauge, width="stretch")
        st.markdown(
            f"**Decision:** {'✅ Likely to buy' if res['prediction'] else '❌ Unlikely to buy'}"
        )
        st.markdown(
            f"**Confidence:** {res['confidence']}  \n*(black line = decision threshold {res['threshold']:.2f})*"
        )
        if res["confidence"] == "low":
            st.warning("Close to the threshold - treat this prediction with caution.")
    with right:
        contrib = pd.DataFrame(res["top_contributors"]).iloc[::-1]
        contrib["label"] = contrib.apply(lambda r: f"{r['feature']} = {r['value']}", axis=1)
        fig = go.Figure(
            go.Bar(
                x=contrib["shap_value"],
                y=contrib["label"],
                orientation="h",
                marker_color=[GREEN if v > 0 else RED for v in contrib["shap_value"]],
            )
        )
        fig.update_layout(
            title="Why this prediction? (SHAP, green = pushes towards purchase)",
            height=380,
            margin=dict(l=0, r=10, t=50, b=0),
            xaxis_title="Impact on model output",
        )
        st.plotly_chart(fig, width="stretch")
    if st.checkbox("Show a second opinion with LIME"):
        with st.spinner("Computing LIME..."):
            lime = pd.DataFrame(
                pred.explainer.lime_explain(pd.DataFrame([session])[S.input_columns])
            )
        st.dataframe(lime, width="stretch", hide_index=True)


# ------------------------------------------------------------------ 3. behaviour analytics
def page_analytics(df: pd.DataFrame) -> None:
    st.header("Behaviour analytics")
    f1, f2 = st.columns(2)
    months = f1.multiselect("Month", MONTHS, default=MONTHS)
    visitors = f2.multiselect(
        "Visitor type", sorted(df.VisitorType.unique()), default=sorted(df.VisitorType.unique())
    )
    d = df[df.Month.isin(months) & df.VisitorType.isin(visitors)]
    if d.empty:
        st.warning("No sessions match these filters.")
        return
    st.caption(f"{len(d):,} sessions | conversion {d[S.target].mean():.1%}")
    c1, c2 = st.columns(2)
    by_month = (
        d.groupby("Month")[S.target]
        .mean()
        .reindex([m for m in MONTHS if m in months])
        .reset_index()
    )
    c1.plotly_chart(
        px.bar(
            by_month,
            x="Month",
            y=S.target,
            title="Conversion rate by month",
            color_discrete_sequence=[BLUE],
        ).update_yaxes(tickformat=".0%"),
        width="stretch",
    )
    by_v = d.groupby("VisitorType")[S.target].mean().reset_index()
    c2.plotly_chart(
        px.bar(
            by_v,
            x="VisitorType",
            y=S.target,
            title="Conversion by visitor type",
            color_discrete_sequence=[GREEN],
        ).update_yaxes(tickformat=".0%"),
        width="stretch",
    )
    c3, c4 = st.columns(2)
    c3.plotly_chart(
        px.box(
            d,
            x=S.target,
            y="PageValues",
            title="PageValues vs purchase (0=no, 1=yes)",
            color=S.target,
            color_discrete_sequence=[RED, GREEN],
        ),
        width="stretch",
    )
    funnel = d.assign(
        Bucket=pd.cut(
            d.ProductRelated,
            [-1, 5, 15, 30, 60, 10000],
            labels=["0-5", "6-15", "16-30", "31-60", "60+"],
        )
    )
    fb = funnel.groupby("Bucket", observed=True)[S.target].mean().reset_index()
    c4.plotly_chart(
        px.bar(
            fb,
            x="Bucket",
            y=S.target,
            title="Conversion by product pages viewed",
            color_discrete_sequence=[BLUE],
        ).update_yaxes(tickformat=".0%"),
        width="stretch",
    )
    wk = d.groupby("Weekend")[S.target].agg(["mean", "count"]).reset_index()
    st.dataframe(
        wk.rename(columns={"mean": "conversion_rate", "count": "sessions"}), hide_index=True
    )


# ------------------------------------------------------------------ 4. model comparison
def page_models() -> None:
    st.header("Model comparison")
    comp = get_report("model_comparison.csv")
    if comp is None:
        st.error("Run training first (reports/model_comparison.csv missing).")
        return
    metric = st.selectbox("Metric", ["pr_auc", "recall", "precision", "f1", "roc_auc", "accuracy"])
    best = comp.sort_values("pr_auc", ascending=False).drop_duplicates("model")
    st.plotly_chart(
        px.bar(
            best.sort_values(metric),
            x=metric,
            y="model",
            orientation="h",
            color="strategy",
            title=f"Best configuration per model - {metric} (5-fold CV)",
        ),
        width="stretch",
    )
    st.subheader("Effect of the imbalance strategy")
    st.plotly_chart(
        px.bar(comp[~comp.tuned], x="model", y=metric, color="strategy", barmode="group"),
        width="stretch",
    )
    st.caption(
        "PR-AUC is threshold-free, so it compares models fairly. Recall/precision here are at 0.5, "
        "which is why class weights and SMOTE raise recall."
    )
    cols = [
        "model",
        "strategy",
        "tuned",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
        "fit_time_s",
    ]
    st.dataframe(
        comp[cols].sort_values("pr_auc", ascending=False).round(4),
        hide_index=True,
        width="stretch",
    )


# ------------------------------------------------------------------ 5. explainability
def page_explain(pred: Predictor | None) -> None:
    st.header("Explainability")
    st.write(
        "**SHAP** shows how much each feature pushes a prediction up or down. "
        "**LIME** fits a small local model around one session as a second opinion."
    )
    imp = get_report("global_shap.csv")
    if imp is not None:
        st.plotly_chart(
            px.bar(
                imp.head(15).iloc[::-1],
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Global importance (mean |SHAP|, test sample)",
                color_discrete_sequence=[BLUE],
            ),
            width="stretch",
        )
    beeswarm = REPORTS / "figures" / "shap_beeswarm.png"
    if beeswarm.exists():
        st.image(
            str(beeswarm),
            caption="Each dot is a session; red = high feature value. Right = pushes towards purchase.",
        )
    ex = get_report("local_examples.json")
    if ex:
        st.subheader("Worked examples from the test set")
        tab = st.radio(
            "Example", list(ex), horizontal=True, format_func=lambda s: s.replace("_", " ").title()
        )
        e = ex[tab]
        st.write(
            f"Predicted probability **{e['probability']:.1%}** | actual outcome: **{'bought' if e['actual'] else 'did not buy'}**"
        )
        a, b = st.columns(2)
        a.markdown("**SHAP top contributors**")
        a.dataframe(
            pd.DataFrame(e["shap_top"])[["feature", "value", "shap_value"]], hide_index=True
        )
        b.markdown("**LIME rules**")
        b.dataframe(pd.DataFrame(e["lime_top"]), hide_index=True)
        with st.expander("Session inputs"):
            st.json(e["input"])


# ------------------------------------------------------------------ 6. model & system
def page_system(pred: Predictor | None) -> None:
    st.header("Model & system")
    if not need_model(pred):
        return
    m = pred.metadata
    t, d = m["test_metrics_at_threshold"], m["test_metrics_at_0.5"]
    st.subheader("Held-out test performance")
    tbl = pd.DataFrame({"threshold 0.5": d, f"chosen threshold {m['threshold']:.2f}": t}).loc[
        ["pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy", "tp", "fp", "fn", "tn"]
    ]
    st.dataframe(tbl.round(4), width="stretch")
    b = m["baseline"]
    st.caption(f"'Nobody buys' baseline: accuracy {b['accuracy']:.1%}, recall 0%.")
    c1, c2 = st.columns(2)
    cm = REPORTS / "figures" / "confusion_matrix.png"
    pr = REPORTS / "figures" / "pr_curve.png"
    if cm.exists():
        c1.image(str(cm))
    if pr.exists():
        c2.image(str(pr))
    st.subheader("Decision threshold chosen by business value")
    curve = get_report("threshold_curve.json")
    if curve:
        cdf = pd.DataFrame(curve)
        fig = px.line(
            cdf,
            x="threshold",
            y="net_value",
            title="Expected net value vs threshold (train out-of-fold)",
        )
        fig.add_vline(x=m["threshold"], line_dash="dash", line_color="red")
        st.plotly_chart(fig, width="stretch")
    biz = m["business_case"]
    st.write(
        f"Assumptions: value per saved purchase **{biz['value_per_conversion']}**, nudge cost **{biz['intervention_cost']}**, "
        f"success rate **{biz['success_rate']:.0%}** -> break-even probability **{m['break_even_probability']:.2f}**."
    )
    v = m["net_value_on_test"]
    st.dataframe(
        pd.DataFrame(
            {
                "strategy": ["Chosen threshold", "Default 0.5", "Nudge everyone", "Nudge nobody"],
                "expected net value (test)": [
                    v["chosen_threshold"],
                    v["default_0.5"],
                    v["flag_everyone"],
                    v["flag_nobody"],
                ],
            }
        ),
        hide_index=True,
    )
    abl = m.get("ablation_page_values")
    if abl:
        st.subheader("PageValues leakage check")
        st.write(
            f"PR-AUC with PageValues **{m['cv_metrics']['pr_auc']:.3f}** vs without **{abl['cv_pr_auc']:.3f}**. "
            "PageValues is checkout-proximal, so the model is best used late in a session; "
            "see the model card for the early-scoring trade-off."
        )
    st.subheader("Artifact & environment info")
    st.json(
        {
            k: m[k]
            for k in [
                "model_name",
                "strategy",
                "tuned",
                "use_page_values",
                "trained_at",
                "python",
                "sklearn",
                "seed",
                "n_train",
                "n_test",
            ]
        }
    )


# ------------------------------------------------------------------ router
PAGES = [
    "1. Overview",
    "2. Live prediction",
    "3. Behaviour analytics",
    "4. Model comparison",
    "5. Explainability",
    "6. Model & system",
]


def main() -> None:
    df = get_data()
    pred = get_predictor()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")
    st.sidebar.caption("Purchase Intent XAI · v" + str(S.section("project")["version"]))
    if page == PAGES[0]:
        page_overview(df, pred)
    elif page == PAGES[1]:
        page_live(df, pred)
    elif page == PAGES[2]:
        page_analytics(df)
    elif page == PAGES[3]:
        page_models()
    elif page == PAGES[4]:
        page_explain(pred)
    else:
        page_system(pred)


main()
