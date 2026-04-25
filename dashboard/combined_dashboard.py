import os

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from pyvis.network import Network

px.defaults.template = "plotly_white"

DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_COLUMNS = ["isolation_forest", "oneclass_svm", "autoencoder"]

st.set_page_config(
    page_title="Insider Threat Intelligence Dashboard",
    page_icon="IT",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme_mode = st.sidebar.selectbox("Theme", ["Normal", "Dark"], index=0)
is_dark = theme_mode == "Dark"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --bg: #eef2f8;
    --ink: #111827;
    --brand: #0f766e;
    --accent: #ea580c;
    --panel: #ffffff;
    --muted: #4b5563;
}

html, body, [class*="css"] {
    font-family: "Space Grotesk", sans-serif;
    color: var(--ink);
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(circle at 8% 10%, rgba(15, 118, 110, 0.18), transparent 34%),
        radial-gradient(circle at 90% 12%, rgba(234, 88, 12, 0.16), transparent 34%),
        linear-gradient(180deg, #f8fafc 0%, #eef2f8 100%);
}

[data-testid="stAppViewContainer"] .main .block-container {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #d7deeb;
    border-radius: 16px;
    padding: 1.2rem 1.4rem 2rem 1.4rem;
    box-shadow: 0 18px 44px rgba(15, 23, 42, 0.08);
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp li, .stApp span {
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.25);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] .stSelectbox label {
    color: #cbd5e1 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.35);
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid #d8deea;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 16px 34px rgba(15, 23, 42, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #475569;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #0f172a;
}

[data-baseweb="tab-list"] {
    gap: 12px;
}

[data-baseweb="tab"] {
    background: #e2e8f0;
    border-radius: 12px 12px 0 0;
    padding: 10px 18px;
    border: 1px solid #cbd5e1;
    color: #334155 !important;
    font-weight: 600;
}

[aria-selected="true"][data-baseweb="tab"] {
    background: #ffffff;
    border-bottom-color: #ffffff;
    border-top: 3px solid var(--accent);
    color: #0f172a !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #d7deeb;
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stPlotlyChart"] {
    background: #ffffff;
    border: 1px solid #d7deeb;
    border-radius: 12px;
    padding: 6px;
}

.hero-wrap {
    background: linear-gradient(120deg, rgba(15, 118, 110, 0.95), rgba(14, 116, 144, 0.95));
    color: #ffffff;
    border-radius: 18px;
    padding: 24px 26px;
    margin: 8px 0 14px 0;
    box-shadow: 0 20px 40px rgba(15, 118, 110, 0.25);
}

.hero-kicker {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #d1fae5;
}

.hero-title {
    font-size: 40px;
    line-height: 1.1;
    font-weight: 700;
    margin: 8px 0 10px 0;
    color: #ffffff;
}

.hero-sub {
    font-size: 16px;
    color: #ecfeff;
    margin: 0;
}
</style>
""",
    unsafe_allow_html=True,
)

if is_dark:
    st.markdown(
        """
<style>
:root {
    --bg: #0b1220;
    --ink: #e2e8f0;
    --brand: #22d3ee;
    --accent: #f97316;
    --panel: #111827;
    --muted: #94a3b8;
}

.stApp {
    background:
        radial-gradient(circle at 8% 10%, rgba(34, 211, 238, 0.16), transparent 34%),
        radial-gradient(circle at 90% 12%, rgba(249, 115, 22, 0.14), transparent 34%),
        linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
}

[data-testid="stAppViewContainer"] .main .block-container {
    background: rgba(17, 24, 39, 0.92);
    border: 1px solid #273449;
    box-shadow: 0 18px 44px rgba(2, 6, 23, 0.45);
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp li, .stApp span {
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0b1220 100%);
    border-right: 1px solid rgba(71, 85, 105, 0.5);
}

[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid #273449;
    box-shadow: 0 10px 28px rgba(2, 6, 23, 0.35);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

[data-testid="stMetricValue"] {
    color: #f8fafc;
}

[data-baseweb="tab"] {
    background: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1 !important;
}

[aria-selected="true"][data-baseweb="tab"] {
    background: #0f172a;
    border-bottom-color: #0f172a;
    color: #f8fafc !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #334155;
}

[data-testid="stPlotlyChart"] {
    background: #0f172a;
    border: 1px solid #334155;
}

.hero-wrap {
    background: linear-gradient(120deg, rgba(2, 132, 199, 0.9), rgba(15, 118, 110, 0.9));
    box-shadow: 0 18px 38px rgba(2, 6, 23, 0.5);
}

.hero-kicker {
    color: #bae6fd;
}

.hero-sub {
    color: #e0f2fe;
}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_data
def load_all_data():
    features = pd.read_csv(os.path.join(DATA_DIR, "merged_features.csv"))
    scores = pd.read_csv(os.path.join(DATA_DIR, "anomaly_scores.csv"))
    file_access = pd.read_csv(
        os.path.join(DATA_DIR, "file_access.csv"), parse_dates=["access_time"]
    )
    usb_usage = pd.read_csv(
        os.path.join(DATA_DIR, "usb_usage.csv"), parse_dates=["plug_time", "unplug_time"]
    )
    emails = pd.read_csv(
        os.path.join(DATA_DIR, "emails.csv"), parse_dates=["time"]
    )
    merged = pd.merge(features, scores, on="user", how="inner", suffixes=("_feat", "_score"))

    red_col_candidates = [
        "is_red_team_score",
        "is_red_team_feat",
        "is_red_team",
        "is_red_team_x",
        "is_red_team_y",
    ]
    red_col = next((c for c in red_col_candidates if c in merged.columns), None)
    merged["red_team_flag"] = merged[red_col].fillna(0).astype(int) if red_col else 0
    return merged, scores, file_access, usb_usage, emails


@st.cache_resource
def load_models():
    return {
        "isolation_forest": joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl")),
        "oneclass_svm": joblib.load(os.path.join(MODEL_DIR, "oneclass_svm.pkl")),
        "autoencoder": joblib.load(os.path.join(MODEL_DIR, "autoencoder.pkl")),
    }


def get_feature_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Training used merged_features.csv with `is_red_team` dropped; after merging with scores we can
    # end up with multiple `is_red_team*` columns (e.g. `is_red_team_feat`, `is_red_team_score`).
    red_team_cols = [c for c in numeric_df.columns if c.startswith("is_red_team")]
    drop_cols = ["red_team_flag", *red_team_cols, *MODEL_COLUMNS]

    return numeric_df.drop(
        columns=[c for c in drop_cols if c in numeric_df.columns],
        errors="ignore",
    )


def _parse_uploaded_csv(uploaded_file, parse_dates=None):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file, parse_dates=parse_dates or [])
    except Exception as exc:
        st.error(f"Failed to read {uploaded_file.name}: {exc}")
        return None


def _read_csv_from_path(path_value, parse_dates=None, label="file"):
    if not path_value:
        return None
    try:
        return pd.read_csv(path_value, parse_dates=parse_dates or [])
    except Exception as exc:
        st.error(f"Failed to read {label} from path '{path_value}': {exc}")
        return None


def _empty_df(columns):
    return pd.DataFrame(columns=columns)


def _sender_to_user(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace("@company.com", "", regex=False)
    )


def extract_features_for_single_day(logins_df, file_access_df, usb_usage_df, emails_df, selected_day):
    day_ts = pd.Timestamp(selected_day)

    # Apply day filter to each source so the output represents exactly one day.
    ldf = logins_df[logins_df["login"].dt.date == day_ts.date()].copy()
    fdf = file_access_df[file_access_df["access_time"].dt.date == day_ts.date()].copy()
    udf = usb_usage_df[usb_usage_df["plug_time"].dt.date == day_ts.date()].copy()
    edf = emails_df[emails_df["time"].dt.date == day_ts.date()].copy()
    if not edf.empty and "sender" in edf.columns:
        edf["user"] = _sender_to_user(edf["sender"])

    user_pool = set()
    if "user" in ldf.columns:
        user_pool.update(ldf["user"].dropna().astype(str).tolist())
    if "user" in fdf.columns:
        user_pool.update(fdf["user"].dropna().astype(str).tolist())
    if "user" in udf.columns:
        user_pool.update(udf["user"].dropna().astype(str).tolist())
    if "user" in edf.columns:
        user_pool.update(edf["user"].dropna().astype(str).tolist())

    users = sorted(user_pool)
    if not users:
        return pd.DataFrame()

    graph = nx.Graph()
    for _, row in fdf.iterrows():
        graph.add_edge(row["user"], row["file"])
    for _, row in udf.iterrows():
        graph.add_edge(row["user"], row["device"])
    degree = nx.degree_centrality(graph) if graph.number_of_nodes() > 0 else {}
    betweenness = nx.betweenness_centrality(graph) if graph.number_of_nodes() > 0 else {}

    keyword_regex = r"Confidential|Emergency|Secret|Password"
    rows = []
    for user in users:
        user_logins = ldf[ldf["user"] == user]
        user_files = fdf[fdf["user"] == user]
        user_usb = udf[udf["user"] == user]
        user_emails = edf[edf["user"] == user] if "user" in edf.columns else _empty_df(["subject"])

        mean_login_hour = user_logins["login"].dt.hour.mean() if not user_logins.empty else np.nan
        mean_logout_hour = user_logins["logout"].dt.hour.mean() if not user_logins.empty else np.nan
        files_per_day = float(len(user_files))
        usb_per_day = float(len(user_usb))
        emails_per_day = float(len(user_emails))

        out_of_session = 0
        if not user_files.empty and not user_logins.empty:
            for _, file_row in user_files.iterrows():
                in_session = user_logins[
                    (user_logins["login"] <= file_row["access_time"])
                    & (user_logins["logout"] >= file_row["access_time"])
                ]
                if in_session.empty:
                    out_of_session += 1
        elif not user_files.empty and user_logins.empty:
            out_of_session = int(len(user_files))

        keyword_flag = 0.0
        subject_len = np.nan
        sentiment = 0.0
        if not user_emails.empty:
            subj = user_emails["subject"].fillna("").astype(str)
            keyword_flag = float(subj.str.contains(keyword_regex, case=False, regex=True).mean())
            subject_len = float(subj.str.len().mean())

        rows.append(
            {
                "user": user,
                "mean_login_hour": mean_login_hour,
                "mean_logout_hour": mean_logout_hour,
                "files_per_day": files_per_day,
                "usb_per_day": usb_per_day,
                "emails_per_day": emails_per_day,
                "out_of_session_access": float(out_of_session),
                "degree_centrality": float(degree.get(user, 0.0)),
                "betweenness_centrality": float(betweenness.get(user, 0.0)),
                "keyword_flag": keyword_flag,
                "subject_len": subject_len,
                "sentiment": sentiment,
            }
        )

    return pd.DataFrame(rows)


def score_input_features(input_features_df, baseline_df, models, red_team_users=None):
    expected_cols = list(get_feature_matrix(baseline_df).columns)
    baseline_X = baseline_df[expected_cols].copy()
    fill_values = baseline_X.median(numeric_only=True)

    working = input_features_df.copy()
    for col in expected_cols:
        if col not in working.columns:
            working[col] = fill_values.get(col, 0.0)

    working = working[["user", *expected_cols]]
    X_input = working[expected_cols].copy()
    X_input = X_input.fillna(fill_values).fillna(0.0)

    # Recreate training-time scaling from current baseline feature set.
    mu = baseline_X.mean()
    sigma = baseline_X.std(ddof=0).replace(0, 1.0)
    baseline_scaled = (baseline_X - mu) / sigma
    input_scaled = (X_input - mu) / sigma

    iso_scores = -models["isolation_forest"].score_samples(input_scaled)
    svm_scores = -models["oneclass_svm"].decision_function(input_scaled)
    auto = models["autoencoder"]
    baseline_recon = np.mean((baseline_scaled - auto.predict(baseline_scaled)) ** 2, axis=1)
    auto_recon = np.mean((input_scaled - auto.predict(input_scaled)) ** 2, axis=1)

    result = working.copy()
    red_team_users = set(red_team_users or [])

    result["red_team_flag"] = result["user"].astype(str).isin(red_team_users).astype(int)
    result["Red Team"] = result["red_team_flag"].map({1: "🚩", 0: ""})
    result["isolation_forest"] = iso_scores
    result["oneclass_svm"] = svm_scores
    result["autoencoder"] = auto_recon
    result["risk_score"] = result[MODEL_COLUMNS].max(axis=1)
    result["risk_percentile"] = (
        pd.Series(auto_recon).rank(pct=True).values * 100.0
    )
    # Compare input reconstruction error against baseline for a practical risk status.
    p90 = np.percentile(baseline_recon, 90)
    p98 = np.percentile(baseline_recon, 98)
    result["risk_level"] = np.where(
        result["autoencoder"] >= p98,
        "High",
        np.where(result["autoencoder"] >= p90, "Medium", "Low"),
    )
    return result.sort_values("risk_score", ascending=False)


def _validate_columns(df, required_cols, dataset_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"{dataset_name} is missing required columns: {', '.join(missing)}")
        return False
    return True


def _load_project_red_team_users():
    red_team_users_path = os.path.join(DATA_DIR, "red_team_users.csv")
    if os.path.exists(red_team_users_path):
        red_df = pd.read_csv(red_team_users_path)
        if "user" in red_df.columns:
            return set(red_df["user"].astype(str).tolist())
    return set()


def apply_plot_style(fig):
    if is_dark:
        font_color = "#e2e8f0"
        title_color = "#f8fafc"
        plot_bg = "#0f172a"
        paper_bg = "#0f172a"
        grid_color = "#243349"
        zero_color = "#334155"
        axis_color = "#64748b"
    else:
        font_color = "#0f172a"
        title_color = "#0f172a"
        plot_bg = "#ffffff"
        paper_bg = "#ffffff"
        grid_color = "#e2e8f0"
        zero_color = "#cbd5e1"
        axis_color = "#94a3b8"

    fig.update_layout(
        font={"color": font_color, "size": 13},
        title_font={"color": title_color, "size": 30},
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        margin={"l": 54, "r": 40, "t": 72, "b": 56},
        coloraxis_colorbar={
            "tickfont": {"color": font_color},
            "title": {"font": {"color": font_color}},
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        zerolinecolor=zero_color,
        linecolor=axis_color,
        tickfont={"color": font_color},
        title_font={"color": font_color},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        zerolinecolor=zero_color,
        linecolor=axis_color,
        tickfont={"color": font_color},
        title_font={"color": font_color},
    )
    return fig


def get_shap_explanation(user_id, selected_model, models, base_df):
    X = get_feature_matrix(base_df)
    user_idx = base_df[base_df["user"] == user_id].index[0]
    row = X.loc[user_idx:user_idx]
    model = models[selected_model]

    if selected_model == "isolation_forest":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row)
    else:
        background = X.sample(min(30, len(X)), random_state=42)
        # Autoencoder is a regressor that reconstructs features; explain scalar reconstruction error.
        if selected_model == "autoencoder":
            def predict_fn(data):
                recon = model.predict(data)
                return np.mean((data - recon) ** 2, axis=1)
        else:
            # Use decision_function if available, otherwise use predict.
            predict_fn = getattr(model, "decision_function", None) or model.predict
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(row, nsamples=80)

    shap_array = np.array(shap_values[0] if isinstance(shap_values, list) else shap_values)
    # SHAP can return (1, n_features, n_outputs) for vector outputs; reduce to per-feature impacts.
    if shap_array.ndim == 3:
        shap_array = np.mean(shap_array, axis=2)
    elif shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    if shap_array.shape[1] != len(X.columns):
        shap_array = shap_array[:, : len(X.columns)]

    return shap_array, list(X.columns), row.iloc[0]


def get_counterfactual(user_id, selected_model, models, base_df):
    shap_vals, feature_names, user_vector = get_shap_explanation(
        user_id=user_id,
        selected_model=selected_model,
        models=models,
        base_df=base_df,
    )
    X = get_feature_matrix(base_df)

    impacts = shap_vals[0]
    top_idx = np.argsort(np.abs(impacts))[-4:]
    target = user_vector.copy()
    for idx in top_idx:
        feature = feature_names[idx]
        target[feature] = X[feature].median()

    cf_df = pd.DataFrame(
        {
            "feature": feature_names,
            "current": user_vector.values,
            "target": target.values,
            "change": target.values - user_vector.values,
        }
    )
    cf_df = cf_df[cf_df["change"] != 0].sort_values("change", key=np.abs, ascending=False)
    return cf_df


def build_graph(file_access_df, usb_usage_df):
    graph = nx.Graph()
    for _, row in file_access_df.iterrows():
        graph.add_edge(row["user"], row["file"], edge_type="access")
    for _, row in usb_usage_df.iterrows():
        graph.add_edge(row["user"], row["device"], edge_type="usb")
    return graph


def high_risk_subgraph(graph, merged_df):
    score = merged_df[MODEL_COLUMNS].max(axis=1)
    risky_users = set(merged_df.loc[(score > 1.0) | (merged_df["red_team_flag"] == 1), "user"])
    scope = set()
    for user in risky_users:
        if user in graph:
            scope.add(user)
            scope.update(graph.neighbors(user))
    return graph.subgraph(scope).copy()


df, score_df, file_access, usb_usage, emails = load_all_data()
models = load_models()
graph = build_graph(file_access, usb_usage)

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-kicker">Security Analytics Console</div>
        <div class="hero-title">Insider Threat Intelligence Dashboard</div>
        <p class="hero-sub">Unified operations, anomaly review, graph context, and explainability in one interface.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Controls")
selected_user = st.sidebar.selectbox("User", sorted(df["user"].unique()))
selected_model = st.sidebar.selectbox("Scoring model", MODEL_COLUMNS)

tabs = st.tabs(
    [
        "Overview",
        "User Detail",
        "Relationship Graph",
        "Explainability",
        "Counterfactuals",
        "Input Prediction",
        "Methodology",
    ]
)

with tabs[0]:
    st.subheader("Security Posture")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total users", len(df))
    c2.metric("Flagged by red team", int(df["red_team_flag"].sum()))
    c3.metric("Mean anomaly score", f"{df[selected_model].mean():.3f}")
    c4.metric("Top score", f"{df[selected_model].max():.3f}")

    left, right = st.columns([1.4, 1])
    with left:
        hist = px.histogram(
            df,
            x=selected_model,
            color="red_team_flag",
            barmode="overlay",
            nbins=30,
            color_discrete_map={0: "#0f766e", 1: "#d97706"},
            title=f"Score distribution: {selected_model}",
            labels={"red_team_flag": "red team"},
        )
        hist.update_layout(
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
        hist = apply_plot_style(hist)
        st.plotly_chart(hist, width="stretch")

    with right:
        top_n = (
            df[["user", "red_team_flag", *MODEL_COLUMNS]]
            .sort_values(selected_model, ascending=False)
            .head(12)
            .rename(columns={"red_team_flag": "red_team"})
        )
        st.dataframe(top_n, width="stretch")

with tabs[1]:
    user_row = df[df["user"] == selected_user].iloc[0]
    st.subheader(f"User profile: {selected_user}")
    
    if user_row["red_team_flag"] == 1:
        st.error("🚨 **RED TEAM THREAT DETECTED** 🚨")
        st.markdown(f"**Reason for User {selected_user} Flagging:**")
        
        # Analyze Email Activity for this user
        user_emails = emails[emails["sender"] == selected_user].copy()
        if not user_emails.empty:
            suspicious_emails = user_emails[
                (user_emails["time"].dt.hour < 5) | 
                (user_emails["subject"].str.contains("Confidential|Emergency|Secret|Password", case=False, na=False))
            ]
            if not suspicious_emails.empty:
                st.error(f"**Suspicious Email Activity:** Detected {len(suspicious_emails)} anomalous emails (after-hours or suspicious subjects).")
                for _, row in suspicious_emails.iterrows():
                    st.write(f"- **To:** {row['recipient']} | **Time:** {row['time']} | **Subject:** {row['subject']}")
            
            # Check mass emailing
            if not user_emails.empty:
                email_counts = user_emails.set_index('time').resample('h').size()
                mass_emails = email_counts[email_counts > 10]
                if not mass_emails.empty:
                    st.error(f"**Mass Emailing:** Detected high-volume email anomalies.")
                    
        # Analyze USB usage for this user
        user_usb = usb_usage[usb_usage["user"] == selected_user].copy()
        if not user_usb.empty:
            user_usb["duration_mins"] = (user_usb["unplug_time"] - user_usb["plug_time"]).dt.total_seconds() / 60
            # Red team USB behavior is specifically injected as being BOTH after-hours AND > 60 mins
            suspicious_usb = user_usb[(user_usb["plug_time"].dt.hour < 5) & (user_usb["duration_mins"] > 60)]
            if not suspicious_usb.empty:
                st.warning(f"**Suspicious USB Activity:** Detected anomalous USB connections (after-hours and long duration).")
                for _, row in suspicious_usb.iterrows():
                    plug_date = row['plug_time'].strftime("%Y-%m-%d")
                    plug_time_str = row['plug_time'].strftime("%H:%M:%S")
                    hours = int(row['duration_mins'] // 60)
                    mins = int(row['duration_mins'] % 60)
                    duration_str = f"{hours} hours {mins} mins" if hours > 0 else f"{mins} mins"
                    st.write(f"- **Device:** {row['device']} | **Use Date:** {plug_date} | **Used Time:** {plug_time_str} | **Used Hours:** {duration_str}")
                
        # Analyze File Access for this user
        user_files = file_access[file_access["user"] == selected_user].copy()
        if not user_files.empty:
            after_hours_files = user_files[user_files["access_time"].dt.hour < 5]
            if not after_hours_files.empty:
                st.error(f"**Suspicious File Access:** Detected {len(after_hours_files)} file accesses between 12 AM and 5 AM.")
            
            # Check mass downloads
            file_counts = user_files.set_index('access_time').resample('h').size()
            mass_downloads = file_counts[file_counts > 10]
            if not mass_downloads.empty:
                st.error(f"**Mass File Access:** Detected high-volume file access anomalies. (Data Exfiltration Risk)")
                peak_time = mass_downloads.idxmax()
                peak_count = mass_downloads.max()
                peak_window_end = peak_time + pd.Timedelta(hours=1)
                peak_files = user_files[(user_files['access_time'] >= peak_time) & (user_files['access_time'] < peak_window_end)]['file'].unique()
                
                st.write(f"- **Peak Time:** {peak_time} | **Files Accessed in 1 Hour:** {peak_count}")
                st.write(f"- **Downloaded Files:** {', '.join(peak_files)}")
                
    st.markdown("---")
    
    u1, u2, u3 = st.columns(3)
    u1.metric("Red team status", "YES" if user_row["red_team_flag"] == 1 else "NO")
    u2.metric("Current score", f"{user_row[selected_model]:.3f}")
    rank = int(df[selected_model].rank(ascending=False)[df["user"] == selected_user].iloc[0])
    u3.metric("Risk rank", f"#{rank}")

    numeric_cols = get_feature_matrix(df).columns
    user_values = user_row[numeric_cols]
    avg_values = df[numeric_cols].mean()
    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(r=user_values.values, theta=numeric_cols, fill="toself", name=selected_user)
    )
    radar.add_trace(
        go.Scatterpolar(r=avg_values.values, theta=numeric_cols, fill="toself", name="population_mean")
    )
    radar.update_layout(
        title="Behavior fingerprint",
        polar={"radialaxis": {"visible": True}},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    radar = apply_plot_style(radar)
    st.plotly_chart(radar, width="stretch")

with tabs[2]:
    st.subheader("At-risk relationship map")
    sub_graph = high_risk_subgraph(graph, df)
    network = Network(height="900px", width="100%", notebook=False, bgcolor="#222222", font_color="white")
    network.barnes_hut(
        gravity=-2000,
        central_gravity=0.1,
        spring_length=200,
        spring_strength=0.01,
        damping=0.85,
        overlap=1,
    )
    network.set_options(
        """
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "fit": true, "iterations": 2500, "updateInterval": 50},
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.01,
          "damping": 0.85,
          "avoidOverlap": 1
        }
      }
    }
    """
    )

    user_nodes = set(df["user"])

    for node in sub_graph.nodes():
        if node in user_nodes:
            node_row = df[df["user"] == node].iloc[0]
            score = float(node_row[MODEL_COLUMNS].max())
            is_red = int(node_row["red_team_flag"]) == 1
            if node == selected_user:
                color = "#ff4b4b"
                size = 34
            else:
                color = "red" if is_red else ("orange" if score > 1.5 else "yellow" if score > 1.0 else "lightblue")
                size = 30 if is_red else (20 if score > 1.5 else 15 if score > 1.0 else 10)
            title = f"User: {node}<br>Anomaly Score: {score:.2f}<br>Red Team: {'Yes' if is_red else 'No'}"
        elif str(node).startswith("file"):
            color = "green"
            size = 8
            title = f"File: {node}"
        elif str(node).startswith("usb"):
            color = "purple"
            size = 8
            title = f"Device: {node}"
        else:
            color = "gray"
            size = 8
            title = str(node)

        network.add_node(node, label=str(node), color=color, size=size, title=title)

    for source, target, attrs in sub_graph.edges(data=True):
        edge_type = attrs.get("edge_type") or attrs.get("type")
        edge_color = "gray" if edge_type == "access" else "purple"
        network.add_edge(source, target, color=edge_color)

    graph_file = os.path.join("dashboard", "graph.html")
    network.save_graph(graph_file)
    with open(graph_file, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=900, scrolling=False)

with tabs[3]:
    st.subheader("Feature attribution")
    with st.spinner("Computing SHAP attribution..."):
        shap_values, features, _ = get_shap_explanation(selected_user, selected_model, models, df)
    impact_df = pd.DataFrame(
        {
            "feature": features,
            "impact": shap_values[0],
            "abs_impact": np.abs(shap_values[0]),
        }
    ).sort_values("abs_impact", ascending=False)
    bar = px.bar(
        impact_df.head(12),
        x="impact",
        y="feature",
        orientation="h",
        color="impact",
        color_continuous_scale="RdBu_r",
        title=f"Top contributors for {selected_user}",
    )
    bar = apply_plot_style(bar)
    st.plotly_chart(bar, width="stretch")

with tabs[4]:
    st.subheader("What-to-change guidance")
    with st.spinner("Generating counterfactual plan..."):
        changes = get_counterfactual(selected_user, selected_model, models, df)
    if changes.empty:
        st.success("This user is already close to baseline behavior.")
    else:
        st.dataframe(changes, width="stretch")
        comp = go.Figure()
        comp.add_trace(go.Bar(name="current", x=changes["feature"], y=changes["current"]))
        comp.add_trace(go.Bar(name="target", x=changes["feature"], y=changes["target"]))
        comp.update_layout(
            barmode="group",
            title="Current vs target values",
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
        comp = apply_plot_style(comp)
        st.plotly_chart(comp, width="stretch")

with tabs[5]:
    st.subheader("Single-day prediction and analysis from raw input data")
    st.markdown(
        "Use this section to run prediction directly on **raw logs** for one selected day. "
        "You can use existing project data or upload your own CSV files."
    )

    input_mode = st.radio(
        "Raw data source",
        ["Use existing raw data", "Upload new raw CSV files", "Use CSV file paths"],
        horizontal=True,
    )

    can_process = True
    if input_mode == "Upload new raw CSV files":
        c1, c2 = st.columns(2)
        with c1:
            up_logins = st.file_uploader("Upload logins.csv", type=["csv"], key="up_logins")
            up_files = st.file_uploader("Upload file_access.csv", type=["csv"], key="up_files")
        with c2:
            up_usb = st.file_uploader("Upload usb_usage.csv", type=["csv"], key="up_usb")
            up_emails = st.file_uploader("Upload emails.csv", type=["csv"], key="up_emails")

        input_logins = _parse_uploaded_csv(up_logins, parse_dates=["login", "logout"])
        input_files = _parse_uploaded_csv(up_files, parse_dates=["access_time"])
        input_usb = _parse_uploaded_csv(up_usb, parse_dates=["plug_time", "unplug_time"])
        input_emails = _parse_uploaded_csv(up_emails, parse_dates=["time"])

        if any(x is None for x in [input_logins, input_files, input_usb, input_emails]):
            st.info("Upload all 4 files to run single-day prediction.")
            can_process = False
    elif input_mode == "Use CSV file paths":
        st.caption("Paste absolute paths to your single-day raw files.")
        p1, p2 = st.columns(2)
        with p1:
            login_path = st.text_input(
                "logins.csv path",
                value=r"c:\ai-powered-insider-threat-detection-system\data purpose\1updated\data\2026-04-20_logins.csv",
                key="path_logins",
            )
            files_path = st.text_input(
                "file_access.csv path",
                value=r"c:\ai-powered-insider-threat-detection-system\data purpose\1updated\data\2026-04-20_file_access.csv",
                key="path_file_access",
            )
        with p2:
            usb_path = st.text_input(
                "usb_usage.csv path",
                value=r"c:\ai-powered-insider-threat-detection-system\data purpose\1updated\data\2026-04-20_usb_usage.csv",
                key="path_usb",
            )
            emails_path = st.text_input(
                "emails.csv path",
                value=r"c:\ai-powered-insider-threat-detection-system\data purpose\1updated\data\2026-04-20_emails.csv",
                key="path_emails",
            )

        if any(not p.strip() for p in [login_path, files_path, usb_path, emails_path]):
            st.info("Provide all 4 file paths to run single-day prediction.")
            can_process = False
            input_logins = _empty_df(["user", "login", "logout"])
            input_files = _empty_df(["user", "file", "access_time"])
            input_usb = _empty_df(["user", "device", "plug_time", "unplug_time"])
            input_emails = _empty_df(["sender", "recipient", "time", "subject"])
        else:
            input_logins = _read_csv_from_path(login_path, parse_dates=["login", "logout"], label="logins.csv")
            input_files = _read_csv_from_path(files_path, parse_dates=["access_time"], label="file_access.csv")
            input_usb = _read_csv_from_path(usb_path, parse_dates=["plug_time", "unplug_time"], label="usb_usage.csv")
            input_emails = _read_csv_from_path(emails_path, parse_dates=["time"], label="emails.csv")
            if any(x is None for x in [input_logins, input_files, input_usb, input_emails]):
                can_process = False
    else:
        input_logins = pd.read_csv(
            os.path.join(DATA_DIR, "logins.csv"), parse_dates=["login", "logout"]
        )
        input_files = pd.read_csv(
            os.path.join(DATA_DIR, "file_access.csv"), parse_dates=["access_time"]
        )
        input_usb = pd.read_csv(
            os.path.join(DATA_DIR, "usb_usage.csv"), parse_dates=["plug_time", "unplug_time"]
        )
        input_emails = pd.read_csv(
            os.path.join(DATA_DIR, "emails.csv"), parse_dates=["time"]
        )

    if can_process:
        can_process &= _validate_columns(input_logins, ["user", "login", "logout"], "logins.csv")
        can_process &= _validate_columns(input_files, ["user", "file", "access_time"], "file_access.csv")
        can_process &= _validate_columns(input_usb, ["user", "device", "plug_time", "unplug_time"], "usb_usage.csv")
        can_process &= _validate_columns(input_emails, ["sender", "recipient", "time", "subject"], "emails.csv")

    available_days = []
    if can_process:
        available_days = sorted(
            {
                *input_logins["login"].dropna().dt.date.tolist(),
                *input_files["access_time"].dropna().dt.date.tolist(),
                *input_usb["plug_time"].dropna().dt.date.tolist(),
                *input_emails["time"].dropna().dt.date.tolist(),
            }
        )

    if can_process and not available_days:
        st.warning("No date values found in raw inputs.")
    elif can_process:
        selected_day = st.date_input(
            "Select single day for prediction",
            value=available_days[0],
            min_value=min(available_days),
            max_value=max(available_days),
        )

        day_features = extract_features_for_single_day(
            input_logins,
            input_files,
            input_usb,
            input_emails,
            selected_day,
        )

        if day_features.empty:
            st.warning("No user activity found for the selected day.")
        else:
            if input_mode == "Use existing raw data":
                default_red_idx = 1
            else:
                default_red_idx = 0

            red_source = st.radio(
                "Red Team source for single-day view",
                [
                    "No red team labels",
                    "Use project red_team_users.csv",
                    "Upload red_team_users.csv",
                    "Enter users manually",
                ],
                index=default_red_idx,
                horizontal=True,
                key="single_day_red_source",
            )

            red_team_users = set()
            if red_source == "Use project red_team_users.csv":
                red_team_users = _load_project_red_team_users()
            elif red_source == "Upload red_team_users.csv":
                up_red = st.file_uploader(
                    "Upload red_team_users.csv (must contain 'user' column)",
                    type=["csv"],
                    key="up_red_team",
                )
                if up_red is not None:
                    red_df = _parse_uploaded_csv(up_red)
                    if red_df is not None and "user" in red_df.columns:
                        red_team_users = set(red_df["user"].astype(str).tolist())
                    elif red_df is not None:
                        st.error("Uploaded red_team_users.csv must contain 'user' column.")
            elif red_source == "Enter users manually":
                red_user_text = st.text_input(
                    "Enter red-team users (comma separated)",
                    value="",
                    key="manual_red_users",
                )
                if red_user_text.strip():
                    red_team_users = {
                        token.strip() for token in red_user_text.split(",") if token.strip()
                    }

            day_scores = score_input_features(
                day_features,
                df,
                models,
                red_team_users=red_team_users,
            )
            st.success(
                f"Predicted {len(day_scores)} users for {pd.Timestamp(selected_day).date()}"
            )

            sub_tabs = st.tabs(["Single-day Overview", "Single-day User Detail", "Single-day Raw Events"])

            with sub_tabs[0]:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Users scored", len(day_scores))
                s2.metric("Mean score", f"{day_scores[selected_model].mean():.3f}")
                s3.metric("Top risk score", f"{day_scores['risk_score'].max():.3f}")
                s4.metric("Red team users", int(day_scores["red_team_flag"].sum()))

                hist_day = px.histogram(
                    day_scores,
                    x=selected_model,
                    color="red_team_flag",
                    nbins=25,
                    title=f"Single-day score distribution: {selected_model}",
                    labels={"red_team_flag": "red team"},
                    color_discrete_map={0: "#0f766e", 1: "#b91c1c"},
                )
                hist_day = apply_plot_style(hist_day)
                st.plotly_chart(hist_day, width="stretch")

                view_cols = [
                    "user",
                    "Red Team",
                    "risk_score",
                    "isolation_forest",
                    "oneclass_svm",
                    "autoencoder",
                    "files_per_day",
                    "usb_per_day",
                    "emails_per_day",
                    "out_of_session_access",
                ]
                st.dataframe(day_scores[view_cols], width="stretch")

                st.markdown("### Generate Red Team List for This Day")
                gen_col1, gen_col2 = st.columns(2)
                with gen_col1:
                    threshold_method = st.selectbox(
                        "Red team detection method",
                        ["Top N users", "Score threshold", "Percentile threshold"],
                        key="threshold_method",
                    )
                with gen_col2:
                    if threshold_method == "Top N users":
                        threshold_val = st.number_input(
                            "Number of top users to flag",
                            min_value=1,
                            max_value=len(day_scores),
                            value=min(2, len(day_scores)),
                            key="top_n",
                        )
                        flagged_users = set(
                            day_scores.nlargest(int(threshold_val), selected_model)["user"].tolist()
                        )
                    elif threshold_method == "Score threshold":
                        threshold_val = st.slider(
                            f"Minimum {selected_model} score",
                            min_value=float(day_scores[selected_model].min()),
                            max_value=float(day_scores[selected_model].max()),
                            value=float(day_scores[selected_model].quantile(0.75)),
                            key="score_thresh",
                        )
                        flagged_users = set(
                            day_scores[day_scores[selected_model] >= threshold_val]["user"].tolist()
                        )
                    else:  # Percentile
                        threshold_val = st.slider(
                            "Percentile (0-100)",
                            min_value=0,
                            max_value=100,
                            value=75,
                            key="perc_thresh",
                        )
                        cutoff_score = day_scores[selected_model].quantile(threshold_val / 100.0)
                        flagged_users = set(
                            day_scores[day_scores[selected_model] >= cutoff_score]["user"].tolist()
                        )

                if flagged_users:
                    st.info(f"Detected {len(flagged_users)} suspicious user(s): {', '.join(sorted(flagged_users))}")
                    
                    flagged_df = pd.DataFrame({"user": sorted(flagged_users)})
                    csv_str = flagged_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download as red_team_users.csv",
                        data=csv_str,
                        file_name=f"red_team_users_{pd.Timestamp(selected_day).date()}.csv",
                        mime="text/csv",
                        key="download_red_team",
                    )
                else:
                    st.success("No suspicious users detected with current threshold.")

            with sub_tabs[1]:
                selected_input_user = st.selectbox(
                    "Select user for single-day detail",
                    day_scores["user"].tolist(),
                    key="single_day_user",
                )
                urow = day_scores[day_scores["user"] == selected_input_user].iloc[0]

                m1, m2, m3 = st.columns(3)
                m1.metric("Red team status", "YES" if int(urow["red_team_flag"]) == 1 else "NO")
                m2.metric("Selected model score", f"{urow[selected_model]:.3f}")
                u_rank = int(day_scores[selected_model].rank(ascending=False)[day_scores["user"] == selected_input_user].iloc[0])
                m3.metric("Single-day score rank", f"#{u_rank}")

                if int(urow["red_team_flag"]) == 1:
                    st.error("🚨 This selected user is marked as Red Team.")

                day_numeric_cols = get_feature_matrix(day_scores).columns
                radar_user_values = urow[day_numeric_cols]
                radar_avg_values = day_scores[day_numeric_cols].mean()
                day_radar = go.Figure()
                day_radar.add_trace(
                    go.Scatterpolar(
                        r=radar_user_values.values,
                        theta=day_numeric_cols,
                        fill="toself",
                        name=selected_input_user,
                    )
                )
                day_radar.add_trace(
                    go.Scatterpolar(
                        r=radar_avg_values.values,
                        theta=day_numeric_cols,
                        fill="toself",
                        name="day_mean",
                    )
                )
                day_radar.update_layout(
                    title="Single-day behavior fingerprint",
                    polar={"radialaxis": {"visible": True}},
                    legend={"orientation": "h", "y": 1.08, "x": 0},
                )
                day_radar = apply_plot_style(day_radar)
                st.plotly_chart(day_radar, width="stretch")

                st.markdown("### Automated reason summary")
                reasons = []
                if urow["out_of_session_access"] > 0:
                    reasons.append(f"Out-of-session file accesses: {int(urow['out_of_session_access'])}")
                if urow["files_per_day"] >= day_scores["files_per_day"].quantile(0.9):
                    reasons.append("File access volume is in the top 10% for this day")
                if urow["usb_per_day"] >= day_scores["usb_per_day"].quantile(0.9):
                    reasons.append("USB activity is in the top 10% for this day")
                if urow["emails_per_day"] >= day_scores["emails_per_day"].quantile(0.9):
                    reasons.append("Email activity is in the top 10% for this day")
                if not reasons:
                    st.success("No strong abnormal pattern found compared with other users on this day.")
                else:
                    for msg in reasons:
                        st.warning(msg)

            with sub_tabs[2]:
                selected_input_user = st.selectbox(
                    "Select user for raw event view",
                    day_scores["user"].tolist(),
                    key="single_day_user_events",
                )

                user_file_events = input_files[
                    (input_files["user"] == selected_input_user)
                    & (input_files["access_time"].dt.date == pd.Timestamp(selected_day).date())
                ]
                user_usb_events = input_usb[
                    (input_usb["user"] == selected_input_user)
                    & (input_usb["plug_time"].dt.date == pd.Timestamp(selected_day).date())
                ]
                user_email_events = input_emails[
                    (_sender_to_user(input_emails["sender"]) == selected_input_user)
                    & (input_emails["time"].dt.date == pd.Timestamp(selected_day).date())
                ]

                d1, d2 = st.columns(2)
                with d1:
                    st.caption("Single-day file access")
                    st.dataframe(user_file_events.head(200), width="stretch")
                    st.caption("Single-day USB activity")
                    st.dataframe(user_usb_events.head(200), width="stretch")
                with d2:
                    st.caption("Single-day email activity")
                    st.dataframe(user_email_events.head(200), width="stretch")

with tabs[6]:
    st.subheader("How the unified dashboard works")
    st.markdown(
        """
1. Data from behavioral logs and engineered features is merged with model scores.
2. Isolation Forest, One-Class SVM, and Autoencoder scores are available for comparison.
3. Network graph context shows user-to-resource links around high-risk nodes.
4. SHAP provides local feature attribution per selected user and selected model.
5. Counterfactual guidance proposes practical value shifts toward baseline behavior.
6. Input Prediction lets you provide raw log files and score/inspect a single selected day.
"""
    )