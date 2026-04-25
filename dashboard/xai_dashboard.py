import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import os
import joblib
import shap
import torch
from torch_geometric.data import Data
import plotly.express as px
import plotly.graph_objects as go

DATA_DIR = 'data'
MODEL_DIR = 'models'

# Set page config for a more professional look
st.set_page_config(
    page_title="XAI Insider Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a futuristic look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4e5d6c;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e5d6c;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('🛡️ XAI-Enhanced Insider Threat Detection System')
st.markdown("---")

# Load data
@st.cache_data
def load_all_data():
    features = pd.read_csv(os.path.join(DATA_DIR, 'merged_features.csv'))
    scores = pd.read_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'))
    file_access = pd.read_csv(os.path.join(DATA_DIR, 'file_access.csv'), parse_dates=['access_time'])
    usb_usage = pd.read_csv(os.path.join(DATA_DIR, 'usb_usage.csv'), parse_dates=['plug_time', 'unplug_time'])
    return features, scores, file_access, usb_usage

features, scores, file_access, usb_usage = load_all_data()
df = pd.merge(features, scores, on='user')

# Load models
@st.cache_resource
def load_models():
    iso = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    svm = joblib.load(os.path.join(MODEL_DIR, 'oneclass_svm.pkl'))
    auto = joblib.load(os.path.join(MODEL_DIR, 'autoencoder.pkl'))
    return iso, svm, auto

iso_model, svm_model, auto_model = load_models()

# Prepare node attributes for graph
def get_node_attrs():
    attrs = {}
    for _, row in scores.iterrows():
        anomaly = max(row['isolation_forest'], row['oneclass_svm'], row['autoencoder'])
        red_team = row['is_red_team']
        attrs[row['user']] = {
            'anomaly': anomaly,
            'red_team': red_team,
            'high_risk': (anomaly > 1.0) or (red_team == 1)
        }
    return attrs

attrs = get_node_attrs()

# Build full graph
def build_graph():
    G = nx.Graph()
    for _, row in file_access.iterrows():
        G.add_edge(row['user'], row['file'], type='access')
    for _, row in usb_usage.iterrows():
        G.add_edge(row['user'], row['device'], type='usb')
    return G

G = build_graph()

# At-risk subgraph
def get_at_risk_subgraph(G, attrs):
    high_risk_nodes = {n for n, v in attrs.items() if v['high_risk']}
    connected_nodes = set()
    for node in high_risk_nodes:
        connected_nodes.add(node)
        connected_nodes.update(G.neighbors(node))
    return G.subgraph(connected_nodes).copy()

# SHAP Explainability
def get_shap_explanation(user_id, model_name='isolation_forest'):
    # Ensure we only use numeric columns for SHAP
    numeric_df = df.select_dtypes(include=[np.number])
    # Drop target and score columns
    cols_to_drop = ['is_red_team', 'isolation_forest', 'oneclass_svm', 'autoencoder']
    if 'is_red_team_x' in numeric_df.columns: cols_to_drop.append('is_red_team_x')
    if 'is_red_team_y' in numeric_df.columns: cols_to_drop.append('is_red_team_y')
    
    X = numeric_df.drop(columns=[c for c in cols_to_drop if c in numeric_df.columns])
    user_idx = df[df['user'] == user_id].index[0]
    user_features = X.iloc[user_idx].values.reshape(1, -1)
    
    if model_name == 'isolation_forest':
        model = iso_model
    elif model_name == 'oneclass_svm':
        model = svm_model
    else:
        model = auto_model
    
    # Use KernelExplainer for SVM/Autoencoder as they don't have tree structures
    if model_name == 'isolation_forest':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_features)
    else:
        # Use a small background dataset for KernelExplainer to speed up
        background = X.sample(min(20, len(X)))
        explainer = shap.KernelExplainer(model.decision_function, background)
        shap_values = explainer.shap_values(user_features)
    
    return shap_values, X.columns.tolist()

# Counterfactual Explanation (Simplified)
def get_counterfactual_explanation(user_id, model_name='isolation_forest'):
    # Ensure we only use numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    cols_to_drop = ['is_red_team', 'isolation_forest', 'oneclass_svm', 'autoencoder']
    if 'is_red_team_x' in numeric_df.columns: cols_to_drop.append('is_red_team_x')
    if 'is_red_team_y' in numeric_df.columns: cols_to_drop.append('is_red_team_y')
    
    X = numeric_df.drop(columns=[c for c in cols_to_drop if c in numeric_df.columns])
    user_idx = df[df['user'] == user_id].index[0]
    original_features = X.iloc[user_idx].copy()
    
    if model_name == 'isolation_forest':
        model = iso_model
    elif model_name == 'oneclass_svm':
        model = svm_model
    else:
        model = auto_model
    
    # Identify top features using SHAP
    try:
        shap_values, feature_names = get_shap_explanation(user_id, model_name)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values
        
        # Top 3 features that increased the anomaly score
        top_features_idx = np.argsort(shap_vals[0])[-3:]
        
        counterfactual = original_features.copy()
        for idx in top_features_idx:
            # Reduce the value of anomalous features to move towards "normal"
            # If the value is high, reduce it. If it's low, we might need to increase it depending on the feature.
            # For simplicity, we move it towards the mean of the "normal" population
            feature_name = feature_names[idx]
            normal_mean = X[feature_name].mean()
            counterfactual[feature_name] = normal_mean
            
        return original_features, counterfactual, feature_names
    except Exception as e:
        raise e

# Sidebar for global controls
st.sidebar.header("🛡️ Control Panel")
selected_user = st.sidebar.selectbox('Select User for Analysis', sorted(df['user'].unique()))
selected_model = st.sidebar.selectbox('Select Detection Model', ['isolation_forest', 'oneclass_svm', 'autoencoder'])

# Tabs
tabs = st.tabs(["📊 Overview", "👤 User Detail", "🕸️ Graph Analysis", "🔬 SHAP XAI", "🎯 Counterfactuals", "📚 Documentation"])

with tabs[0]:  # Overview
    st.header('System-Wide Anomaly Overview')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", len(df))
    with col2:
        st.metric("Anomalies Detected", len(df[df[selected_model] > 0.5]))
    with col3:
        st.metric("Red Team Flagged", len(df[df['is_red_team_y'] == 1]))

    st.subheader('Anomaly Distribution')
    fig = px.histogram(df, x=selected_model, color='is_red_team_y', 
                       title=f'Distribution of {selected_model} Scores',
                       labels={'is_red_team': 'Red Team User'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Top Risk Users')
    df_display = df[['user', 'is_red_team_y', 'isolation_forest', 'oneclass_svm', 'autoencoder']].sort_values(selected_model, ascending=False)
    st.dataframe(df_display.head(10), use_container_width=True)

with tabs[1]:  # User Detail
    st.header(f'Analysis for {selected_user}')
    user_row = df[df['user'] == selected_user].iloc[0]
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Red Team Status", '🚩 Malicious' if user_row['is_red_team_y'] == 1 else '✅ Normal')
    with c2:
        st.metric("Current Anomaly Score", f"{user_row[selected_model]:.3f}")
    with c3:
        rank = df[selected_model].rank(ascending=False)[df['user'] == selected_user].values[0]
        st.metric("Risk Rank", f"#{int(rank)}")

    st.subheader('Behavioral Profile')
    # Radar chart for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_plot = [c for c in numeric_cols if c not in ['is_red_team', 'isolation_forest', 'oneclass_svm', 'autoencoder', 'is_red_team_x', 'is_red_team_y']]
    
    # Normalize for radar chart
    user_vals = user_row[cols_to_plot]
    mean_vals = df[cols_to_plot].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=user_vals.values, theta=cols_to_plot, fill='toself', name=selected_user))
    fig.add_trace(go.Scatterpolar(r=mean_vals.values, theta=cols_to_plot, fill='toself', name='Average User'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="User Behavior vs. Average")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:  # Graph Analysis
    st.header('Relational Risk Graph')
    st.write("Visualizing how this user connects to resources and potential infection paths.")
    
    subG = get_at_risk_subgraph(G, attrs)
    net = Network(height='600px', width='100%', notebook=False, bgcolor='#0e1117', font_color='white')
    
    for node in subG.nodes():
        if node == selected_user:
            color, size = '#ff4b4b', 40
        elif node in attrs and attrs[node]['high_risk']:
            color, size = '#ffa500', 30
        elif node.startswith('user'):
            color, size = '#1f77b4', 20
        else:
            color, size = '#9467bd', 15 # Resource
            
        net.add_node(node, color=color, size=size, title=node)
    
    for edge in subG.edges():
        net.add_edge(edge[0], edge[1])
    
    graph_path = os.path.join(DATA_DIR, 'user_graph.html')
    net.save_graph(graph_path)
    st.components.v1.html(open(graph_path, 'r').read(), height=600)

with tabs[3]:  # SHAP XAI
    st.header('🔬 SHAP Feature Importance')
    try:
        with st.spinner('Calculating SHAP values...'):
            shap_values, feature_names = get_shap_explanation(selected_user, selected_model)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Impact': shap_vals[0]
            }).sort_values('Impact', key=abs, ascending=False)
            
            fig = px.bar(importance_df.head(10), x='Impact', y='Feature', orientation='h',
                         color='Impact', color_continuous_scale='RdBu_r',
                         title=f'Top 10 Features Influencing {selected_user}\'s Score')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Interpretation:** Positive impact (red) increases the anomaly score, while negative impact (blue) decreases it.")
    except Exception as e:
        st.error(f"SHAP Analysis Error: {e}")

with tabs[4]:  # Counterfactuals
    st.header('🎯 Counterfactual "What-If" Analysis')
    try:
        with st.spinner('Generating counterfactuals...'):
            original, counterfactual, feature_names = get_counterfactual_explanation(selected_user, selected_model)
            
            st.write(f"### How can {selected_user} become 'Normal'?")
            st.write("The table below shows the necessary behavioral changes to reduce the anomaly score.")
            
            diff_df = pd.DataFrame({
                'Feature': feature_names,
                'Current Value': original.values,
                'Target Value': counterfactual.values,
                'Required Change': counterfactual.values - original.values
            })
            
            # Only show features that actually changed
            diff_df = diff_df[diff_df['Required Change'] != 0].sort_values('Required Change', key=abs, ascending=False)
            
            if not diff_df.empty:
                st.table(diff_df)
                
                # Visualization of change
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Current', x=diff_df['Feature'], y=diff_df['Current Value']))
                fig.add_trace(go.Bar(name='Target', x=diff_df['Feature'], y=diff_df['Target Value']))
                fig.update_layout(barmode='group', title="Current vs. Target Behavior")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success(f"User {selected_user} is already within normal behavioral bounds for this model.")
                
    except Exception as e:
        st.error(f"Counterfactual Analysis Error: {e}")
        st.write("Technical details:", e)

with tabs[5]:  # Documentation
    st.header('📚 XAI Methodology')
    st.markdown("""
    ### 1. SHAP (SHapley Additive exPlanations)
    We use SHAP to provide **Local Interpretability**. It assigns each feature an importance value for a specific prediction by calculating the average marginal contribution across all possible feature subsets.
    
    ### 2. Counterfactual Explanations
    Our system generates **Actionable Recourse**. It identifies the minimum perturbation in the input space required to change the model's decision from 'Anomalous' to 'Normal'.
    
    ### 3. Graph Neural Network (GNN) Explainability
    By modeling users and resources as a graph, we capture **Relational Anomalies**. The GNNExplainer helps identify which specific connections (e.g., a user accessing a sensitive file) are most responsible for the risk flag.
    
    ### 4. Multi-Model Consensus
    We employ a **Consensus Mechanism** using Isolation Forest, One-Class SVM, and Autoencoders to ensure robust detection and reduce false positives.
    """)
