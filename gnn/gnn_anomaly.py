import pandas as pd
import networkx as nx
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np

DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load logs
def load_logs():
    file_access = pd.read_csv(os.path.join(DATA_DIR, 'file_access.csv'), parse_dates=['access_time'])
    usb_usage = pd.read_csv(os.path.join(DATA_DIR, 'usb_usage.csv'), parse_dates=['plug_time', 'unplug_time'])
    return file_access, usb_usage

# Build graph and prepare for GNN
def prepare_gnn_data():
    file_access, usb_usage = load_logs()
    
    # Create a bipartite graph: users <-> files, users <-> devices
    G = nx.Graph()
    
    users = sorted(file_access['user'].unique())
    files = sorted(file_access['file'].unique())
    devices = sorted(usb_usage['device'].unique())
    
    # Node mapping
    all_nodes = users + files + devices
    node_map = {node: i for i, node in enumerate(all_nodes)}
    num_users = len(users)
    
    # Edges
    edge_index = []
    for _, row in file_access.iterrows():
        u, f = node_map[row['user']], node_map[row['file']]
        edge_index.append([u, f])
        edge_index.append([f, u])
        
    for _, row in usb_usage.iterrows():
        u, d = node_map[row['user']], node_map[row['device']]
        edge_index.append([u, d])
        edge_index.append([d, u])
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Node features (identity matrix)
    x = torch.eye(len(node_map))
    
    # Labels (1 for red team users, 0 otherwise)
    red_team_users = []
    if os.path.exists(os.path.join(DATA_DIR, 'red_team_users.csv')):
        red_team_users = pd.read_csv(os.path.join(DATA_DIR, 'red_team_users.csv'))['user'].tolist()
    
    y = torch.zeros(len(node_map), dtype=torch.long)
    for user in red_team_users:
        if user in node_map:
            y[node_map[user]] = 1
            
    # Train mask (only users)
    train_mask = torch.zeros(len(node_map), dtype=torch.bool)
    train_mask[:num_users] = True
    
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
    return data, node_map, users, all_nodes

# Simple GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_gnn():
    data, node_map, users, all_nodes = prepare_gnn_data()
    model = GCN(data.num_node_features, 16, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'gnn_model.pth'))
    print("GNN model trained and saved.")
    return model, data, node_map, all_nodes

def explain_node(node_idx, model, data):
    # Enhanced GNNExplainer with better configuration
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    
    # Extract top contributing edges
    edge_mask = explanation.edge_mask
    top_edges_idx = torch.where(edge_mask > 0.1)[0]
    top_edges = data.edge_index[:, top_edges_idx]
    
    return explanation, top_edges

if __name__ == "__main__":
    model, data, node_map, all_nodes = train_gnn()
    
    # Save graph features for other models
    G = nx.Graph()
    edge_index_np = data.edge_index.numpy()
    for i in range(edge_index_np.shape[1]):
        u, v = edge_index_np[0, i], edge_index_np[1, i]
        G.add_edge(all_nodes[u], all_nodes[v])
    
    degrees = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    features = []
    for user in [n for n in all_nodes if n.startswith('user')]:
        features.append({
            'user': user,
            'degree_centrality': degrees[user],
            'betweenness_centrality': betweenness[user]
        })
    pd.DataFrame(features).to_csv(os.path.join(DATA_DIR, 'graph_features.csv'), index=False)
    print('Graph features saved to data/graph_features.csv')
