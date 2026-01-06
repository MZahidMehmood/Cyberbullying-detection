import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # AXW
        x = self.linear(x)
        return torch.spmm(adj, x)

class GCN(nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(n_feat, n_hidden)
        self.gc2 = GCNLayer(n_hidden, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

def build_graph(embeddings, k=10):
    """
    Constructs a k-NN graph from embeddings.
    Returns normalized adjacency matrix (Renormalization trick).
    """
    print(f"Building k-NN graph with k={k}...")
    # k-NN graph
    A = kneighbors_graph(embeddings, k, mode='connectivity', include_self=True)
    
    # Symmeterize
    A = A + A.T
    A.data = np.ones_like(A.data) # Binarize
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    from scipy.sparse import coo_matrix
    
    # Add self-loops (already included via include_self=True in sk-learn 0.22+, but let's encourage it)
    # Actually kneighbors_graph might not guarantee symmetry if we just do A+A.T for directed.
    # Standard GCN: A_hat = A + I
    
    row_sum = np.array(A.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    # Sparse diagonal D_inv
    from scipy.sparse import diags
    D_hat = diags(d_inv_sqrt)
    
    # A_norm = D * A * D
    A_norm = D_hat.dot(A).dot(D_hat)
    
    # Convert to PyTorch sparse tensor
    A_coo = A_norm.tocoo()
    values = A_coo.data
    indices = np.vstack((A_coo.row, A_coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A_coo.shape
    
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def run_sosnet_baseline(data_dir='data/splits', output_dir='results/baselines'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data for GCN...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train_full.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Concat for transductive setting
    # Important: Preserve order to map back to train/test masks
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    train_idx = range(0, len(train_df))
    test_idx = range(len(train_df), len(full_df))
    
    print(f"Total nodes: {len(full_df)} (Train: {len(train_df)}, Test: {len(test_df)})")
    
    # 2. Embeddings (Features)
    model_st = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    texts = full_df['cleaned_text'].fillna('').tolist()
    embeddings = model_st.encode(texts, show_progress_bar=True, batch_size=32)
    features = torch.FloatTensor(embeddings).to(device)
    
    # 3. Graph Construction
    adj = build_graph(embeddings, k=10).to(device)
    
    # 4. Labels
    le = LabelEncoder()
    labels = le.fit_transform(full_df['cyberbullying_type'])
    labels = torch.LongTensor(labels).to(device)
    
    num_classes = len(le.classes_)
    
    # 5. Model
    model = GCN(n_feat=features.shape[1], n_hidden=64, n_class=num_classes, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # Standard GCN/GAT settings
    
    # 6. Training Loop (Transductive)
    print("Training GCN...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.nll_loss(output[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
    # 7. Evaluation
    print("Evaluating GCN...")
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        # Test set only
        test_output = output[test_idx]
        test_labels = labels[test_idx]
        
        preds = test_output.argmax(dim=1).cpu().numpy()
        true = test_labels.cpu().numpy()
        probs = torch.exp(test_output).cpu().numpy() # Softmax probs
        
    # Metrics
    macro_f1 = f1_score(true, preds, average='macro')
    mcc = matthews_corrcoef(true, preds)
    try:
        auprc = average_precision_score(pd.get_dummies(true), probs, average='macro')
    except:
        auprc = 0.0
        
    print(f"SOSNet (GCN) Results: Macro-F1: {macro_f1:.4f} | MCC: {mcc:.4f} | AUPRC: {auprc:.4f}")
    
    results = {
        'Model': 'SOSNet_GCN',
        'Macro_F1': macro_f1,
        'MCC': mcc,
        'AUPRC': auprc
    }
    
    # Save Report
    report = classification_report(true, preds, target_names=le.classes_, output_dict=True)
    with open(os.path.join(output_dir, 'SOSNet_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
        
    # Append to summary if exists, else create
    summary_path = os.path.join(output_dir, 'baseline_summary.csv')
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        # Remove old SOSNet if exists
        summary_df = summary_df[summary_df['Model'] != 'SOSNet_GCN']
        summary_df = pd.concat([summary_df, pd.DataFrame([results])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([results])
        
    summary_df.to_csv(summary_path, index=False)
    print("SOSNet evaluation complete.")

if __name__ == "__main__":
    run_sosnet_baseline()
