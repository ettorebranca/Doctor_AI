import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, JumpingKnowledge
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import random

# NOTE: updated imports for explainability (new PyG API)
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer as GNNExplainerAlgo

# --- Load dataset and preprocess ---
df=pd.read_csv('dataset_balanced_filtered.csv')
delay_max = df['hosp_to_icu_1'].max()
dur_max   = df['icu_duration_1'].max()
vital_cols  = [ 'mean_hr','min_hr','max_hr','mean_sysbp','min_sysbp','max_sysbp','mean_rr','min_rr','max_rr','mean_temp','min_temp','max_temp','mean_spo2','min_spo2','max_spo2','mean_glucose','min_glucose','max_glucose' ]
vitals_mean = df[vital_cols].mean()
vitals_std  = df[vital_cols].std().replace(0,1)
vitals_fill = vitals_mean.to_dict()

# --- Stage 1: Transformer-based Clinical State Estimator for ICU Risk ---
class ClinicalStateEstimator(nn.Module):
    def __init__(self,n_categories,cat_emb_dim,attn_heads=8,attn_layers=4,mlp_hidden_dims=[256, 128, 64],latent_dim=32,dropout=0.3):
        super().__init__()
        # Embeddings for categorical features
        self.embs = nn.ModuleList([nn.Embedding(num_cat, cat_emb_dim) for num_cat in n_categories])
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cat_emb_dim))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=cat_emb_dim, nhead=attn_heads, dim_feedforward=cat_emb_dim * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        # MLP after pooling
        layers = []
        dims = [cat_emb_dim] + mlp_hidden_dims
        for in_f, out_f in zip(dims, dims[1:]):
            layers += [nn.Linear(in_f, out_f), nn.LayerNorm(out_f), nn.ReLU(), nn.Dropout(dropout) ]
        self.encoder = nn.Sequential(*layers)
        self.fc_latent = nn.Linear(mlp_hidden_dims[-1], latent_dim)
        # Multi-task heads
        self.heads = nn.ModuleDict({'icu_risk': nn.Linear(latent_dim, 1), 'delay': nn.Linear(latent_dim, 1)})   

    def forward(self, x_cat):
        bsz = x_cat.size(0)
        # Embed categorical features
        cat_toks = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        cat_toks = torch.stack(cat_toks, dim=1)  # [bsz, n_cat, emb_dim]
        # Prepend CLS token
        cls = self.cls_token.expand(bsz, -1, -1)
        tokens = torch.cat([cls, cat_toks], dim=1)
        # Transformer
        attn_out = self.transformer(tokens)      # [bsz, seq_len, emb_dim]
        pooled   = attn_out[:, 0, :]             # [bsz, emb_dim]
        # MLP encoder
        h = self.encoder(pooled)
        z = F.relu(self.fc_latent(h))            # [bsz, latent_dim]
        out = {'latent': z}
        icu_logit      = self.heads['icu_risk'](z).squeeze(-1)
        out['icu_risk'] = icu_logit  # logits for BCEWithLogits
        out['delay']    = self.heads['delay'](z).squeeze(-1)
        return out


# --- Stage 2: Two-Stage GNN for ICU Transfer & Temporal Graphs ---
class TwoStageGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, num_classes, use_icutransfer=True, use_temporal=True, 
                 use_similarity=True, edge_hidden=32, jk_mode='cat', dropedge_rate=0.2):
        super().__init__()
        # flags
        self.use_icutransfer = use_icutransfer
        self.use_temporal    = use_temporal
        self.use_similarity  = use_similarity
        self.dropedge_rate   = dropedge_rate
        # Convs + Norms
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        # Jumping Knowledge
        self.jump = JumpingKnowledge(mode=jk_mode)
        # Edge MLP for dynamic edge weights
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim*2 + 1, edge_hidden), nn.ReLU(), nn.Linear(edge_hidden, 1))
        # Heads (dimension depends on JK mode)
        in_dim = hidden_dim * (3 if jk_mode=='cat' else 1)
        self.heads = nn.ModuleDict({
            'mort':     nn.Linear(in_dim, 1),
            'hours':    nn.Linear(in_dim, 1),
            'disc_loc': nn.Linear(in_dim, num_classes)}) 

    def forward(self, data: GeoData):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # 1) DropEdge augmentation
        if self.training and self.dropedge_rate > 0:
            mask_e = torch.rand(edge_index.size(1), device=edge_index.device) > self.dropedge_rate
            edge_index  = edge_index[:, mask_e]
            edge_weight = edge_weight[mask_e]
        # 2) Dynamic edge weighting
        x_proj = self.input_proj(x)
        h1 = F.relu(self.norm1(self.conv1(x, edge_index, edge_weight)))
        h_init = h1 + x_proj
        if edge_weight is not None:
            src, dst = edge_index
            feat_cat = torch.cat([h_init[src], h_init[dst],edge_weight.unsqueeze(-1)], dim=1)
            edge_weight = torch.sigmoid(self.edge_mlp(feat_cat)).squeeze(-1)
        # 3) Other two convolutional blocks with residual & norm
        h2 = F.relu(self.norm2(self.conv2(h_init, edge_index))) + h_init
        h3 = F.relu(self.norm3(self.conv3(h2, edge_index))) + h2
        # 4) Jumping Knowledge aggregation
        x_jk = self.jump([h_init, h2, h3])
        # 5) Mask out dummy nodes: keep only real patients
        node_mask = data.node_mask.to(x.device)
        readout   = x_jk[node_mask]
        # 6) Multi-task heads on masked readout
        mort  = self.heads['mort'](readout).squeeze(-1)
        hours = self.heads['hours'](readout).squeeze(-1)
        disc  = self.heads['disc_loc'](readout)
        return mort, hours, disc


# --- Graph Construction ---
def build_graph(batch_df, state_preds, k=5, use_similarity=True, use_icutransfer=True, use_temporal=True):
    """Input:
      - batch_df: DataFrame di N righe, contenente le N admissions.
      - state_preds: dict con chiavi 'latent', 'icu_risk', 'delay'. 
    Output: un unico GeoData con N nodi (più un dummy node opzionale), contenente similarity edges, transfer edge e temporal edges. """
    # cat + latent
    cat_cols = [ 'gender','age_group','adm_type','adm_loc','bicarbonate_test','creatinine_test','glucose_test', 'ast_test','bilirubin_test','hematocrit_test']
    tab_cat = torch.tensor(batch_df[cat_cols].values, dtype=torch.long)  # [N, n_cat]
    latent = state_preds['latent']    # [N, latent_dim]
    x_base = torch.cat([tab_cat.float(), latent], dim=1)  # [N, feat_dim]
    # “was in ICU” (logit→prob→binary)
    icu_logits = state_preds['icu_risk'].detach()             # [N]
    icu_probs  = torch.sigmoid(icu_logits)                    # [N]
    #2b) MASK da was_in_icu (calcola la mask ground-truth per chi è realmente andato in ICU)
    was_in_icu  = batch_df['was_in_icu'].fillna(0).astype(int).values
    is_icu_true = torch.tensor(was_in_icu, dtype=torch.float).unsqueeze(1)
    # Vitals (z‐score normalizzato) → applico mask solo ai veri ICU (ground-truth)
    df_vitals = batch_df[vital_cols].fillna(value=vitals_fill)
    vitals_np   = df_vitals.values                              # [N, 18]
    vitals_norm = (vitals_np - vitals_mean.values) / vitals_std.values
    vitals_raw  = torch.tensor(vitals_norm, dtype=torch.float)  # [N,18]
    vitals_masked = vitals_raw * is_icu_true
    # Concateno tutte le features nodali: [cat, latent, is_icu_true, vitals_masked]
    x = torch.cat([x_base, is_icu_true, vitals_masked], dim=1)
    num_nodes = x.size(0)  # = N
    edge_index_list  = []
    edge_weight_list = []

    # Similarity‐edges (K‐NN su feature space) ---
    if use_similarity and num_nodes > 1:
        x_np = x.detach().cpu().numpy()  # [N, D]
        nbrs = NearestNeighbors(n_neighbors=min(k+1, num_nodes)).fit(x_np)
        dist, idx = nbrs.kneighbors(x_np)  # dist e idx shape [N, k+1]
        for i in range(num_nodes):
            for j, d in zip(idx[i][1:], dist[i][1:]):
                edge_index_list.append([i, j])
                edge_weight_list.append(1.0 / (1.0 + d))

    # ICU‐transfer edges (verso dummy node)
    if use_icutransfer:
        icu_node = num_nodes
        x = torch.cat([x, x.new_zeros((1, x.size(1)))], dim=0)  # aggiungo un nodo dummy alla fine
        icu_probs_np = icu_probs.cpu().numpy()  # [N]
        for i, p in enumerate(icu_probs_np):
            edge_index_list.append([i, icu_node])
            edge_weight_list.append(float(p))

    # Temporal‐edges per subject (ordinati per anno) 
    if use_temporal:
        subjects = batch_df['subject_id'].values  # [N]
        years    = batch_df['year'].values        # [N]
        subj_to_nodes = {}
        for idx, s in enumerate(subjects):
            subj_to_nodes.setdefault(s, []).append(idx)
        for nodes in subj_to_nodes.values():
            # ordino gli indici interni secondo l’anno
            nodes_sorted = sorted(nodes, key=lambda i: years[i])
            for u, v in zip(nodes_sorted, nodes_sorted[1:]):
                year1, year2 = years[u], years[v]
                w = 1.0 / (1.0 + abs(year1 - year2))
                edge_index_list.append([u, v])
                edge_weight_list.append(w)

    # Assemblaggio finale di edge_index e edge_weight
    if len(edge_index_list) == 0:
        # No edges case: create an empty edge_index with shape [2,0]
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    data = GeoData(x=x, edge_index=edge_index, edge_weight=edge_weight)
    # Labels
    num_orig = batch_df.shape[0]
    data.y_mort = torch.tensor(batch_df['in_hosp_mortality'].values, dtype=torch.float)
    hours = batch_df['icu_duration_1'].fillna(0).values.astype(float)
    y_hours = hours / dur_max
    data.y_hours = torch.tensor(y_hours, dtype=torch.float)
    data.y_disc = torch.tensor(batch_df['disc_loc'].values, dtype=torch.long)
    data.is_icu_node = torch.tensor(batch_df['was_in_icu'].fillna(0).astype(int).values, dtype=torch.bool)
    data.node_mask   = torch.tensor([True] * num_orig + ([False] if use_icutransfer else []), dtype=torch.bool)
    return data


# --- Dataset ---
class MIMICDataset(Dataset):
    def __init__(self, csv_path=None, df=None):
        if df is not None:
            self.df = df.copy()
        else:
            self.df = pd.read_csv(csv_path)
        self.cat_cols = ['gender', 'age_group' ,'adm_type','adm_loc','bicarbonate_test','creatinine_test','glucose_test', 'ast_test','bilirubin_test','hematocrit_test']
        self.df['icu_duration_1']= self.df['icu_duration_1'].fillna(0.0).astype(float)
        self.df['was_in_icu']    = self.df['was_in_icu'].fillna(0).astype(int)
        self.df['hosp_to_icu_1'] = self.df['hosp_to_icu_1'].fillna(0.0).astype(float)
        self.n_categories = [int(self.df[col].max())+1 for col in ['gender','age_group', 'adm_type','adm_loc','bicarbonate_test','creatinine_test','glucose_test','ast_test','bilirubin_test','hematocrit_test']]

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cat_vals = [int(row[col]) for col in self.cat_cols]
        x_cat = torch.tensor(cat_vals, dtype=torch.long)
        y = {'was_in_icu':    torch.tensor(int(row['was_in_icu']), dtype=torch.float),
            'hosp_to_icu_1': torch.tensor(row['hosp_to_icu_1'] / delay_max, dtype=torch.float)}  
        return x_cat, y, idx
    
def collate_stage1(batch):
    x_cats, ys, idxs = zip(*batch)
    x_cat = torch.stack(x_cats)
    y_batch = {k: torch.stack([y[k] for y in ys]) for k in ys[0]}
    return {'x_cat':x_cat, **y_batch, 'idxs':idxs}


# --- Main Training & Evaluation Pipeline ---
if __name__ == "__main__":
    device = torch.device('cpu')
    # Split into train/val/test for early stopping
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=1)
    train_dataset = MIMICDataset(df=train_df)
    val_dataset   = MIMICDataset(df=val_df)
    test_dataset  = MIMICDataset(df=test_df)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_stage1)
    val_loader    = DataLoader(val_dataset,   batch_size=64, shuffle=False, collate_fn=collate_stage1)
    test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False, collate_fn=collate_stage1)
    estimator = ClinicalStateEstimator(n_categories=train_dataset.n_categories, cat_emb_dim=32, attn_heads=8, attn_layers=4, mlp_hidden_dims=[256,128,64], latent_dim=32, dropout=0.3).to(device)
    optim1 = torch.optim.Adam(estimator.parameters(), lr=1e-4)
    stage1_ckpt = 'estimator_best_wo.pt'
    patience = 5
    best_val_loss = float('inf')
    wait = 0
    if os.path.exists(stage1_ckpt):
        estimator.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        print(f"Modello Stage1 caricato da {stage1_ckpt}")
    else:
        # Stage1 Training
        epochs = 50
        train_losses = []
        for epoch in range(1, epochs+1):
            estimator.train()
            total_loss = 0.0
            for batch in train_loader:
                x_cat = batch['x_cat'].to(device)
                preds  = estimator(x_cat)
                loss = (F.binary_cross_entropy_with_logits(preds['icu_risk'], batch['was_in_icu'].to(device)) +
                        F.mse_loss(preds['delay'], batch['hosp_to_icu_1'].to(device)))
                optim1.zero_grad()
                loss.backward()
                optim1.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")
            # Validation
            estimator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_cat = batch['x_cat'].to(device)
                    preds  = estimator(x_cat)
                    loss = ( F.binary_cross_entropy_with_logits(preds['icu_risk'], batch['was_in_icu'].to(device)) +
                             F.mse_loss(preds['delay'], batch['hosp_to_icu_1'].to(device)))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")
            # Early stopping & save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                wait = 0
                torch.save(estimator.state_dict(), stage1_ckpt)
                print(f"Salvato miglior modello Stage1 (Val Loss={avg_val_loss:.4f})")
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping Stage1 dopo {patience} epoche senza miglioramento.")
                    break
    estimator.load_state_dict(torch.load(stage1_ckpt, map_location=device))
    print("Modello Stage1 ricaricato dal miglior checkpoint")
    # Stage1 Evaluation
    estimator.eval()
    icu_logits, icu_trues = [], []
    delays_pred, delays_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_cat = batch['x_cat'].to(device)
            out   = estimator(x_cat)
            icu_logits.extend(out['icu_risk'].cpu().numpy())
            icu_trues.extend(batch['was_in_icu'].numpy())
            delays_pred.extend(out['delay'].cpu().numpy())
            delays_true.extend(batch['hosp_to_icu_1'].numpy())
    icu_probs = torch.sigmoid(torch.tensor(icu_logits)).numpy()
    icu_pred_flags = icu_probs > 0.5
    icu_true_flags = np.array(icu_trues, dtype=bool)
    delays_pred_orig = np.array(delays_pred) * delay_max
    delays_pred_orig = np.where(icu_pred_flags, delays_pred_orig, 0.0)
    delays_true_orig = np.array(delays_true) * delay_max
    # Avoid crash if no true ICU in test set
    if icu_true_flags.sum() > 0:
        rmse_delay = np.sqrt(mean_squared_error(delays_true_orig[icu_true_flags],delays_pred_orig[icu_true_flags]))
        mae_delay = mean_absolute_error(delays_true_orig[icu_true_flags],delays_pred_orig[icu_true_flags])
    else:
        rmse_delay = float('nan'); mae_delay = float('nan')
    print("--- Stage1 Test Metrics ---")
    try:
        auc1 = roc_auc_score(icu_trues, icu_probs)
    except ValueError:
        auc1 = float('nan')
    print(f"ICU Risk AUC: {auc1:.4f}")
    print(f"ICU Risk Acc: {accuracy_score(icu_trues, icu_pred_flags.astype(int)):.4f}")
    print(f"Delay RMSE (ore, solo veri ICU): {rmse_delay:.2f}")
    print(f"Delay MAE (ore, solo veri ICU): {mae_delay:.2f}")
    n_total       = len(icu_logits)
    n_pred_icu    = icu_pred_flags.sum()
    n_true_icu    = icu_true_flags.sum()
    print(f"Pazienti totali nel test:        {n_total}")
    print(f"Pazienti predetti ICU:           {n_pred_icu}")
    print(f"Pazienti realmente andati in ICU: {n_true_icu}")
    print(f"Pazienti predetti ICU e andati in ICU: {np.logical_and(icu_pred_flags, icu_true_flags).sum()}")
    print(f"Pazienti predetti ICU e NON andati in ICU: {np.logical_and(icu_pred_flags, np.logical_not(icu_true_flags)).sum()}")
    print(f"Pazienti NON predetti ICU e andati in ICU: {np.logical_and(np.logical_not(icu_pred_flags), icu_true_flags).sum()}")
    print(f"Pazienti NON predetti ICU e NON andati in ICU: {np.logical_and(np.logical_not(icu_pred_flags), np.logical_not(icu_true_flags)).sum()}")

    # --- Stage2 Build & Train & Evaluate ---
    # Build graphs per patient using Stage1 estimator predictions
    batch_size_stage2 = 32
    # Creiamo un semplice DataLoader di indici, per iterare sui batch di righe
    train_idx_loader = DataLoader(train_df.index.values, batch_size=batch_size_stage2, shuffle=True)
    val_idx_loader   = DataLoader(val_df.index.values,   batch_size=batch_size_stage2, shuffle=False)
    test_idx_loader  = DataLoader(test_df.index.values,  batch_size=batch_size_stage2, shuffle=False)
    gnn_model = TwoStageGNN(node_feat_dim = len(train_dataset.cat_cols) + 32 + 1 + len(vital_cols), hidden_dim=64, num_classes=4).to(device)
    optim2 = torch.optim.Adam(gnn_model.parameters(), lr=1e-4)
    stage2_ckpt = 'gnn_best_wo_gt.pt'
    patience2 = 5
    best_val_loss2 = float('inf')
    wait2 = 0
    epochs2 = 100
    lambda_time = 1.0
    # Se esiste checkpoint, carico e salto training
    if os.path.exists(stage2_ckpt):
        gnn_model.load_state_dict(torch.load(stage2_ckpt, map_location=device))
        print(f"Modello Stage2 caricato da {stage2_ckpt}, skip training.")
    else:
        for epoch in range(1, epochs2+1):
            gnn_model.train()
            total_loss = 0.0
            # Loop su ogni batch di indici di train_df
            for idx_batch in train_idx_loader:
                idx_batch = idx_batch.numpy()
                batch_df = train_df.loc[idx_batch]      # DataFrame di N righe (nodi)
                # Calcolo predizioni Stage1 per tutto il batch
                x_cats = []
                for i in idx_batch:
                    row = train_df.loc[i]
                    cat_vals = [int(row[col]) for col in train_dataset.cat_cols]
                    x_cats.append(cat_vals)
                x_cats = torch.tensor(x_cats, dtype=torch.long).to(device)  # [N, n_cat]

                with torch.no_grad():
                    stage1_out = estimator(x_cats) # stage1_out['latent'] ha shape [N, latent_dim], etc.
                # build_graph su TUTTE le N righe del batch
                data = build_graph(batch_df, {
                    'latent':    stage1_out['latent'].cpu(),
                    'icu_risk':  stage1_out['icu_risk'].cpu(),
                    'delay':     stage1_out['delay'].cpu(),
                    'icu': batch_df['was_in_icu'].fillna(0).astype(int).values,
                })
                data = data.to(device)
                pred_mort, pred_hours, pred_disc = gnn_model(data)
                mask = data.is_icu_node  # bool mask di lunghezza N
                loss_mort = F.binary_cross_entropy_with_logits(pred_mort, data.y_mort)
                loss_disc = F.cross_entropy(pred_disc, data.y_disc)
                loss_dur = torch.tensor(0.0, device=device)
                if mask.sum() > 0:
                    loss_dur = F.mse_loss(pred_hours[mask], data.y_hours[mask])
                loss = loss_mort + loss_disc + lambda_time * loss_dur
                optim2.zero_grad()
                loss.backward()
                optim2.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_idx_loader)
            print(f"[Stage2] Epoch {epoch} Train Loss: {avg_train_loss:.4f}")
            # Validation
            gnn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx_batch in val_idx_loader:
                    idx_batch = idx_batch.numpy()
                    batch_df = val_df.loc[idx_batch]
                    x_cats = torch.tensor([[int(row[col]) for col in train_dataset.cat_cols] for _, row in batch_df.iterrows()], dtype=torch.long).to(device)
                    stage1_out = estimator(x_cats)
                    data = build_graph(batch_df, {
                        'latent':    stage1_out['latent'].cpu(),
                        'icu_risk':  stage1_out['icu_risk'].cpu(),
                        'icu': batch_df['was_in_icu'].fillna(0).astype(int).values,
                        'delay':     stage1_out['delay'].cpu()})
                    data = data.to(device)
                    pred_mort, pred_hours, pred_disc = gnn_model(data)
                    mask = data.is_icu_node
                    l_mort = F.binary_cross_entropy_with_logits(pred_mort, data.y_mort)
                    l_disc = F.cross_entropy(pred_disc, data.y_disc)
                    l_dur = torch.tensor(0.0, device=device)
                    if mask.sum() > 0:
                        l_dur = F.mse_loss(pred_hours[mask], data.y_hours[mask])
                    val_loss += (l_mort + l_disc + lambda_time * l_dur).item()
            avg_val_loss2 = val_loss / len(val_idx_loader)
            print(f"[Stage2] Epoch {epoch} Val Loss: {avg_val_loss2:.4f}")
            if avg_val_loss2 < best_val_loss2:
                best_val_loss2 = avg_val_loss2
                wait2 = 0
                torch.save(gnn_model.state_dict(), stage2_ckpt)
                print(f"Salvato miglior modello Stage2 (Val Loss={avg_val_loss2:.4f})")
            else:
                wait2 += 1
                if wait2 >= patience2:
                    print(f"Early stopping Stage2 dopo {patience2} epoche senza miglioramento.")
                    break
    # Carico il miglior checkpoint Stage2
    gnn_model.load_state_dict(torch.load(stage2_ckpt, map_location=device))
    print("Modello Stage2 ricaricato dal miglior checkpoint")
    # Stage 2 Evaluation (batch di N nodi per grafo)
    gnn_model.eval()
    all_mort_logits, all_mort_trues = [], []
    all_hours_preds,  all_hours_trues = [], []
    all_disc_preds,   all_disc_trues = [], []
    with torch.no_grad():
        for idx_batch in test_idx_loader:
            idx_batch = idx_batch.numpy()
            batch_df = test_df.loc[idx_batch]
            x_cats = torch.tensor([[int(row[col]) for col in train_dataset.cat_cols] for _, row in batch_df.iterrows()], dtype=torch.long).to(device)
            stage1_out = estimator(x_cats)
            data = build_graph(batch_df, {
                'latent':    stage1_out['latent'].cpu(),
                'icu_risk':  stage1_out['icu_risk'].cpu(),
                'icu': batch_df['was_in_icu'].fillna(0).astype(int).values,
                'delay':     stage1_out['delay'].cpu()})
            data = data.to(device)
            mort_logits, hours_p, disc_p = gnn_model(data)
            mask = data.is_icu_node
            all_mort_logits.append(mort_logits.cpu())
            all_mort_trues.append(data.y_mort.cpu())
            all_hours_preds.append(hours_p.cpu()[mask])
            all_hours_trues.append(data.y_hours.cpu()[mask])
            all_disc_preds.append(disc_p.argmax(dim=1).cpu())
            all_disc_trues.append(data.y_disc.cpu())
    mort_logits_cat = torch.cat(all_mort_logits)
    mort_trues_cat  = torch.cat(all_mort_trues)
    hours_pred_cat  = torch.cat(all_hours_preds) if len(all_hours_preds) > 0 else torch.tensor([])
    hours_true_cat  = torch.cat(all_hours_trues) if len(all_hours_trues) > 0 else torch.tensor([])
    disc_pred_cat   = torch.cat(all_disc_preds)
    disc_true_cat   = torch.cat(all_disc_trues)
    # Da y_hours normalizzato [0,1] a ore reali:
    hours_pred_orig = hours_pred_cat.numpy() * dur_max if hours_pred_cat.numel() > 0 else np.array([])
    hours_pred_orig = np.where(hours_pred_orig>0.0, hours_pred_orig, 0.0)
    hours_true_orig = hours_true_cat.numpy() * dur_max if hours_true_cat.numel() > 0 else np.array([])
    mort_probs = torch.sigmoid(mort_logits_cat).numpy()
    print("--- Stage2 Test Metrics ---")
    try:
        auc_m = roc_auc_score(mort_trues_cat.numpy(), mort_probs)
    except Exception:
        auc_m = float('nan')
    print(f"Mortality Risk AUC: {auc_m:.4f}")
    acc = accuracy_score(mort_trues_cat.numpy(), (mort_probs > 0.5).astype(int))
    print(f"Mortality Risk Acc: {acc:.4f}")
    if hours_true_cat.numel() > 0:
        rmse_icu = np.sqrt(mean_squared_error(hours_true_orig, hours_pred_orig))
        mae_icu  = mean_absolute_error(hours_true_orig, hours_pred_orig)
    else:
        rmse_icu = float('nan'); mae_icu = float('nan')
    print(f"ICU duration Hours RMSE: {rmse_icu:.2f}")
    print(f"ICU Duration MAE: {mae_icu:.2f}")
    print(f"Discharge Loc Acc: {accuracy_score(disc_true_cat.numpy(), disc_pred_cat.numpy()):.4f}")
    print(f"Pazienti predetti morti e real. morti:  {np.logical_and(mort_probs > 0.5, mort_trues_cat.numpy()==1).sum()}")
    print(f"Pazienti predetti morti e non morti:    {np.logical_and(mort_probs > 0.5, mort_trues_cat.numpy()==0).sum()}")
    print(f"Pazienti non predetti morti e real. morti: {np.logical_and(mort_probs <= 0.5, mort_trues_cat.numpy()==1).sum()}")
    print(f"Pazienti non predetti morti e non morti:  {np.logical_and(mort_probs <= 0.5, mort_trues_cat.numpy()==0).sum()}")

#----------------------------------------------------------------------------------------
# --- GNN EXPLAINABILITY con torch_geometric.explain (NUOVA API) ---

# Aggiungo qui una routine di explainability:
# - estrae casualmente un paziente dal test split
# - costruisce un batch piccolo (il paziente + alcuni campioni casuali dal test)
# - calcola predizioni stage1, costruisce il grafo tramite build_graph
# - usa la nuova API Explainer(GNNExplainerAlgo(...)) per spiegare la predizione di mortalità del nodo

class ExplainerWrapper(nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
    def forward(self, x, edge_index, edge_weight=None):
        # costruisce un GeoData minimale e forza node_mask = all True affinché il modello
        # ritorni logits per ogni nodo del grafo (utile per l'explainer).
        data = GeoData(x=x, edge_index=edge_index, edge_weight=edge_weight)
        # se non esiste node_mask, imposta tutti True
        data.node_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        mort, hours, disc = self.gnn(data)
        # mort shape: [N] (per tutti i nodi, dato il node_mask tutti True)
        # restituiamo un output di forma [N, 1] per compatibilità con Explainer/GNNExplainerAlgo
        return mort.unsqueeze(-1)
    
def explain_random_patient(test_df, estimator, gnn_model, train_dataset, n_batch=32, seed=None, k=5):
    """Seleziona un paziente casuale dal test_df e spiega la predizione di mortalità sul grafo costruito
       dal batch contenente il paziente + n_batch-1 campioni casuali dal test_df.
       Restituisce anche la figura matplotlib dell'explanation (feature importance + subgraph)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if len(test_df) == 0:
        print("Test set vuoto: impossibile eseguire explainability.")
        return None

    # 1) seleziona paziente casuale (index reale nel dataframe)
    chosen_idx = random.choice(list(test_df.index))
    # Costruiamo un piccolo batch contenente il paziente + alcuni altri campioni casuali
    other_idxs = list(test_df.index.drop(chosen_idx))
    n_others = max(0, min(n_batch-1, len(other_idxs)))
    sampled_others = random.sample(other_idxs, n_others) if n_others > 0 else []
    batch_idxs = [chosen_idx] + sampled_others
    batch_df = test_df.loc[batch_idxs].reset_index(drop=True)
    # 2) Calcola predizioni Stage1 per il batch
    x_cats = torch.tensor([[int(row[col]) for col in train_dataset.cat_cols] for _, row in batch_df.iterrows()], dtype=torch.long).to(device)
    with torch.no_grad():
        stage1_out = estimator(x_cats)
    # Convert to cpu tensors expected by build_graph (it expects latent on cpu)
    state_preds = {
        'latent': stage1_out['latent'].cpu(),
        'icu_risk': stage1_out['icu_risk'].cpu(),
        'delay': stage1_out['delay'].cpu(),
        'icu': batch_df['was_in_icu'].fillna(0).astype(int).values
    }
    # 3) costruisci grafo
    data = build_graph(batch_df, state_preds, k=k, use_similarity=True, use_icutransfer=True, use_temporal=True)
    # Assicuriamoci che edge_index e edge_weight abbiano la forma corretta per l'explainer
    x = data.x.clone().to(device)
    edge_index = data.edge_index.clone().to(device)
    edge_weight = data.edge_weight.clone().to(device) if data.edge_weight is not None else None
    # 4) wrapper per l'explainer (usa la NUOVA API Explainer + GNNExplainerAlgo)
    wrapper = ExplainerWrapper(gnn_model).to(device)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainerAlgo(epochs=200, lr=0.01),
        explanation_type='model',
        node_mask_type='attributes',   # vogliamo importance sulle feature del nodo
        edge_mask_type='object',       # vogliamo importance sugli edge
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='raw'   # wrapper restituisce logits grezzi
        ),)
    node_idx = 15
    # Run the explainer (safe try/except per non interrompere lo script)
    try:
        # La nuova API permette chiamare explainer come una funzione
        explanation = explainer(x, edge_index, edge_weight=edge_weight, index=node_idx)
    except Exception as e:
        print("Errore durante l'explainability con la nuova API Explainer:", e)
        return None
    # Recupera masks (edge_mask e node_mask possono essere None o avere shape diverse a seconda dell'algoritmo)
    edge_mask = getattr(explanation, 'edge_mask', None)
    node_mask = getattr(explanation, 'node_mask', None)
    # Normalizziamo il node_feat_mask per il solo nodo che stiamo spiegando
    if node_mask is None:
        node_feat_mask = torch.zeros(x.size(1), device=x.device).cpu()
    else:
        # node_mask può essere [N, F] oppure [F]
        if node_mask.dim() == 2:
            node_feat_mask = node_mask[node_idx].detach().cpu()
        elif node_mask.dim() == 1:
            node_feat_mask = node_mask.detach().cpu()
        else:
            node_feat_mask = node_mask.view(-1).detach().cpu()
    edge_mask_cpu = edge_mask.detach().cpu() if edge_mask is not None else None
    # 5) Recupera predizioni del modello sul grafo per mostrare i valori
    wrapper.eval()
    with torch.no_grad():
        mort_logits_all = wrapper(x, edge_index, edge_weight).squeeze(-1).cpu().numpy()  # [N]
    mort_probs_all = torch.sigmoid(torch.tensor(mort_logits_all)).numpy()
    chosen_mort_prob = mort_probs_all[node_idx]
    data_for_model = data.to(device)
    gnn_model.eval()
    # 6) Visualizzazioni: feature importance e subgraph edge importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    latent_dim = state_preds['latent'].shape[1] if 'latent' in state_preds else 32
    feat_names = []
    for c in train_dataset.cat_cols:
        feat_names.append(f"cat_{c}")
    for j in range(latent_dim):
        feat_names.append(f"latent_{j}")
    feat_names.append("is_icu_true")
    for v in vital_cols:
        feat_names.append(v)
    feat_mask = node_feat_mask.numpy()
    topk = min(20, feat_mask.shape[0])
    idxs_top = np.argsort(-np.abs(feat_mask))[:topk]
    axes[0].barh(np.arange(topk)[::-1], feat_mask[idxs_top][::-1])
    axes[0].set_yticks(np.arange(topk))
    axes[0].set_yticklabels([feat_names[i] if i < len(feat_names) else f"f{i}" for i in idxs_top[::-1]])
    axes[0].set_title(f"Feature importance (top {topk}) per nodo {node_idx}\nPred Mort prob: {chosen_mort_prob:.3f}")
    axes[0].set_xlabel("Importance (signed)")
    # Edge importance: visualizziamo l'importanza sugli edge che coinvolgono il nodo scelto
    if edge_mask_cpu is not None and edge_index.numel() > 0:
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        edge_mask_np = edge_mask_cpu.numpy()
        incident_idx = np.where((src == node_idx) | (dst == node_idx))[0]
        if len(incident_idx) > 0:
            vals = edge_mask_np[incident_idx]
            labels = [f"{int(src[i])}-{int(dst[i])}" for i in incident_idx]
            axes[1].barh(np.arange(len(vals))[::-1], vals[::-1])
            axes[1].set_yticks(np.arange(len(vals)))
            axes[1].set_yticklabels([labels[i] for i in range(len(labels))[::-1]])
            axes[1].set_title(f"Edge importance incidenti a nodo {node_idx}")
            axes[1].set_xlabel("Edge importance")
        else:
            axes[1].text(0.1, 0.5, "Nessun edge incidente al nodo scelto", fontsize=12)
            axes[1].axis('off')
    else:
        axes[1].text(0.1, 0.5, "Edge importance non disponibile o grafo senza archi", fontsize=12)
        axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    # Stampiamo un breve report testuale
    print("\n--- Explainability report ---")
    print(f"Paziente testato: DataFrame index originale = {chosen_idx}")
    print(f"Numero nodi nel batch (incluso dummy se presente): {x.size(0)}")
    print(f"Indice nodo nel grafo usato per explain: {node_idx}")
    print(f"Probabilità mortalità (wrapper/explainer): {chosen_mort_prob:.4f}")
    # Ritorniamo alcuni oggetti utili per ulteriori analisi
    return {
        'chosen_idx': chosen_idx,
        'batch_df': batch_df,
        'data': data,
        'node_idx': node_idx,
        'node_feat_mask': node_feat_mask.detach().cpu(),
        'edge_mask': edge_mask_cpu.detach().cpu() if edge_mask_cpu is not None else None,
        'mort_probs_all': mort_probs_all,
        'fig': fig}
try:
    explain_res = explain_random_patient(test_df, estimator, gnn_model, train_dataset, n_batch=32, seed=42, k=5)
    if explain_res is None:
        print("Explainability non completata (res == None).")
    else:
        print("Explainability completata — vedi figura e report sopra.")
except Exception as e:
    print("Errore nell'esecuzione della routine di explainability:", e)