import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GATv2Conv, SAGEConv, JumpingKnowledge, RGCNConv  ####!!!!!!!
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- Load dataset and preprocess ---
df = pd.read_csv('dataset_balanced_filtered.csv')
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


# --- Stage 2: Two-Stage GNN with R-GCN ---
class TwoStageGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, num_classes, num_relations=3,
                 use_icutransfer=True, use_temporal=True, use_similarity=True, edge_hidden=32, jk_mode='cat', dropedge_rate=0.2):
        super().__init__()
        # flags
        self.use_icutransfer = use_icutransfer
        self.use_temporal    = use_temporal
        self.use_similarity  = use_similarity
        self.dropedge_rate   = dropedge_rate

        # R-GCN conv as first layer (uses edge_type)
        # num_bases to reduce params; set None for full paramization
        self.conv1 = RGCNConv(in_channels=node_feat_dim, out_channels=hidden_dim, num_relations=num_relations, num_bases=4)
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # subsequent layers
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        # Jumping Knowledge
        self.jump = JumpingKnowledge(mode=jk_mode)
        # Edge MLP for dynamic edge weights (optional, used for weighting after conv)
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim*2 + 1, edge_hidden), nn.ReLU(), nn.Linear(edge_hidden, 1))
        # Heads
        in_dim = hidden_dim * (3 if jk_mode=='cat' else 1)
        self.heads = nn.ModuleDict({
            'mort':     nn.Linear(in_dim, 1),
            'hours':    nn.Linear(in_dim, 1),
            'disc_loc': nn.Linear(in_dim, num_classes)}) 

    def forward(self, data: GeoData):
        x = data.x
        edge_index = getattr(data, 'edge_index', None)
        edge_weight = getattr(data, 'edge_weight', None)
        edge_type = getattr(data, 'edge_type', None)

        # 1) DropEdge augmentation (filter edge_index, edge_weight, edge_type consistently)
        if self.training and self.dropedge_rate > 0 and edge_index is not None:
            mask_e = torch.rand(edge_index.size(1), device=edge_index.device) > self.dropedge_rate
            edge_index  = edge_index[:, mask_e]
            if edge_weight is not None:
                edge_weight = edge_weight[mask_e]
            if edge_type is not None:
                edge_type = edge_type[mask_e]

        # 2) R-GCN convolution (uses edge_type)
        # RGCNConv signature: (x, edge_index, edge_type)
        if edge_index is None or edge_index.size(1) == 0:
            # if no edges, apply input_proj only
            h1 = F.relu(self.norm1(self.input_proj(x)))
        else:
            h1 = F.relu(self.norm1(self.conv1(x, edge_index, edge_type)))
        x_proj = self.input_proj(x)
        h_init = h1 + x_proj

        # Optional: re-compute dynamic edge weights using learned node features (still works w/o)
        if edge_index is not None and getattr(data, 'edge_weight', None) is not None:
            # simple example: recompute weights from features (for later use if desired)
            src, dst = edge_index
            feat_cat = torch.cat([h_init[src], h_init[dst], edge_weight.unsqueeze(-1)], dim=1)
            # new_edge_weight = torch.sigmoid(self.edge_mlp(feat_cat)).squeeze(-1)
            # We do not feed new_edge_weight to RGCNConv (it doesn't accept weights), but could be used later.
            pass

        # 3) Further conv blocks (GATv2 / SAGE)
        if edge_index is not None and edge_index.size(1) > 0:
            h2 = F.relu(self.norm2(self.conv2(h_init, edge_index))) + h_init
            h3 = F.relu(self.norm3(self.conv3(h2, edge_index))) + h2
        else:
            # no edges -> identity residuals
            h2 = h_init
            h3 = h_init

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


# --- Graph Construction (returns edge_type) ---
def build_graph(batch_df, state_preds, k=5, use_similarity=True, use_icutransfer=True, use_temporal=True):
    """Input:
      - batch_df: DataFrame di N righe, contenente le N admissions.
      - state_preds: dict con chiavi 'latent', 'icu_risk', 'delay'.
    Output: un unico GeoData con N nodi (più un dummy node opzionale), contenente similarity edges, transfer edge e temporal edges,
            e con edge_type: 0=similarity,1=icutransfer,2=temporal
    """
    # cat + latent
    cat_cols = [ 'gender','age_group','adm_type','adm_loc','bicarbonate_test','creatinine_test','glucose_test', 'ast_test','bilirubin_test','hematocrit_test']
    tab_cat = torch.tensor(batch_df[cat_cols].values, dtype=torch.long)  # [N, n_cat]
    latent = state_preds['latent']    # [N, latent_dim]
    x_base = torch.cat([tab_cat.float(), latent], dim=1)  # [N, feat_dim]
    # “was in ICU” (logit→prob→binary)
    icu_logits = state_preds['icu_risk'].detach()             # [N]
    icu_probs  = torch.sigmoid(icu_logits)                    # [N]
    was_in_icu  = batch_df['was_in_icu'].fillna(0).astype(int).values
    is_icu_true = torch.tensor(was_in_icu, dtype=torch.float).unsqueeze(1)
    # Vitals (z‐score normalizzato)
    df_vitals = batch_df[vital_cols].fillna(value=vitals_fill)
    vitals_np   = df_vitals.values                              # [N, 18]
    vitals_norm = (vitals_np - vitals_mean.values) / vitals_std.values
    vitals_raw  = torch.tensor(vitals_norm, dtype=torch.float)  # [N,18]
    vitals_masked = vitals_raw * is_icu_true
    # Concatenazione features nodali
    x = torch.cat([x_base, is_icu_true, vitals_masked], dim=1)
    num_nodes = x.size(0)  # = N
    edge_index_list  = []
    edge_weight_list = []
    edge_type_list   = []

    # Similarity‐edges (K‐NN su feature space) --- relation 0
    if use_similarity and num_nodes > 1:
        x_np = x.detach().cpu().numpy()  # [N, D]
        nbrs = NearestNeighbors(n_neighbors=min(k+1, num_nodes)).fit(x_np)
        dist, idx = nbrs.kneighbors(x_np)  # dist e idx shape [N, k+1]
        for i in range(num_nodes):
            for j, d in zip(idx[i][1:], dist[i][1:]):
                edge_index_list.append([i, j])
                edge_weight_list.append(1.0 / (1.0 + d))
                edge_type_list.append(0)

    # ICU‐transfer edges (relation 1) -> verso dummy node
    if use_icutransfer:
        icu_node = num_nodes
        x = torch.cat([x, x.new_zeros((1, x.size(1)))], dim=0)  # aggiungo un nodo dummy alla fine
        icu_probs_np = icu_probs.cpu().numpy()  # [N]
        for i, p in enumerate(icu_probs_np):
            edge_index_list.append([i, icu_node])
            edge_weight_list.append(float(p))
            edge_type_list.append(1)

    # Temporal‐edges per subject (relation 2)
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
                edge_type_list.append(2)

    # Assemblaggio finale di edge_index, edge_weight, edge_type
    if len(edge_index_list) == 0:
        # fallback: self-loop sul primo nodo per evitare errori downstream
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0], dtype=torch.float)
        edge_type = torch.tensor([0], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)

    data = GeoData(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_type=edge_type)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split into train/val/test for early stopping
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=1)
    train_dataset = MIMICDataset(df=train_df)
    val_dataset   = MIMICDataset(df=val_df)
    test_dataset  = MIMICDataset(df=test_df)
    train_loader  = TorchDataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_stage1)
    val_loader    = TorchDataLoader(val_dataset,   batch_size=64, shuffle=False, collate_fn=collate_stage1)
    test_loader   = TorchDataLoader(test_dataset,  batch_size=64, shuffle=False, collate_fn=collate_stage1)

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
    # compute metrics only where there is true ICU
    if icu_true_flags.sum() > 0:
        rmse_delay = np.sqrt(mean_squared_error(delays_true_orig[icu_true_flags],delays_pred_orig[icu_true_flags]))
        mae_delay = mean_absolute_error(delays_true_orig[icu_true_flags],delays_pred_orig[icu_true_flags])
    else:
        rmse_delay = float('nan')
        mae_delay = float('nan')
    print("--- Stage1 Test Metrics ---")
    print(f"ICU Risk AUC: {roc_auc_score(icu_trues, icu_probs):.4f}")
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
    batch_size_stage2 = 32
    # usiamo DataLoader di indici (torch) per iterare sugli indici di dataframe
    train_idx_loader = TorchDataLoader(train_df.index.values, batch_size=batch_size_stage2, shuffle=True)
    val_idx_loader   = TorchDataLoader(val_df.index.values,   batch_size=batch_size_stage2, shuffle=False)
    test_idx_loader  = TorchDataLoader(test_df.index.values,  batch_size=batch_size_stage2, shuffle=False)
    # Calcolo node_feat_dim coerente con costruzione x in build_graph:
    node_feat_dim = len(train_dataset.cat_cols) + 32 + 1 + len(vital_cols)  # cat (as floats), latent_dim=32, is_icu_true(1), vitals(18)
    gnn_model = TwoStageGNN(node_feat_dim = node_feat_dim, hidden_dim=64, num_classes=4, num_relations=3).to(device)
    optim2 = torch.optim.Adam(gnn_model.parameters(), lr=1e-4)
    stage2_ckpt = 'gnn_best_wo_gt_R.pt'
    patience2 = 10
    best_val_loss2 = float('inf')
    wait2 = 0
    epochs2 = 500
    lambda_time = 1.0
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
    if len(all_hours_preds) > 0:
        hours_pred_cat  = torch.cat(all_hours_preds)
        hours_true_cat  = torch.cat(all_hours_trues)
    else:
        hours_pred_cat = torch.tensor([], dtype=torch.float)
        hours_true_cat = torch.tensor([], dtype=torch.float)
    disc_pred_cat   = torch.cat(all_disc_preds)
    disc_true_cat   = torch.cat(all_disc_trues)

    # Da y_hours normalizzato [0,1] a ore reali:
    hours_pred_orig = hours_pred_cat.numpy() * dur_max if hours_pred_cat.numel() > 0 else np.array([])
    hours_pred_orig = np.where(hours_pred_orig>0.0, hours_pred_orig, 0.0)
    hours_true_orig = hours_true_cat.numpy() * dur_max if hours_true_cat.numel() > 0 else np.array([])
    mort_probs = torch.sigmoid(mort_logits_cat).numpy()
    print("--- Stage2 Test Metrics ---")
    print(f"Mortality Risk AUC: {roc_auc_score(mort_trues_cat.numpy(), mort_probs):.4f}")
    acc = accuracy_score(mort_trues_cat.numpy(), (mort_probs > 0.5).astype(int))
    print(f"Mortality Risk Acc: {acc:.4f}")
    if hours_true_orig.size > 0:
        rmse_icu = np.sqrt(mean_squared_error(hours_true_orig, hours_pred_orig))
        mae_icu  = mean_absolute_error(hours_true_orig, hours_pred_orig)
    else:
        rmse_icu = float('nan'); mae_icu = float('nan')
    print(f"ICU duration Hours RMSE: {rmse_icu:.2f}")
    print(f"ICU Duration MAE: {mae_icu:.2f}")
    print(f"Discharge Loc Acc: {accuracy_score(disc_true_cat.numpy(), disc_pred_cat.numpy()):.4f}")
    # Alcuni conteggi di True/False positives/negatives su mortalità
    print(f"Pazienti predetti morti e real. morti:  {np.logical_and(mort_probs > 0.5, mort_trues_cat.numpy()==1).sum()}")
    print(f"Pazienti predetti morti e non morti:    {np.logical_and(mort_probs > 0.5, mort_trues_cat.numpy()==0).sum()}")
    print(f"Pazienti non predetti morti e real. morti: {np.logical_and(mort_probs <= 0.5, mort_trues_cat.numpy()==1).sum()}")
    print(f"Pazienti non predetti morti e non morti:  {np.logical_and(mort_probs <= 0.5, mort_trues_cat.numpy()==0).sum()}")

    # Errori ICU duration in tolleranze
    if hours_true_orig.size > 0:
        print(f"Errore ≤ 6h:   {np.sum(np.abs(hours_pred_orig - hours_true_orig) <= 6)}")
        print(f"Errore ≤ 12h:  {np.sum(np.abs(hours_pred_orig - hours_true_orig) <= 12)}")
        print(f"Errore ≤ 24h:  {np.sum(np.abs(hours_pred_orig - hours_true_orig) <= 24)}")
        print(f"Errore ≤ 36h:  {np.sum(np.abs(hours_pred_orig - hours_true_orig) <= 36)}")
        print(f"Errore ≤ 48h:  {np.sum(np.abs(hours_pred_orig - hours_true_orig) <= 48)}")
        print(f"Errore ≤ 72h:  {np.sum(np.abs(hours_pred_orig - hours_true_orig) <= 72)}")
        print(f"Errore ≤ 96h:  {np.sum(np.abs(hours_pred_orig - hours_true_orig) <= 96)}")
        print(f"Errore > 96h: {np.sum(np.abs(hours_pred_orig - hours_true_orig) > 96)}")
        print(f"Errore ≥ 120h: {np.sum(np.abs(hours_pred_orig - hours_true_orig) >= 120)}")
    else:
        print("Nessuna predizione di ICU duration disponibile nel test set.")

    print("\nConfusion Matrix per Discharge Location:")
    cm_disc = confusion_matrix(disc_true_cat.numpy(), disc_pred_cat.numpy())
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_disc, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['mort', 'casa', 'other', 'transf'], 
                yticklabels=['mort', 'casa', 'other', 'transf'])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix - Discharge Location')
    plt.show()

    # AUC per ICU duration thresholds (solo veri ICU)
    if hours_true_orig.size > 0:
        mask = hours_true_orig > 0
        y_true = hours_true_orig[mask]
        y_pred = hours_pred_orig[mask]
        print(f"Numero di pazienti ICU valutati: {len(y_true)}")
        y_bin_3 = (y_true > 72).astype(int)
        if len(np.unique(y_bin_3)) > 1:
            auc_3 = roc_auc_score(y_bin_3, y_pred)
            print(f"AUC-ROC per predire ICU duration >3 giorni: {auc_3:.4f}")
        else:
            print("Variabilità insufficiente su soglia 3 giorni: impossibile calcolare AUC-ROC")
        y_bin_7 = (y_true > 168).astype(int)
        if len(np.unique(y_bin_7)) > 1:
            auc_7 = roc_auc_score(y_bin_7, y_pred)
            print(f"AUC-ROC per predire ICU duration >7 giorni: {auc_7:.4f}")
        else:
            print("Variabilità insufficiente su soglia 7 giorni: impossibile calcolare AUC-ROC")
    else:
        print("Nessuna predizione di ICU duration disponibile nel test set.")