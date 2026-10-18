"""PGExplainer (Thiện phụ trách).
Mạng neural MLP dự đoán cạnh quan trọng theo cấu trúc chuẩn canonical (Luo et al., NeurIPS 2020).
MLP nhận concatenated node embeddings [z_u || z_v || z_target] và tối ưu hóa Mutual Information
với Size và Entropy regularization trên các đồ thị con 2-hop của tập train_mask (BG03).
"""

import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sna.core.types import ExplainerResult, CandidateEdgesResult
from sna.explainers.base import BaseExplainer
from sna.data.candidate_edges import extract_candidate_edges


class PGExplainerMLP(nn.Module):
    """Mạng MLP 2 lớp tham số hóa xác suất tồn tại của cạnh (Luo et al., 2020).

    Đầu vào là vector ghép nối node embeddings: [z_u || z_v || z_target] có kích thước emb_dim * 3.
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        emb_u: torch.Tensor,
        emb_v: Optional[torch.Tensor] = None,
        emb_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if emb_v is None and emb_target is None:
            # Nhận trực tiếp tensor đã ghép nối sẵn [z_u || z_v || z_target]
            if emb_u.dim() == 1:
                emb_u = emb_u.unsqueeze(0)
            return self.mlp(emb_u).view(-1)

        # Chuẩn hóa mọi tensor về 2D (batch_size, emb_dim)
        if emb_u.dim() == 1:
            emb_u = emb_u.unsqueeze(0)
        if emb_v is not None and emb_v.dim() == 1:
            emb_v = emb_v.unsqueeze(0)
        if emb_target is not None and emb_target.dim() == 1:
            emb_target = emb_target.unsqueeze(0)

        b_u = emb_u.size(0)
        b_v = emb_v.size(0) if emb_v is not None else b_u
        b_t = emb_target.size(0) if emb_target is not None else b_u

        if b_u == 0 or b_v == 0:
            batch_size = 0
        else:
            batch_size = max(b_u, b_v, b_t)

        if emb_u.size(0) != batch_size:
            emb_u = emb_u.expand(batch_size, -1)
        if emb_v is not None and emb_v.size(0) != batch_size:
            emb_v = emb_v.expand(batch_size, -1)

        if emb_target is None:
            # Fallback nếu chỉ truyền 2 vector (zero-pad target để tránh lỗi kích thước ma trận)
            emb_target = torch.zeros(batch_size, emb_u.size(-1), device=emb_u.device, dtype=emb_u.dtype)
        elif emb_target.size(0) != batch_size:
            emb_target = emb_target.expand(batch_size, -1)

        x = torch.cat([emb_u, emb_v, emb_target], dim=-1)
        return self.mlp(x).view(-1)


class PGExplainerModel:
    """Mô hình quản lý huấn luyện PGExplainer trên tập huấn luyện train_mask."""
    def __init__(
        self,
        model: nn.Module,
        emb_dim: Optional[int] = None,
        hidden_dim: int = 64,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        if emb_dim is None:
            if hasattr(model, 'c1') and hasattr(model.c1, 'out_channels'):
                emb_dim = model.c1.out_channels
            elif hasattr(model, 'conv1') and hasattr(model.conv1, 'out_channels'):
                emb_dim = model.conv1.out_channels
            elif hasattr(model, 'hidden_dim'):
                emb_dim = model.hidden_dim
            elif hasattr(model, 'gat1') and hasattr(model.gat1, 'out_channels'):
                emb_dim = model.gat1.out_channels * getattr(model.gat1, 'heads', 1)
            else:
                emb_dim = 64
        self.emb_dim = emb_dim
        self.mlp = PGExplainerMLP(emb_dim=emb_dim, hidden_dim=hidden_dim).to(self.device)
        self.is_trained = False

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Trích xuất node representations từ tầng ẩn c1 của GCN."""
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'c1'):
                z = self.model.c1(x.to(self.device), edge_index.to(self.device))
                z = F.relu(z)
            elif hasattr(self.model, 'conv1'):
                z = self.model.conv1(x.to(self.device), edge_index.to(self.device))
                z = F.relu(z)
            elif hasattr(self.model, 'gat1'):
                z = self.model.gat1(x.to(self.device), edge_index.to(self.device))
                z = F.elu(z)
            else:
                z = self.model(x.to(self.device), edge_index.to(self.device))
        return z

    def train_on_graph(
        self,
        data: Dict[str, Any],
        epochs: int = 40,
        lr: float = 0.005,
        temp_init: float = 5.0,
        temp_min: float = 0.5,
        coeff_size: Optional[float] = None,
        coeff_ent: Optional[float] = None,
        seed: int = 0,
        batch_size: int = 32
    ) -> float:
        """Huấn luyện PGExplainer chỉ trên train_mask (BG03) qua 2-hop computation subgraphs."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        start_time = time.time()
        self.model.eval()

        x = data['x'].to(self.device)
        edge_index = data['edge_index'].to(self.device)
        train_mask = data.get('train_mask', None)
        if train_mask is None:
            train_node_indices = list(range(x.size(0)))
        elif isinstance(train_mask, torch.Tensor):
            if train_mask.dtype == torch.bool:
                train_node_indices = torch.where(train_mask)[0].cpu().tolist()
            else:
                train_node_indices = [int(i) for i in train_mask.cpu().tolist()]
        elif isinstance(train_mask, np.ndarray):
            if train_mask.dtype == bool or np.issubdtype(train_mask.dtype, np.bool_):
                train_node_indices = np.where(train_mask)[0].tolist()
            else:
                train_node_indices = [int(i) for i in train_mask.tolist()]
        elif isinstance(train_mask, (list, tuple)):
            if len(train_mask) > 0 and isinstance(train_mask[0], (bool, np.bool_)):
                train_node_indices = [i for i, b in enumerate(train_mask) if b]
            else:
                train_node_indices = [int(i) for i in train_mask]
        else:
            arr = list(train_mask)
            if len(arr) > 0 and isinstance(arr[0], (bool, np.bool_)):
                train_node_indices = [i for i, b in enumerate(arr) if b]
            else:
                train_node_indices = [int(i) for i in arr]

        # Xác định default regularizers theo dataset nếu không được chỉ định
        num_total_nodes = x.size(0)
        if coeff_size is None:
            if num_total_nodes < 1500:  # BA-Shapes (700 nodes)
                coeff_size = 0.005
            else:  # Cora (2708 nodes)
                coeff_size = 0.0001

        if coeff_ent is None:
            if coeff_size >= 0.005:
                coeff_ent = 0.05
            else:
                coeff_ent = 0.005

        z = self.get_embeddings(x, edge_index)

        with torch.no_grad():
            if 'pred' in data and data['pred'] is not None:
                full_preds = data['pred'].to(self.device)
            else:
                full_preds = self.model(x, edge_index).argmax(dim=-1)

        # Tiền trích xuất đồ thị con 2-hop cho các node trong train_mask
        precomputed: List[Dict[str, Any]] = []
        for node in train_node_indices:
            cand = extract_candidate_edges(edge_index, node, num_hops=2, num_nodes=num_total_nodes)
            if cand.num_candidate_edges == 0:
                continue

            sub_nodes = cand.subgraph_nodes.to(self.device)
            node_map = {n.item(): i for i, n in enumerate(sub_nodes)}
            local_target = node_map[node]

            pairs = cand.candidate_pairs
            u_cand = torch.tensor([p[0] for p in pairs], device=self.device, dtype=torch.long)
            v_cand = torch.tensor([p[1] for p in pairs], device=self.device, dtype=torch.long)
            t_cand = torch.full_like(u_cand, node)

            sub_edges = cand.subgraph_edge_index.to(self.device)
            local_src = torch.tensor([node_map[u.item()] for u in sub_edges[0]], device=self.device, dtype=torch.long)
            local_dst = torch.tensor([node_map[v.item()] for v in sub_edges[1]], device=self.device, dtype=torch.long)
            local_edge_index = torch.stack([local_src, local_dst], dim=0)

            precomputed.append({
                'node': node,
                'sub_x': x[sub_nodes],
                'local_edge_index': local_edge_index,
                'local_target': local_target,
                'u_cand': u_cand,
                'v_cand': v_cand,
                't_cand': t_cand,
                'edge_to_pair_idx': cand.edge_to_pair_idx.to(self.device),
                'target_class': full_preds[node]
            })

        num_subgraphs = len(precomputed)
        if num_subgraphs == 0:
            self.is_trained = True
            return time.time() - start_time

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self.mlp.train()

        for epoch in range(1, epochs + 1):
            tau = temp_init * ((temp_min / temp_init) ** (epoch / epochs))
            indices = torch.randperm(num_subgraphs)

            for b_start in range(0, num_subgraphs, batch_size):
                b_idx = indices[b_start : b_start + batch_size]
                u_cand_list = [precomputed[i]['u_cand'] for i in b_idx]
                v_cand_list = [precomputed[i]['v_cand'] for i in b_idx]
                t_cand_list = [precomputed[i]['t_cand'] for i in b_idx]

                batch_u = torch.cat(u_cand_list)
                batch_v = torch.cat(v_cand_list)
                batch_t = torch.cat(t_cand_list)

                # Đối xứng hóa: trung bình logit giữa (u, v) và (v, u)
                inp_uv = torch.cat([z[batch_u], z[batch_v], z[batch_t]], dim=-1)
                inp_vu = torch.cat([z[batch_v], z[batch_u], z[batch_t]], dim=-1)
                logits = (self.mlp(inp_uv).view(-1) + self.mlp(inp_vu).view(-1)) * 0.5

                # Gumbel-Sigmoid reparameterization
                eps = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
                gumbel_noise = torch.log(eps) - torch.log(1 - eps)
                soft_weights = torch.sigmoid((logits + gumbel_noise) / tau).view(-1)

                soft_weight_slices = torch.split(soft_weights, [len(u) for u in u_cand_list])

                data_list = []
                target_indices = []
                curr_offset = 0
                batch_labels = []

                for k, idx in enumerate(b_idx):
                    item = precomputed[idx]
                    sw = soft_weight_slices[k]
                    sub_w = sw[item['edge_to_pair_idx']]
                    data_list.append(Data(x=item['sub_x'], edge_index=item['local_edge_index'], edge_weight=sub_w))
                    target_indices.append(curr_offset + item['local_target'])
                    curr_offset += item['sub_x'].size(0)
                    batch_labels.append(item['target_class'])

                batch = Batch.from_data_list(data_list)
                out = self.model(batch.x, batch.edge_index, edge_weight=batch.edge_weight)

                target_out = out[target_indices]
                labels = torch.stack(batch_labels)

                loss_pred = F.cross_entropy(target_out, labels)
                loss_size = coeff_size * (soft_weights.sum() / len(b_idx))
                p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
                loss_ent = coeff_ent * (-p * torch.log(p) - (1 - p) * torch.log(1 - p)).mean()

                loss = loss_pred + loss_size + loss_ent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.mlp.eval()
        self.is_trained = True
        return time.time() - start_time


class PGExplainer(BaseExplainer):
    """Explainer interface bọc PGExplainerModel tuân theo Unified API BG04."""
    def __init__(self, trained_model: Optional[PGExplainerModel] = None):
        self.pg_model = trained_model

    def explain(
        self,
        model: torch.nn.Module,
        data: Dict[str, Any],
        node_id: int,
        target_class: int,
        candidate_edges: CandidateEdgesResult,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 0
    ) -> ExplainerResult:
        start_time = time.time()
        try:
            device = next(model.parameters()).device
            cfg = config or {}

            # Nếu chưa có model đã train, tự động train trên tập train_mask
            if self.pg_model is None or not self.pg_model.is_trained:
                self.pg_model = PGExplainerModel(model=model, device=device)
                self.pg_model.train_on_graph(
                    data=data,
                    epochs=cfg.get('epochs', 40),
                    lr=cfg.get('lr', 0.005),
                    temp_init=cfg.get('temp_init', 5.0),
                    temp_min=cfg.get('temp_min', 0.5),
                    coeff_size=cfg.get('coeff_size', None),
                    coeff_ent=cfg.get('coeff_ent', None),
                    seed=seed
                )
            else:
                if getattr(self.pg_model, 'device', None) != device:
                    self.pg_model.mlp = self.pg_model.mlp.to(device)
                    self.pg_model.device = device

            pairs = candidate_edges.candidate_pairs
            if len(pairs) == 0:
                return ExplainerResult(
                    edge_ids=list(candidate_edges.edge_ids),
                    scores=[],
                    runtime_sec=time.time() - start_time,
                    status="success"
                )

            x = data['x'].to(device)
            edge_index = data['edge_index'].to(device)
            z = self.pg_model.get_embeddings(x, edge_index)

            u_cand = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
            v_cand = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
            t_cand = torch.full_like(u_cand, node_id)

            self.pg_model.mlp.eval()
            with torch.no_grad():
                inp_uv = torch.cat([z[u_cand], z[v_cand], z[t_cand]], dim=-1)
                inp_vu = torch.cat([z[v_cand], z[u_cand], z[t_cand]], dim=-1)
                logits = (self.pg_model.mlp(inp_uv).view(-1) + self.pg_model.mlp(inp_vu).view(-1)) * 0.5
                raw_scores = torch.sigmoid(logits).cpu().numpy().tolist()
                if not isinstance(raw_scores, list):
                    raw_scores = [raw_scores]
                scores = [float(s) for s in raw_scores]

            return ExplainerResult(
                edge_ids=list(candidate_edges.edge_ids),
                scores=scores,
                runtime_sec=time.time() - start_time,
                status="success"
            )

        except Exception as e:
            return ExplainerResult(
                edge_ids=[],
                scores=[],
                runtime_sec=time.time() - start_time,
                status="failed",
                error_message=str(e)
            )
