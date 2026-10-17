"""PGExplainer (Thiện phụ trách).
Mạng neural MLP dự đoán cạnh quan trọng, train trên train_mask (BG03).
"""

import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sna.core.types import ExplainerResult, CandidateEdgesResult
from sna.explainers.base import BaseExplainer


class PGExplainerMLP(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, emb_u: torch.Tensor, emb_v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb_u, emb_v], dim=-1)
        return self.mlp(x).squeeze(-1)


class PGExplainerModel:
    def __init__(self, model: nn.Module, emb_dim: int, hidden_dim: int = 64, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.mlp = PGExplainerMLP(emb_dim=emb_dim, hidden_dim=hidden_dim).to(self.device)
        self.is_trained = False

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            z = self.model.c1(x.to(self.device), edge_index.to(self.device))
            z = F.relu(z)
        return z

    def train_on_graph(
        self,
        data: Dict[str, Any],
        epochs: int = 40,
        lr: float = 0.005,
        temp_init: float = 5.0,
        temp_min: float = 0.5,
        coeff_size: float = 0.005,
        coeff_ent: float = 0.1,
        seed: int = 0
    ) -> float:
        """Huấn luyện PGExplainer chỉ trên train_mask (BG03)."""
        torch.manual_seed(seed)
        start_time = time.time()

        x = data['x'].to(self.device)
        edge_index = data['edge_index'].to(self.device)
        y = data['y'].to(self.device)
        train_mask = data['train_mask']
        train_node_indices = [i for i, is_train in enumerate(train_mask) if is_train]

        z = self.get_embeddings(x, edge_index)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self.mlp.train()

        u_idx = edge_index[0]
        v_idx = edge_index[1]

        for epoch in range(1, epochs + 1):
            tau = temp_init * ((temp_min / temp_init) ** (epoch / epochs))
            optimizer.zero_grad()

            edge_logits = self.mlp(z[u_idx], z[v_idx])
            eps = torch.rand_like(edge_logits).clamp(1e-7, 1 - 1e-7)
            gumbel_noise = torch.log(eps) - torch.log(1 - eps)
            soft_edge_mask = torch.sigmoid((edge_logits + gumbel_noise) / tau)

            h = self.model.c1(x, edge_index, edge_weight=soft_edge_mask)
            h = F.relu(h)
            out = self.model.c2(h, edge_index, edge_weight=soft_edge_mask)

            train_logits = out[train_node_indices]
            train_labels = y[train_node_indices]
            loss_pred = F.cross_entropy(train_logits, train_labels)

            loss_size = coeff_size * torch.mean(soft_edge_mask)
            p = torch.sigmoid(edge_logits)
            ent = -p * torch.log(p + 1e-12) - (1 - p) * torch.log(1 - p + 1e-12)
            loss_ent = coeff_ent * torch.mean(ent)

            loss = loss_pred + loss_size + loss_ent
            loss.backward()
            optimizer.step()

        self.mlp.eval()
        self.is_trained = True
        return time.time() - start_time


class PGExplainer(BaseExplainer):
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

            # Nếu chưa có model đã train, tự động train nhanh
            if self.pg_model is None or not self.pg_model.is_trained:
                emb_dim = model.c1.out_channels
                self.pg_model = PGExplainerModel(model=model, emb_dim=emb_dim, device=device)
                self.pg_model.train_on_graph(
                    data=data,
                    epochs=cfg.get('epochs', 40),
                    lr=cfg.get('lr', 0.005),
                    seed=seed
                )

            x = data['x'].to(device)
            edge_index = data['edge_index'].to(device)
            z = self.pg_model.get_embeddings(x, edge_index)

            scores = []
            self.pg_model.mlp.eval()
            with torch.no_grad():
                for (u, v) in candidate_edges.candidate_pairs:
                    logit_uv = self.pg_model.mlp(z[u:u+1], z[v:v+1])
                    prob_uv = torch.sigmoid(logit_uv).item()

                    logit_vu = self.pg_model.mlp(z[v:v+1], z[u:u+1])
                    prob_vu = torch.sigmoid(logit_vu).item()

                    score = max(prob_uv, prob_vu)
                    scores.append(float(score))

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
