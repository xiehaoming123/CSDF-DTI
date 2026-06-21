import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import MambaBlock, MambaConfig
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import TransformerConv, global_max_pool, GINEConv
from torch_geometric.nn.norm import GraphNorm

def masked_max(x, mask, dim=1):
    mask = mask.unsqueeze(-1).expand_as(x)
    x = x.masked_fill(~mask, float('-inf'))
    out = x.max(dim=dim).values
    out[out == float('-inf')] = 0
    return out

def masked_mean(x, mask, dim=1, eps=1e-08):
    mask_f = mask.unsqueeze(-1).float()
    x = x * mask_f
    denom = mask_f.sum(dim=dim).clamp_min(eps)
    return x.sum(dim=dim) / denom

def weighted_masked_pool(x, weight, mask, eps=1e-08):
    weight = weight * mask.unsqueeze(-1).float()
    norm = weight.sum(dim=1, keepdim=True).clamp_min(eps)
    weight = weight / norm
    pooled = (x * weight).sum(dim=1)
    return pooled

def reverse_valid(x, mask):
    if mask is None:
        return x.flip(1)
    mask = mask.bool()
    B, L, D = x.shape
    lengths = mask.long().sum(dim=1)
    idx = torch.arange(L, device=x.device).view(1, L).expand(B, L)
    rev_idx = (lengths.view(B, 1) - 1 - idx).clamp(min=0)
    y = x.gather(dim=1, index=rev_idx.unsqueeze(-1).expand(B, L, D))
    y = y * mask.unsqueeze(-1).to(x.dtype)
    return y

class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (-ctx.lambd * grad_output, None)

def grad_reverse(x, lambd=1.0):
    return GradientReversalFunction.apply(x, lambd)

def pairwise_rank_loss(logits, labels, margin=0.15, tau=0.15, max_pairs=4096):
    logits = logits.view(-1)
    labels = labels.float().view(-1)
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    num_pairs = pos.numel() * neg.numel()
    if num_pairs > max_pairs:
        pair_pos = torch.randint(0, pos.numel(), (max_pairs,), device=logits.device)
        pair_neg = torch.randint(0, neg.numel(), (max_pairs,), device=logits.device)
        diff = pos[pair_pos] - neg[pair_neg]
    else:
        diff = pos[:, None] - neg[None, :]
    return F.softplus((margin - diff) / tau).mean()

def e4_pr_loss(logits, labels, aux=None, sample_weight=None, bce_weight=1.0, rank_weight=0.25, causal_bce_weight=0.3, cf_consistency_weight=0.05, gcl_weight=0.2, conf_align_weight=0.005, rank_margin=0.15, rank_tau=0.15):
    labels = labels.float().view(-1)
    logits = logits.view(-1)
    bce_each = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    if sample_weight is not None:
        sw = sample_weight.to(logits.device, dtype=logits.dtype).view(-1)
        bce = (bce_each * sw).sum() / sw.sum().clamp_min(1e-08)
    else:
        bce = bce_each.mean()
    rank = pairwise_rank_loss(logits, labels, margin=rank_margin, tau=rank_tau)
    total = bce_weight * bce + rank_weight * rank
    logs = {'loss_bce': bce.detach(), 'loss_rank': rank.detach()}
    if aux is not None:
        if 'pred_causal' in aux:
            causal_bce = F.binary_cross_entropy_with_logits(aux['pred_causal'].view(-1), labels)
            total = total + causal_bce_weight * causal_bce
            logs['loss_causal_bce'] = causal_bce.detach()
        if 'pred_cf_drug' in aux and 'pred_cf_prot' in aux and ('pred_causal' in aux):
            pred_causal_detached = aux['pred_causal'].detach()
            cf_loss = F.mse_loss(aux['pred_cf_drug'], pred_causal_detached) + F.mse_loss(aux['pred_cf_prot'], pred_causal_detached)
            total = total + cf_consistency_weight * cf_loss
            logs['loss_cf_consistency'] = cf_loss.detach()
        if 'gcl_loss' in aux:
            total = total + gcl_weight * aux['gcl_loss']
            logs['loss_gcl'] = aux['gcl_loss'].detach()
        if 'conf_align_loss' in aux:
            total = total + conf_align_weight * aux['conf_align_loss']
            logs['loss_conf_align'] = aux['conf_align_loss'].detach()
    logs['loss_total'] = total.detach()
    return (total, logs)

def build_protein_physchem_table():
    table = torch.zeros(26, 5, dtype=torch.float32)
    table[1] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    table[2] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    table[3] = torch.tensor([-0.5, 0.5, 1.0, 0.0, 0.0])
    table[4] = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0])
    table[5] = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0])
    table[6] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    table[7] = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    table[8] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    table[9] = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    table[10] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    table[11] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    table[12] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    table[13] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    table[14] = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
    table[15] = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
    table[16] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    table[17] = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
    table[18] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    table[19] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    table[20] = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
    table[21] = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    table[22] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    table[23] = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    table[24] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    table[25] = torch.tensor([-0.5, 0.5, 1.0, 0.0, 0.0])
    return table

class PhyschemFeatureFusion(nn.Module):

    def __init__(self, hidden_dim, phys_dim=5, dropout=0.1):
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(hidden_dim + phys_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, phys):
        delta = self.fuse(torch.cat([x, phys], dim=-1))
        out = self.norm(x + delta)
        return out

def shuffle_valid_phys(phys, mask):
    out = phys.clone()
    B = phys.size(0)
    for b in range(B):
        valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        if valid_idx.numel() > 1:
            perm = valid_idx[torch.randperm(valid_idx.numel(), device=phys.device)]
            out[b, valid_idx] = phys[b, perm]
    return out

def shuffle_graph_phys(phys, batch):
    out = phys.clone()
    if batch.numel() == 0:
        return out
    num_graphs = int(batch.max().item()) + 1
    for g in range(num_graphs):
        idx = torch.nonzero(batch == g, as_tuple=False).squeeze(-1)
        if idx.numel() > 1:
            perm = idx[torch.randperm(idx.numel(), device=phys.device)]
            out[idx] = phys[perm]
    return out

class GPSConv(torch.nn.Module):

    def __init__(self, channels: int, edge_dim: int, heads: int=1, dropout: float=0.2, attn_dropout: float=0.2, act=torch.relu):
        super().__init__()
        self.channels = channels
        self.act = act
        self.heads = heads
        self.dropout = dropout
        self.conv = TransformerConv(channels, channels // heads, heads=heads, edge_dim=edge_dim, beta=True, dropout=0.1)
        self.linear = nn.Linear(channels * heads, channels)
        self.attn = BidirMambaBlock(channels)
        self.mlp = SwiGLU(channels, channels * 4)
        self.bn = GraphNorm(channels)

    def forward(self, x, edge_index, edge_attr, batch):
        hs = []
        h = self.conv(x, edge_index, edge_attr)
        h = self.linear(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x
        h = self.bn(h, batch)
        hs.append(h)
        dense_h, mask = to_dense_batch(h, batch)
        dense_h = self.attn(dense_h, mask=mask)
        dense_h = dense_h[mask]
        dense_h = F.dropout(dense_h, p=self.dropout, training=self.training)
        dense_h = dense_h + h
        dense_h = self.bn(dense_h, batch)
        hs.append(dense_h)
        out = sum(hs)
        out = out + self.mlp(out)
        out = self.bn(out, batch)
        return out

class SwiGLU(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class BidirMambaBlock(nn.Module):

    def __init__(self, n_embed, weight_tie=True) -> None:
        super().__init__()
        config = MambaConfig(d_model=n_embed, n_layers=1)
        self.mixer = MambaBlock(config)
        self.mixer_back = MambaBlock(config)
        self.ln = nn.LayerNorm(n_embed)
        if weight_tie:
            self.mixer_back.in_proj.weight = self.mixer.in_proj.weight
            self.mixer_back.in_proj.bias = self.mixer.in_proj.bias
            self.mixer_back.out_proj.weight = self.mixer.out_proj.weight
            self.mixer_back.out_proj.bias = self.mixer.out_proj.bias

    def forward(self, x, mask=None):
        if mask is None:
            mix_flip = self.mixer_back(x.flip(1))
            x = self.ln(x + self.mixer(x) + mix_flip.flip(1))
            return x
        mask = mask.bool()
        mask_f = mask.unsqueeze(-1).float()
        x = x * mask_f
        mix_fwd = self.mixer(x) * mask_f
        x_rev = reverse_valid(x, mask)
        mix_bwd = self.mixer_back(x_rev)
        mix_bwd = reverse_valid(mix_bwd, mask) * mask_f
        out = self.ln(x + mix_fwd + mix_bwd)
        out = out * mask_f
        return out

class AM_Layer(nn.Module):

    def __init__(self, d_model):
        super(AM_Layer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, d_model // 32, batch_first=True)
        self.mamba = BidirMambaBlock(d_model)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x = x + self.self_attention(x, x, x, key_padding_mask=key_padding_mask)[0]
        x = self.norm1(x)
        valid_mask = None
        if key_padding_mask is not None:
            valid_mask = ~key_padding_mask
        x = self.mamba(x, mask=valid_mask)
        return x

class PhysGuidedBidirectionalFusion(nn.Module):

    def __init__(self, dim, phys_dim=5, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        fused_in_dim = dim + phys_dim
        self.q_a = nn.Linear(fused_in_dim, dim)
        self.k_a = nn.Linear(fused_in_dim, dim)
        self.v_a = nn.Linear(dim, dim)
        self.q_b = nn.Linear(fused_in_dim, dim)
        self.k_b = nn.Linear(fused_in_dim, dim)
        self.v_b = nn.Linear(dim, dim)
        self.phys_bias_ab = nn.Sequential(nn.Linear(phys_dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.phys_bias_ba = nn.Sequential(nn.Linear(phys_dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.gate_a = nn.Sequential(nn.Linear(dim * 2 + phys_dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_b = nn.Sequential(nn.Linear(dim * 2 + phys_dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.Sigmoid())
        self.update_a = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.update_b = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, feat_a, feat_b, phys_a, phys_b):
        qa = self.q_a(torch.cat([feat_a, phys_a], dim=-1))
        kb = self.k_b(torch.cat([feat_b, phys_b], dim=-1))
        vb = self.v_b(feat_b)
        att_ab = (qa * kb).sum(dim=-1, keepdim=True) / math.sqrt(self.dim)
        att_ab = att_ab + self.phys_bias_ab(torch.cat([phys_a, phys_b], dim=-1))
        att_ab = torch.sigmoid(att_ab)
        ctx_ab = att_ab * vb
        gate_a = self.gate_a(torch.cat([feat_a, feat_b, phys_a, phys_b], dim=-1))
        delta_a = self.update_a(torch.cat([feat_a, ctx_ab], dim=-1))
        fused_a = self.norm_a(feat_a + gate_a * delta_a)
        qb = self.q_b(torch.cat([feat_b, phys_b], dim=-1))
        ka = self.k_a(torch.cat([feat_a, phys_a], dim=-1))
        va = self.v_a(feat_a)
        att_ba = (qb * ka).sum(dim=-1, keepdim=True) / math.sqrt(self.dim)
        att_ba = att_ba + self.phys_bias_ba(torch.cat([phys_b, phys_a], dim=-1))
        att_ba = torch.sigmoid(att_ba)
        ctx_ba = att_ba * va
        gate_b = self.gate_b(torch.cat([feat_b, feat_a, phys_b, phys_a], dim=-1))
        delta_b = self.update_b(torch.cat([feat_b, ctx_ba], dim=-1))
        fused_b = self.norm_b(feat_b + gate_b * delta_b)
        return torch.cat((fused_a, fused_b), dim=1)

class TwoModalPhysGuidedFusion(nn.Module):

    def __init__(self, dim, phys_dim=5, dropout=0.1):
        super(TwoModalPhysGuidedFusion, self).__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.gated_d = SwiGLU(dim, dim * 4)
        self.gated_p = SwiGLU(dim, dim * 4)
        self.norm_d = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        self.fusion_dp = PhysGuidedBidirectionalFusion(dim, phys_dim=phys_dim, dropout=dropout)

    def forward(self, drug_feat, prot_feat, drug_phys=None, prot_phys=None):
        if drug_phys is None:
            drug_phys = torch.zeros(drug_feat.size(0), self.phys_dim, device=drug_feat.device, dtype=drug_feat.dtype)
        if prot_phys is None:
            prot_phys = torch.zeros(prot_feat.size(0), self.phys_dim, device=prot_feat.device, dtype=prot_feat.dtype)
        drug_feat = self.norm_d(drug_feat + self.gated_d(drug_feat))
        prot_feat = self.norm_p(prot_feat + self.gated_p(prot_feat))
        return self.fusion_dp(drug_feat, prot_feat, drug_phys, prot_phys)

class CausalShortcutSplitter(nn.Module):

    def __init__(self, hidden_dim, phys_dim=5, dropout=0.1, hidden_assist_scale=0.3, num_confounders=5, grl_lambda=1.0, target_causal_ratio=0.6):
        super().__init__()
        mid_dim = hidden_dim // 2
        self.phys_dim = phys_dim
        self.hidden_assist_scale = hidden_assist_scale
        self.num_confounders = num_confounders
        self.grl_lambda = grl_lambda
        self.target_causal_ratio = target_causal_ratio
        self.conf_assign_head = nn.Sequential(nn.Linear(phys_dim, mid_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid_dim, num_confounders))
        self.confounder_token_bank = nn.Parameter(torch.randn(num_confounders, hidden_dim) * 0.02)
        self.shortcut_gate_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.shortcut_bias_classifier = nn.Sequential(nn.Linear(hidden_dim, mid_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid_dim, num_confounders))
        self.causal_adv_classifier = nn.Sequential(nn.Linear(hidden_dim, mid_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid_dim, num_confounders))

    def forward(self, x, mask, phys=None, ext_conf_id=None):
        mask = mask.bool()
        if phys is None:
            phys = torch.zeros(x.size(0), x.size(1), self.phys_dim, dtype=x.dtype, device=x.device)
        conf_logits_raw = self.conf_assign_head(phys)
        conf_logits = conf_logits_raw.masked_fill(~mask.unsqueeze(-1), -10000.0)
        conf_prob = F.softmax(conf_logits, dim=-1)
        internal_conf_context = torch.matmul(conf_prob, self.confounder_token_bank)
        conf_context = internal_conf_context
        conf_align_loss = x.new_zeros(())
        valid_ext_mask = None
        if ext_conf_id is not None:
            ext_conf_id = ext_conf_id.to(x.device).long().view(-1)
            if ext_conf_id.size(0) != x.size(0):
                raise ValueError(f'ext_conf_id batch size mismatch: ext_conf_id.shape={tuple(ext_conf_id.shape)}, x.shape={tuple(x.shape)}')
            if torch.any((ext_conf_id < -1) | (ext_conf_id >= self.num_confounders)):
                raise ValueError(f'Invalid ext_conf_id. Allowed range: [0, {self.num_confounders - 1}], or -1 for fallback.')
            valid_ext_mask = ext_conf_id >= 0
            if valid_ext_mask.any():
                safe_ext_conf_id = ext_conf_id.clamp(min=0)
                ext_conf_context = self.confounder_token_bank[safe_ext_conf_id].unsqueeze(1).expand(-1, x.size(1), -1)
                mixed_conf_context = 0.7 * ext_conf_context + 0.3 * internal_conf_context
                conf_context = torch.where(valid_ext_mask.view(-1, 1, 1), mixed_conf_context, conf_context)
                conf_logits_pool = masked_mean(conf_logits_raw, mask, dim=1)
                conf_align_loss = F.cross_entropy(conf_logits_pool[valid_ext_mask], ext_conf_id[valid_ext_mask])
        gate_inp = torch.cat([x, conf_context], dim=-1)
        shortcut_score = self.shortcut_gate_head(gate_inp)
        shortcut_score = shortcut_score.masked_fill(~mask.unsqueeze(-1), -10000.0)
        shortcut_w = torch.sigmoid(shortcut_score) * mask.unsqueeze(-1).float()
        causal_w = (1.0 - shortcut_w) * mask.unsqueeze(-1).float()
        causal_feat = x * causal_w
        shortcut_feat = x * shortcut_w
        causal_pool = weighted_masked_pool(x, causal_w, mask)
        shortcut_pool = weighted_masked_pool(x, shortcut_w, mask)
        conf_pool = masked_mean(conf_prob, mask, dim=1)
        internal_conf_target = conf_pool.detach().argmax(dim=-1)
        if ext_conf_id is not None:
            if valid_ext_mask is None:
                valid_ext_mask = ext_conf_id >= 0
            conf_target = torch.where(valid_ext_mask, ext_conf_id, internal_conf_target)
            conf_prob_for_aux = conf_pool.clone()
            if valid_ext_mask.any():
                conf_prob_for_aux[valid_ext_mask] = F.one_hot(ext_conf_id[valid_ext_mask], num_classes=self.num_confounders).float()
        else:
            conf_target = internal_conf_target
            conf_prob_for_aux = conf_pool
        shortcut_bias_logits = self.shortcut_bias_classifier(shortcut_pool)
        shortcut_bias_loss = F.cross_entropy(shortcut_bias_logits, conf_target)
        causal_adv_logits = self.causal_adv_classifier(grad_reverse(causal_pool, self.grl_lambda))
        causal_adv_loss = F.cross_entropy(causal_adv_logits, conf_target)
        causal_ratio = causal_w.squeeze(-1).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
        sparse_loss = (causal_ratio - self.target_causal_ratio).pow(2).mean()
        orth_loss = F.cosine_similarity(causal_pool, shortcut_pool, dim=-1).pow(2).mean()
        aux = {'causal_ratio': causal_ratio.mean().detach(), 'sparse_loss': sparse_loss, 'orth_loss': orth_loss, 'shortcut_bias_loss': shortcut_bias_loss, 'causal_adv_loss': causal_adv_loss, 'conf_align_loss': conf_align_loss, 'conf_target': conf_target.detach(), 'conf_prob': conf_prob_for_aux.detach()}
        return (causal_feat, shortcut_feat, causal_pool, shortcut_pool, causal_w, shortcut_w, aux)

class EdgeAwareGINEEncoder(nn.Module):

    def __init__(self, in_dim=26, hidden_dim=256, edge_dim=14, num_layers=4, dropout=0.15):
        super().__init__()
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.virtual_node = nn.Parameter(torch.zeros(1, hidden_dim))
        self.virtual_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(GraphNorm(hidden_dim))
        self.out_mlp = SwiGLU(hidden_dim, hidden_dim * 4)
        self.out_norm = GraphNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_proj(x)
        if batch.numel() > 0:
            num_graphs = int(batch.max().item()) + 1
            virtual = self.virtual_node.expand(num_graphs, -1)
            h = h + virtual[batch]
        else:
            num_graphs = 0
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), self.edge_dim, device=h.device, dtype=h.dtype)
        else:
            edge_attr = edge_attr.to(device=h.device, dtype=h.dtype)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
        for conv, norm in zip(self.layers, self.norms):
            residual = h
            h = conv(h, edge_index, edge_attr)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = norm(h + residual, batch)
            if num_graphs > 0:
                graph_context = global_max_pool(h, batch)
                virtual_update = self.virtual_mlp(graph_context)
                h = h + virtual_update[batch]
        h = h + self.out_mlp(h)
        h = self.out_norm(h, batch)
        return h

class GraphMultiPool(nn.Module):

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, x, batch):
        if batch.numel() == 0:
            return x.new_zeros((0, x.size(-1)))
        num_graphs = int(batch.max().item()) + 1
        max_pool = global_max_pool(x, batch)
        mean_pool = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        mean_pool = mean_pool.index_add(0, batch, x)
        count = torch.bincount(batch, minlength=num_graphs).to(x.device).to(x.dtype).clamp_min(1.0)
        mean_pool = mean_pool / count.unsqueeze(-1)
        score = self.attn(x)
        score = score - global_max_pool(score, batch)[batch]
        weight = torch.exp(score)
        denom = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        denom = denom.index_add(0, batch, weight).clamp_min(1e-08)
        attn_pool = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        attn_pool = attn_pool.index_add(0, batch, x * weight)
        attn_pool = attn_pool / denom
        return self.proj(torch.cat([max_pool, mean_pool, attn_pool], dim=-1))

class MambaCPAModelWoPretrained(nn.Module):

    def __init__(self):
        super().__init__()
        hidden_size = 256
        self.prot_embedding = nn.Embedding(26, hidden_size, padding_idx=0)
        self.smiles_vocab_size = 128
        self.comp_embedding = nn.Embedding(self.smiles_vocab_size, hidden_size, padding_idx=0)
        self.mol_graph_net1 = EdgeAwareGINEEncoder(in_dim=26, hidden_dim=hidden_size, edge_dim=14, num_layers=4, dropout=0.15)
        self.graph_multi_pool = GraphMultiPool(hidden_size, dropout=0.1)
        self.graph_pool_residual_gate = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))
        config = MambaConfig(d_model=hidden_size, n_layers=1)
        self.pre_mamba_prot = MambaBlock(config)
        self.prot_mamba = nn.Sequential(AM_Layer(hidden_size))
        self.pre_mamba_comp = MambaBlock(config)
        self.comp_mamba = nn.Sequential(AM_Layer(hidden_size))
        self.comp_graph_fusion = nn.Sequential(nn.LayerNorm(hidden_size * 4), nn.Linear(hidden_size * 4, hidden_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
        self.comp_graph_gate = nn.Parameter(torch.tensor(math.log(0.25 / 0.75), dtype=torch.float32))
        self.prot_pool_residual_gate = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))
        self.phys_dim = 5
        self.num_confounders = 5
        self.use_local_phys_fusion = True
        self.use_phys_guided_fusion = True
        self.use_causal_splitter = True
        self.drug_phys_alpha = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))
        self.prot_phys_alpha = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))
        self.comp_mask_prob = 0.0
        self.comp_mut_prob = 0.0
        self.external_conf_dropout_comp = 0.0
        self.register_buffer('protein_physchem_table', build_protein_physchem_table())
        self.drug_phys_proj = nn.Sequential(nn.Linear(26, hidden_size), nn.GELU(), nn.Linear(hidden_size, self.phys_dim), nn.Tanh())
        self.drug_phys_fusion = PhyschemFeatureFusion(hidden_dim=hidden_size, phys_dim=self.phys_dim, dropout=0.1)
        self.prot_phys_fusion = PhyschemFeatureFusion(hidden_dim=hidden_size, phys_dim=self.phys_dim, dropout=0.1)
        self.drug_causal_splitter = CausalShortcutSplitter(hidden_dim=hidden_size, phys_dim=self.phys_dim, dropout=0.1, hidden_assist_scale=0.2, num_confounders=self.num_confounders, grl_lambda=1.0, target_causal_ratio=0.7)
        self.prot_causal_splitter = CausalShortcutSplitter(hidden_dim=hidden_size, phys_dim=self.phys_dim, dropout=0.1, hidden_assist_scale=0.2, num_confounders=self.num_confounders, grl_lambda=1.2, target_causal_ratio=0.8)
        self.lambda_split_sparse = 0.002
        self.lambda_split_orth = 0.0005
        self.lambda_split_bias = 0.002
        self.lambda_split_adv = 0.012
        self.lambda_gate_reg = 0.008
        self.external_conf_dropout_drug = 0.65
        self.external_conf_dropout_prot = 0.7
        self.reliability_seen_weight = 0.25
        self.reliability_conf_weight = 0.75
        self.phys_calib_scale = nn.Parameter(torch.tensor(math.log(0.08 / 0.92), dtype=torch.float32))
        self.phys_calibrator = nn.Sequential(nn.Linear(self.phys_dim * 3, hidden_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_size, 2 * hidden_size))
        self.enable_tapb_seq_randomization = True
        self.prot_mask_prob = 0.0
        self.prot_mut_prob = 0.08
        self.shortcut_residual_gate = nn.Parameter(torch.tensor(math.log(0.03 / 0.97), dtype=torch.float32))
        dropout = 0.15
        self.pred_net = nn.Sequential(nn.LayerNorm(2 * hidden_size), nn.Linear(2 * hidden_size, 1024), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(1024), nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(512), nn.Linear(512, 1))
        self.pair_logit_scale = nn.Parameter(torch.tensor(math.log(0.08 / 0.92), dtype=torch.float32))
        self.pair_logit_head = nn.Sequential(nn.LayerNorm(4 * hidden_size), nn.Linear(4 * hidden_size, hidden_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_size, 1))
        self.multi_modal_fusion = TwoModalPhysGuidedFusion(hidden_size, phys_dim=self.phys_dim, dropout=0.1)
        self._cached_fusion_phys = None
        self.model_variant = {'drug_input': '2D_molecular_graph_plus_SMILES', 'drug_encoder': '4_layer_edge_aware_GINEConv_virtual_node_residual_GraphNorm', 'drug_pooling': 'max_mean_attention_GraphMultiPool', 'protein_input': 'protein_sequence', 'protein_encoder': 'Mamba', 'smiles_branch': 'restored_lightweight_Mamba_SMILES_residual', 'use_local_phys_fusion': self.use_local_phys_fusion, 'use_phys_guided_fusion': self.use_phys_guided_fusion, 'use_causal_splitter': self.use_causal_splitter, 'observed_unknown_bin_semantics': 'fallback_-1_to_internal_assignment', 'num_confounders': self.num_confounders}

    def get_protein_physchem_prior(self, pro_seq_id):
        return self.protein_physchem_table[pro_seq_id]

    def apply_phys_mode(self, drug_phys, prot_phys, drug_batch, prot_mask, phys_mode='normal'):
        if phys_mode == 'normal':
            return (drug_phys, prot_phys)
        elif phys_mode == 'zero':
            return (torch.zeros_like(drug_phys), torch.zeros_like(prot_phys))
        elif phys_mode == 'shuffle':
            drug_phys = shuffle_graph_phys(drug_phys, drug_batch)
            prot_phys = shuffle_valid_phys(prot_phys, prot_mask)
            return (drug_phys, prot_phys)
        else:
            raise ValueError(f'Unsupported phys_mode: {phys_mode}')

    def _tapb_randomize_ids(self, seq_id, vocab_high, mask_prob=0.0, mutate_prob=0.0):
        if not self.training or not self.enable_tapb_seq_randomization:
            return seq_id
        out = seq_id.clone()
        valid = out != 0
        if mask_prob > 0:
            mask = valid & (torch.rand(out.shape, device=out.device) < mask_prob)
            out[mask] = 0
            valid = out != 0
        if mutate_prob > 0:
            mutate = valid & (torch.rand(out.shape, device=out.device) < mutate_prob)
            if mutate.any():
                rand_ids = torch.randint(1, vocab_high + 1, size=(int(mutate.sum().item()),), device=out.device)
                out[mutate] = rand_ids.to(out.dtype)
        return out

    def _maybe_dropout_external_conf(self, ext_conf_id, drop_prob=0.0):
        if ext_conf_id is None or not self.training or drop_prob <= 0:
            return ext_conf_id
        ext = ext_conf_id.clone()
        valid = ext >= 0
        if valid.any():
            drop_mask = valid & (torch.rand(ext.shape, device=ext.device) < drop_prob)
            ext[drop_mask] = -1
        return ext

    def _conf_reliability(self, conf_prob):
        if conf_prob is None:
            return None
        conf_prob = conf_prob.clamp_min(1e-08)
        entropy = -(conf_prob * conf_prob.log()).sum(dim=-1) / math.log(conf_prob.size(-1))
        reliability = 1.0 - entropy
        return reliability.clamp(0.0, 1.0)

    def _seen_reliability(self, ext_conf_id, ref_tensor):
        if ext_conf_id is None:
            return torch.zeros(ref_tensor.size(0), device=ref_tensor.device, dtype=ref_tensor.dtype)
        return (ext_conf_id.to(ref_tensor.device) >= 0).float().to(ref_tensor.dtype)

    def _build_shortcut_scale(self, conf_prob, ext_conf_id, ref_tensor):
        seen_rel = self._seen_reliability(ext_conf_id, ref_tensor)
        conf_rel = self._conf_reliability(conf_prob)
        if conf_rel is None:
            conf_rel = torch.ones_like(seen_rel)
        else:
            conf_rel = conf_rel.to(ref_tensor.dtype)
        scale = self.reliability_seen_weight * seen_rel + self.reliability_conf_weight * conf_rel
        scale = scale.clamp(0.05, 1.0)
        return scale.unsqueeze(-1)

    def _encode_smiles_from_graph_attr(self, mol_graph, ref_tensor):
        smiles_id = getattr(mol_graph, 'smiles_id', None)
        if smiles_id is None:
            return None
        smiles_id = smiles_id.to(device=ref_tensor.device).long()
        if smiles_id.dim() == 1:
            smiles_id = smiles_id.unsqueeze(0)
        smiles_id = smiles_id.clamp(min=0, max=self.smiles_vocab_size - 1)
        comp_mask = smiles_id != 0
        comp_mask_f = comp_mask.unsqueeze(-1).to(ref_tensor.dtype)
        comp_emb = self.comp_embedding(smiles_id)
        comp_emb = comp_emb * comp_mask_f
        comp_emb = self.pre_mamba_comp(comp_emb) * comp_mask_f
        comp_emb = self.comp_mamba[0](comp_emb, key_padding_mask=~comp_mask)
        comp_emb = comp_emb * comp_mask_f
        comp_pool = 0.5 * masked_max(comp_emb, comp_mask, dim=1) + 0.5 * masked_mean(comp_emb, comp_mask, dim=1)
        return comp_pool

    def _match_counterfactual_perm(self, causal_pool, conf_target):
        bsz = causal_pool.size(0)
        device = causal_pool.device
        if bsz <= 1:
            return torch.arange(bsz, device=device)
        sim = F.cosine_similarity(causal_pool.unsqueeze(1), causal_pool.unsqueeze(0), dim=-1)
        eye = torch.eye(bsz, device=device, dtype=torch.bool)
        same_conf = conf_target.unsqueeze(0) == conf_target.unsqueeze(1)
        valid = ~eye & ~same_conf
        sim = sim.masked_fill(~valid, -10000.0)
        perm = sim.argmax(dim=1)
        invalid_row = valid.sum(dim=1) == 0
        if invalid_row.any():
            base = torch.arange(bsz, device=device)
            rand_perm = torch.randperm(bsz, device=device)
            while torch.any(rand_perm == base):
                rand_perm = torch.randperm(bsz, device=device)
            perm[invalid_row] = rand_perm[invalid_row]
        return perm

    def _predict_from_parts(self, mol_graph_x, prot_seq_x, return_vis=False):
        if self._cached_fusion_phys is not None:
            drug_phys_pool, prot_phys_pool = self._cached_fusion_phys
        else:
            zero_phys = torch.zeros(mol_graph_x.size(0), self.phys_dim, device=mol_graph_x.device, dtype=mol_graph_x.dtype)
            drug_phys_pool = zero_phys
            prot_phys_pool = zero_phys
        dp_embedding = self.multi_modal_fusion(mol_graph_x, prot_seq_x, drug_phys=drug_phys_pool, prot_phys=prot_phys_pool)
        phys_pair = torch.cat([drug_phys_pool, prot_phys_pool, torch.abs(drug_phys_pool - prot_phys_pool)], dim=-1)
        dp_embedding = dp_embedding + torch.sigmoid(self.phys_calib_scale) * self.phys_calibrator(phys_pair)
        pred = self.pred_net(dp_embedding)
        pair_feat = torch.cat([mol_graph_x, prot_seq_x, torch.abs(mol_graph_x - prot_seq_x), mol_graph_x * prot_seq_x], dim=-1)
        pred = pred + torch.sigmoid(self.pair_logit_scale) * self.pair_logit_head(pair_feat)
        if return_vis:
            pred_vis = {'mol_graph_x': mol_graph_x, 'prot_seq_x': prot_seq_x, 'drug_phys_pool': drug_phys_pool, 'prot_phys_pool': prot_phys_pool, 'dp_embedding': dp_embedding, 'pair_feat': pair_feat}
            return (pred, pred_vis)
        return pred

    def forward(self, pro_seq_id, mol_graph, return_aux=False, return_details=False, pred_branch='hybrid', phys_mode='normal', custom_drug_phys=None, custom_prot_phys=None, external_drug_conf_id=None, external_prot_conf_id=None, compute_cf=True, return_vis=False):
        self._cached_fusion_phys = None
        if pred_branch not in ['hybrid', 'causal', 'shortcut']:
            raise ValueError(f'Unsupported pred_branch: {pred_branch}')
        raw_drug_x = mol_graph.x
        if return_vis:
            raw_drug_x = raw_drug_x.detach().clone().requires_grad_(True)
            mol_graph.x = raw_drug_x
        prot_mask = pro_seq_id != 0
        pro_seq_id = self._tapb_randomize_ids(pro_seq_id, vocab_high=25, mask_prob=self.prot_mask_prob, mutate_prob=self.prot_mut_prob)
        external_drug_conf_id = self._maybe_dropout_external_conf(external_drug_conf_id, self.external_conf_dropout_drug)
        external_prot_conf_id = self._maybe_dropout_external_conf(external_prot_conf_id, self.external_conf_dropout_prot)
        drug_phys = self.drug_phys_proj(raw_drug_x)
        prot_phys = self.get_protein_physchem_prior(pro_seq_id).to(raw_drug_x.dtype)
        if custom_drug_phys is not None:
            drug_phys = custom_drug_phys.to(raw_drug_x.device, dtype=raw_drug_x.dtype)
        if custom_prot_phys is not None:
            prot_phys = custom_prot_phys.to(pro_seq_id.device, dtype=raw_drug_x.dtype)
        if custom_drug_phys is None or custom_prot_phys is None:
            default_drug_phys = drug_phys
            default_prot_phys = prot_phys
            mod_drug_phys, mod_prot_phys = self.apply_phys_mode(default_drug_phys, default_prot_phys, mol_graph.batch, prot_mask, phys_mode=phys_mode)
            if custom_drug_phys is None:
                drug_phys = mod_drug_phys
            if custom_prot_phys is None:
                prot_phys = mod_prot_phys
        graph_x = self.mol_graph_net1(raw_drug_x, mol_graph.edge_index, getattr(mol_graph, 'edge_attr', None), mol_graph.batch)
        graph_global_pool = self.graph_multi_pool(graph_x, mol_graph.batch)
        comp_pool = self._encode_smiles_from_graph_attr(mol_graph, graph_global_pool)
        if comp_pool is not None and comp_pool.size(0) == graph_global_pool.size(0):
            comp_graph_feat = self.comp_graph_fusion(torch.cat([graph_global_pool, comp_pool, torch.abs(graph_global_pool - comp_pool), graph_global_pool * comp_pool], dim=-1))
            graph_global_pool = graph_global_pool + torch.sigmoid(self.comp_graph_gate) * (comp_graph_feat - graph_global_pool)
        prot_seq_emb = self.prot_embedding(pro_seq_id)
        prot_mask_f = prot_mask.unsqueeze(-1).to(prot_seq_emb.dtype)
        prot_seq_emb = prot_seq_emb * prot_mask_f
        prot_phys = prot_phys.to(prot_seq_emb.dtype)
        prot_seq_emb = self.pre_mamba_prot(prot_seq_emb) * prot_mask_f
        prot_seq_emb = self.prot_mamba[0](prot_seq_emb, key_padding_mask=~prot_mask)
        prot_seq_emb = prot_seq_emb * prot_mask_f
        if self.use_local_phys_fusion:
            graph_x_fused = self.drug_phys_fusion(graph_x, drug_phys)
            graph_x = graph_x + torch.sigmoid(self.drug_phys_alpha) * (graph_x_fused - graph_x)
            prot_seq_emb_fused = self.prot_phys_fusion(prot_seq_emb, prot_phys)
            prot_seq_emb = prot_seq_emb + torch.sigmoid(self.prot_phys_alpha) * (prot_seq_emb_fused - prot_seq_emb)
            prot_seq_emb = prot_seq_emb * prot_mask_f
        if return_vis:
            if graph_x.requires_grad:
                graph_x.retain_grad()
            if prot_seq_emb.requires_grad:
                prot_seq_emb.retain_grad()
        prot_global_pool = 0.5 * masked_max(prot_seq_emb, prot_mask, dim=1) + 0.5 * masked_mean(prot_seq_emb, prot_mask, dim=1)
        mol_graph_dense_x, mol_graph_mask = to_dense_batch(graph_x, mol_graph.batch)
        drug_phys_dense, _ = to_dense_batch(drug_phys, mol_graph.batch)
        drug_phys_pool = masked_mean(drug_phys_dense, mol_graph_mask, dim=1)
        prot_phys_pool = masked_mean(prot_phys, prot_mask, dim=1)
        self._cached_fusion_phys = (drug_phys_pool, prot_phys_pool)
        zero_scalar = torch.zeros((), device=prot_seq_emb.device, dtype=prot_seq_emb.dtype)
        if self.use_causal_splitter:
            drug_causal_feat, drug_shortcut_feat, drug_causal_pool, drug_shortcut_pool, drug_causal_w, drug_shortcut_w, drug_split_aux = self.drug_causal_splitter(mol_graph_dense_x, mol_graph_mask, phys=drug_phys_dense, ext_conf_id=external_drug_conf_id)
            graph_pool_gate = torch.sigmoid(self.graph_pool_residual_gate)
            drug_causal_pool = drug_causal_pool + graph_pool_gate * (graph_global_pool - drug_causal_pool)
            drug_shortcut_pool = drug_shortcut_pool + graph_pool_gate * (graph_global_pool - drug_shortcut_pool)
            prot_causal_feat, prot_shortcut_feat, prot_causal_pool, prot_shortcut_pool, prot_causal_w, prot_shortcut_w, prot_split_aux = self.prot_causal_splitter(prot_seq_emb, prot_mask, phys=prot_phys, ext_conf_id=external_prot_conf_id)
            prot_pool_gate = torch.sigmoid(self.prot_pool_residual_gate)
            prot_causal_pool = prot_causal_pool + prot_pool_gate * (prot_global_pool - prot_causal_pool)
            prot_shortcut_pool = prot_shortcut_pool + prot_pool_gate * (prot_global_pool - prot_shortcut_pool)
            shortcut_gate = torch.sigmoid(self.shortcut_residual_gate)
            drug_shortcut_scale = self._build_shortcut_scale(drug_split_aux.get('conf_prob', None), external_drug_conf_id, drug_causal_pool)
            prot_shortcut_scale = self._build_shortcut_scale(prot_split_aux.get('conf_prob', None), external_prot_conf_id, prot_causal_pool)
            drug_main_x = drug_causal_pool + shortcut_gate * drug_shortcut_scale * drug_shortcut_pool
            prot_main_x = prot_causal_pool + shortcut_gate * prot_shortcut_scale * prot_shortcut_pool
            if pred_branch == 'hybrid':
                mol_graph_x = drug_main_x
                prot_seq_x = prot_main_x
            elif pred_branch == 'causal':
                mol_graph_x = drug_causal_pool
                prot_seq_x = prot_causal_pool
            else:
                mol_graph_x = drug_shortcut_pool
                prot_seq_x = prot_shortcut_pool
            split_gcl_loss = self.lambda_split_sparse * (drug_split_aux['sparse_loss'] + prot_split_aux['sparse_loss']) + self.lambda_split_orth * (drug_split_aux['orth_loss'] + prot_split_aux['orth_loss']) + self.lambda_split_bias * (drug_split_aux['shortcut_bias_loss'] + prot_split_aux['shortcut_bias_loss']) + self.lambda_split_adv * (drug_split_aux['causal_adv_loss'] + prot_split_aux['causal_adv_loss']) + self.lambda_gate_reg * shortcut_gate.pow(2)
            conf_align_loss = drug_split_aux['conf_align_loss'] + prot_split_aux['conf_align_loss']
        else:
            mol_graph_x = graph_global_pool
            prot_seq_x = masked_max(prot_seq_emb, prot_mask, dim=1)
            split_gcl_loss = zero_scalar
            conf_align_loss = zero_scalar
            drug_split_aux = {'causal_ratio': zero_scalar}
            prot_split_aux = {'causal_ratio': zero_scalar}
            drug_causal_w = None
            drug_shortcut_w = None
            prot_causal_w = None
            prot_shortcut_w = None
            shortcut_gate = torch.sigmoid(self.shortcut_residual_gate)
            drug_causal_pool = mol_graph_x
            drug_shortcut_pool = mol_graph_x
            prot_causal_pool = prot_seq_x
            prot_shortcut_pool = prot_seq_x
            drug_shortcut_scale = torch.ones(drug_causal_pool.size(0), 1, device=drug_causal_pool.device, dtype=drug_causal_pool.dtype)
            prot_shortcut_scale = torch.ones(prot_causal_pool.size(0), 1, device=prot_causal_pool.device, dtype=prot_causal_pool.dtype)
            drug_main_x = mol_graph_x
            prot_main_x = prot_seq_x
        if return_vis:
            pred, pred_vis = self._predict_from_parts(mol_graph_x, prot_seq_x, return_vis=True)
        else:
            pred = self._predict_from_parts(mol_graph_x, prot_seq_x)
            pred_vis = None
        if return_aux and self.use_causal_splitter:
            pred_causal = self._predict_from_parts(drug_causal_pool, prot_causal_pool)
            if compute_cf:
                drug_perm = self._match_counterfactual_perm(drug_causal_pool, drug_split_aux['conf_target'])
                prot_perm = self._match_counterfactual_perm(prot_causal_pool, prot_split_aux['conf_target'])
                alpha_cf = 0.5
                drug_shortcut_cf = drug_shortcut_pool[drug_perm].detach()
                drug_shortcut_mix = (1.0 - alpha_cf) * drug_shortcut_pool + alpha_cf * drug_shortcut_cf
                pred_cf_drug = self._predict_from_parts(drug_causal_pool + shortcut_gate * drug_shortcut_scale * drug_shortcut_mix, prot_main_x)
                prot_shortcut_cf = prot_shortcut_pool[prot_perm].detach()
                prot_shortcut_mix = (1.0 - alpha_cf) * prot_shortcut_pool + alpha_cf * prot_shortcut_cf
                pred_cf_prot = self._predict_from_parts(drug_main_x, prot_causal_pool + shortcut_gate * prot_shortcut_scale * prot_shortcut_mix)
            else:
                pred_cf_drug = pred.detach()
                pred_cf_prot = pred.detach()
        else:
            pred_causal = pred
            pred_cf_drug = pred
            pred_cf_prot = pred
        if return_details:
            details = {'drug_causal_w': drug_causal_w, 'drug_shortcut_w': drug_shortcut_w, 'prot_causal_w': prot_causal_w, 'prot_shortcut_w': prot_shortcut_w, 'drug_mask': mol_graph_mask, 'prot_mask': prot_mask, 'drug_phys': drug_phys_dense, 'prot_phys': prot_phys, 'pred_branch': pred_branch, 'phys_mode': phys_mode, 'shortcut_gate': shortcut_gate.detach(), 'drug_shortcut_scale': drug_shortcut_scale.detach() if drug_shortcut_scale is not None else None, 'prot_shortcut_scale': prot_shortcut_scale.detach() if prot_shortcut_scale is not None else None, 'model_variant': self.model_variant, 'drug_conf_target': drug_split_aux.get('conf_target', None), 'prot_conf_target': prot_split_aux.get('conf_target', None), 'drug_conf_prob': drug_split_aux.get('conf_prob', None), 'prot_conf_prob': prot_split_aux.get('conf_prob', None), 'external_drug_conf_id': external_drug_conf_id, 'external_prot_conf_id': external_prot_conf_id, 'drug_causal_pool': drug_causal_pool.detach(), 'drug_shortcut_pool': drug_shortcut_pool.detach(), 'prot_causal_pool': prot_causal_pool.detach(), 'prot_shortcut_pool': prot_shortcut_pool.detach(), 'drug_main_pool': drug_main_x.detach(), 'prot_main_pool': prot_main_x.detach(), 'drug_phys_pool': drug_phys_pool.detach(), 'prot_phys_pool': prot_phys_pool.detach(), 'comp_pool': comp_pool.detach() if comp_pool is not None else None}
        if return_vis:
            vis = {'raw_drug_x': raw_drug_x, 'prot_seq_id': pro_seq_id, 'drug_mask': mol_graph_mask, 'prot_mask': prot_mask, 'graph_x': graph_x, 'mol_graph_dense_x': mol_graph_dense_x, 'prot_seq_emb': prot_seq_emb, 'drug_causal_w': drug_causal_w, 'drug_shortcut_w': drug_shortcut_w, 'prot_causal_w': prot_causal_w, 'prot_shortcut_w': prot_shortcut_w, 'drug_causal_pool': drug_causal_pool, 'drug_shortcut_pool': drug_shortcut_pool, 'prot_causal_pool': prot_causal_pool, 'prot_shortcut_pool': prot_shortcut_pool, 'drug_main_pool': drug_main_x, 'prot_main_pool': prot_main_x, 'drug_phys_dense': drug_phys_dense, 'prot_phys': prot_phys, 'drug_phys_pool': drug_phys_pool, 'prot_phys_pool': prot_phys_pool, 'pred_branch': pred_branch, 'phys_mode': phys_mode, 'shortcut_gate': shortcut_gate, 'drug_shortcut_scale': drug_shortcut_scale, 'prot_shortcut_scale': prot_shortcut_scale, 'pred_vis': pred_vis}
            return (pred, vis)
        if return_aux:
            aux = {'gcl_loss': split_gcl_loss, 'conf_align_loss': conf_align_loss, 'drug_causal_ratio': drug_split_aux['causal_ratio'], 'prot_causal_ratio': prot_split_aux['causal_ratio'], 'comp_causal_ratio': zero_scalar, 'pred_causal': pred_causal, 'pred_cf_drug': pred_cf_drug, 'pred_cf_prot': pred_cf_prot}
            if return_details:
                return (pred, aux, details)
            return (pred, aux)
        if return_details:
            return (pred, details)
        return pred

    @torch.no_grad()
    def predict_e4_ensemble(self, pro_seq_id, mol_graph, weights=None, configs=None, custom_drug_phys=None, custom_prot_phys=None, external_drug_conf_id=None, external_prot_conf_id=None):
        if configs is None:
            configs = [('hybrid', 'normal'), ('causal', 'normal'), ('hybrid', 'zero'), ('causal', 'zero')]
        if weights is None:
            weights = [0.45, 0.35, 0.1, 0.1]
        if len(configs) != len(weights):
            raise ValueError('configs and weights must have the same length')
        weights_t = torch.tensor(weights, device=pro_seq_id.device, dtype=torch.float32)
        weights_t = weights_t / weights_t.sum().clamp_min(1e-08)
        was_training = self.training
        self.eval()
        out = None
        for i, (branch, mode) in enumerate(configs):
            logit = self.forward(pro_seq_id, mol_graph, return_aux=False, return_details=False, pred_branch=branch, phys_mode=mode, custom_drug_phys=custom_drug_phys, custom_prot_phys=custom_prot_phys, external_drug_conf_id=external_drug_conf_id, external_prot_conf_id=external_prot_conf_id)
            out = logit * weights_t[i] if out is None else out + logit * weights_t[i]
        if was_training:
            self.train()
        return out

class Feature_Fusion(nn.Module):

    def __init__(self, channels, r=3):
        super(Feature_Fusion, self).__init__()
        inter_channels = int(channels * r)
        layers = [nn.Linear(channels * 2, inter_channels), nn.GELU(), nn.Linear(inter_channels, channels)]
        self.att1 = nn.Sequential(*layers)
        self.att2 = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fd, fp):
        concat = torch.cat((fd, fp), 1)
        w1 = self.sigmoid(self.att1(concat))
        fout1 = torch.cat((fd * w1, fp * (1 - w1)), 1)
        w2 = self.sigmoid(self.att2(fout1))
        fout2 = torch.cat((fd * w2, fp * (1 - w2)), 1)
        return fout2
