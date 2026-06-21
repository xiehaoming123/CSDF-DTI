import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import MambaBlock, MambaConfig
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, GINEConv
from torch_geometric.nn.norm import GraphNorm

def masked_max(x, mask, dim=1):
    y = x.masked_fill(~mask.bool().unsqueeze(-1), float('-inf'))
    out = y.max(dim=dim).values
    out[out == float('-inf')] = 0
    return out

def masked_mean(x, mask, dim=1, eps=1e-08):
    m = mask.bool().unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(eps)

def weighted_masked_pool(x, weight, mask, eps=1e-08):
    w = weight * mask.bool().unsqueeze(-1).to(x.dtype)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
    return (x * w).sum(dim=1)

def reverse_valid(x, mask):
    if mask is None:
        return x.flip(1)
    mask = mask.bool()
    bsz, length, dim = x.shape
    sizes = mask.long().sum(dim=1)
    idx = torch.arange(length, device=x.device).view(1, length).expand(bsz, length)
    rev = (sizes.view(bsz, 1) - 1 - idx).clamp(min=0)
    return x.gather(1, rev.unsqueeze(-1).expand(bsz, length, dim)) * mask.unsqueeze(-1).to(x.dtype)

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

def build_protein_physchem_table():
    return torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [-0.5, 0.5, 1, 0, 0], [-1, 0, 1, 0, 0], [-1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 1, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0], [-0.5, 0.5, 1, 0, 0]], dtype=torch.float32)

class BidirMambaBlock(nn.Module):

    def __init__(self, dim, weight_tie=True):
        super().__init__()
        config = MambaConfig(d_model=dim, n_layers=1)
        self.fwd = MambaBlock(config)
        self.bwd = MambaBlock(config)
        self.norm = nn.LayerNorm(dim)
        if weight_tie:
            self.bwd.in_proj.weight = self.fwd.in_proj.weight
            self.bwd.in_proj.bias = self.fwd.in_proj.bias
            self.bwd.out_proj.weight = self.fwd.out_proj.weight
            self.bwd.out_proj.bias = self.fwd.out_proj.bias

    def forward(self, x, mask=None):
        if mask is None:
            return self.norm(x + self.fwd(x) + self.bwd(x.flip(1)).flip(1))
        m = mask.bool().unsqueeze(-1).to(x.dtype)
        x = x * m
        return self.norm(x + self.fwd(x) * m + reverse_valid(self.bwd(reverse_valid(x, mask)), mask) * m) * m

class SequenceBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, max(1, dim // 32), batch_first=True)
        self.mamba = BidirMambaBlock(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        x = self.norm(x + self.attn(x, x, x, key_padding_mask=key_padding_mask)[0])
        return self.mamba(x, mask=None if key_padding_mask is None else ~key_padding_mask)

class CausalShortcutSplitter(nn.Module):

    def __init__(self, hidden_dim, phys_dim=5, dropout=0.1, num_confounders=5, grl_lambda=1.0, target_causal_ratio=0.7):
        super().__init__()
        mid = hidden_dim // 2
        self.phys_dim = phys_dim
        self.num_confounders = num_confounders
        self.grl_lambda = grl_lambda
        self.target_causal_ratio = target_causal_ratio
        self.conf_assign_head = nn.Sequential(nn.Linear(phys_dim, mid), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid, num_confounders))
        self.confounder_token_bank = nn.Parameter(torch.randn(num_confounders, hidden_dim) * 0.02)
        self.shortcut_gate_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.shortcut_bias_classifier = nn.Sequential(nn.Linear(hidden_dim, mid), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid, num_confounders))
        self.causal_adv_classifier = nn.Sequential(nn.Linear(hidden_dim, mid), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid, num_confounders))

    def forward(self, x, mask, phys=None, ext_conf_id=None):
        mask = mask.bool()
        if phys is None:
            phys = x.new_zeros(x.size(0), x.size(1), self.phys_dim)
        conf_logits_raw = self.conf_assign_head(phys)
        conf_logits = conf_logits_raw.masked_fill(~mask.unsqueeze(-1), -10000.0)
        conf_prob = F.softmax(conf_logits, dim=-1)
        conf_context = torch.matmul(conf_prob, self.confounder_token_bank)
        conf_align_loss = x.new_zeros(())
        valid_ext_mask = None
        if ext_conf_id is not None:
            ext_conf_id = ext_conf_id.to(x.device).long().view(-1)
            if ext_conf_id.size(0) != x.size(0):
                raise ValueError(f'ext_conf_id batch size mismatch: {tuple(ext_conf_id.shape)} vs {tuple(x.shape)}')
            if torch.any((ext_conf_id < -1) | (ext_conf_id >= self.num_confounders)):
                raise ValueError(f'Invalid ext_conf_id. Allowed range is [0, {self.num_confounders - 1}] or -1.')
            valid_ext_mask = ext_conf_id >= 0
            if valid_ext_mask.any():
                safe_id = ext_conf_id.clamp(min=0)
                ext_context = self.confounder_token_bank[safe_id].unsqueeze(1).expand_as(conf_context)
                conf_context = torch.where(valid_ext_mask.view(-1, 1, 1), 0.7 * ext_context + 0.3 * conf_context, conf_context)
                conf_align_loss = F.cross_entropy(masked_mean(conf_logits_raw, mask, 1)[valid_ext_mask], ext_conf_id[valid_ext_mask])
        shortcut_score = self.shortcut_gate_head(torch.cat([x, conf_context], dim=-1)).masked_fill(~mask.unsqueeze(-1), -10000.0)
        shortcut_w = torch.sigmoid(shortcut_score) * mask.unsqueeze(-1).to(x.dtype)
        causal_w = (1.0 - shortcut_w) * mask.unsqueeze(-1).to(x.dtype)
        causal_pool = weighted_masked_pool(x, causal_w, mask)
        shortcut_pool = weighted_masked_pool(x, shortcut_w, mask)
        conf_pool = masked_mean(conf_prob, mask, 1)
        internal_target = conf_pool.detach().argmax(dim=-1)
        if ext_conf_id is None:
            conf_target, conf_prob_aux = (internal_target, conf_pool)
        else:
            valid_ext_mask = ext_conf_id >= 0 if valid_ext_mask is None else valid_ext_mask
            conf_target = torch.where(valid_ext_mask, ext_conf_id, internal_target)
            conf_prob_aux = conf_pool.clone()
            if valid_ext_mask.any():
                conf_prob_aux[valid_ext_mask] = F.one_hot(ext_conf_id[valid_ext_mask], self.num_confounders).float()
        shortcut_loss = F.cross_entropy(self.shortcut_bias_classifier(shortcut_pool), conf_target)
        adv_loss = F.cross_entropy(self.causal_adv_classifier(grad_reverse(causal_pool, self.grl_lambda)), conf_target)
        causal_ratio = causal_w.squeeze(-1).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
        aux = {'causal_ratio': causal_ratio.mean().detach(), 'sparse_loss': (causal_ratio - self.target_causal_ratio).pow(2).mean(), 'orth_loss': F.cosine_similarity(causal_pool, shortcut_pool, dim=-1).pow(2).mean(), 'shortcut_bias_loss': shortcut_loss, 'causal_adv_loss': adv_loss, 'conf_align_loss': conf_align_loss, 'conf_target': conf_target.detach(), 'conf_prob': conf_prob_aux.detach()}
        return (x * causal_w, x * shortcut_w, causal_pool, shortcut_pool, causal_w, shortcut_w, aux)

class EdgeAwareGINEEncoder(nn.Module):

    def __init__(self, in_dim=26, hidden_dim=256, edge_dim=14, num_layers=4, dropout=0.15):
        super().__init__()
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.virtual_node = nn.Parameter(torch.zeros(1, hidden_dim))
        self.virtual_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)), edge_dim=edge_dim))
            self.norms.append(GraphNorm(hidden_dim))
        self.out_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.out_norm = GraphNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_proj(x)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if num_graphs > 0:
            h = h + self.virtual_node.expand(num_graphs, -1)[batch]
        if edge_attr is None:
            edge_attr = h.new_zeros(edge_index.size(1), self.edge_dim)
        else:
            edge_attr = edge_attr.to(device=h.device, dtype=h.dtype)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
        for conv, norm in zip(self.layers, self.norms):
            h = norm(F.dropout(F.gelu(conv(h, edge_index, edge_attr)), p=self.dropout, training=self.training) + h, batch)
            if num_graphs > 0:
                h = h + self.virtual_mlp(global_max_pool(h, batch))[batch]
        return self.out_norm(h + self.out_mlp(h), batch)

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
        mean_pool = x.new_zeros(num_graphs, x.size(-1)).index_add(0, batch, x)
        counts = torch.bincount(batch, minlength=num_graphs).to(x.device).to(x.dtype).clamp_min(1.0)
        mean_pool = mean_pool / counts.unsqueeze(-1)
        score = self.attn(x)
        weight = torch.exp(score - global_max_pool(score, batch)[batch])
        denom = x.new_zeros(num_graphs, 1).index_add(0, batch, weight).clamp_min(1e-08)
        attn_pool = x.new_zeros(num_graphs, x.size(-1)).index_add(0, batch, x * weight) / denom
        return self.proj(torch.cat([max_pool, mean_pool, attn_pool], dim=-1))

class MambaCPAModelWoPretrained(nn.Module):

    def __init__(self):
        super().__init__()
        h = 256
        config = MambaConfig(d_model=h, n_layers=1)
        self.phys_dim = 5
        self.num_confounders = 5
        self.smiles_vocab_size = 128
        self.prot_embedding = nn.Embedding(26, h, padding_idx=0)
        self.comp_embedding = nn.Embedding(self.smiles_vocab_size, h, padding_idx=0)
        self.pre_mamba_prot = MambaBlock(config)
        self.pre_mamba_comp = MambaBlock(config)
        self.prot_mamba = nn.Sequential(SequenceBlock(h))
        self.comp_mamba = nn.Sequential(SequenceBlock(h))
        self.mol_graph_net1 = EdgeAwareGINEEncoder(26, h, 14, 4, 0.15)
        self.graph_multi_pool = GraphMultiPool(h, 0.1)
        self.comp_graph_fusion = nn.Sequential(nn.LayerNorm(h * 4), nn.Linear(h * 4, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.LayerNorm(h))
        self.register_buffer('protein_physchem_table', build_protein_physchem_table())
        self.drug_phys_proj = nn.Sequential(nn.Linear(26, h), nn.GELU(), nn.Linear(h, self.phys_dim), nn.Tanh())
        self.drug_causal_splitter = CausalShortcutSplitter(h, self.phys_dim, 0.1, self.num_confounders, 1.0, 0.7)
        self.prot_causal_splitter = CausalShortcutSplitter(h, self.phys_dim, 0.1, self.num_confounders, 1.2, 0.8)
        self.fusion_head = nn.Sequential(nn.LayerNorm(h * 4 + self.phys_dim * 3), nn.Linear(h * 4 + self.phys_dim * 3, h * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(h * 2, h * 2), nn.LayerNorm(h * 2))
        self.pred_net = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, 1024), nn.GELU(), nn.Dropout(0.15), nn.LayerNorm(1024), nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.15), nn.LayerNorm(512), nn.Linear(512, 1))
        self.pair_logit_head = nn.Sequential(nn.LayerNorm(4 * h), nn.Linear(4 * h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))
        self.graph_pool_residual_gate = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))
        self.comp_graph_gate = nn.Parameter(torch.tensor(math.log(0.25 / 0.75), dtype=torch.float32))
        self.prot_pool_residual_gate = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))
        self.shortcut_residual_gate = nn.Parameter(torch.tensor(math.log(0.03 / 0.97), dtype=torch.float32))
        self.pair_logit_scale = nn.Parameter(torch.tensor(math.log(0.08 / 0.92), dtype=torch.float32))
        self.lambda_split_sparse = 0.002
        self.lambda_split_orth = 0.0005
        self.lambda_split_bias = 0.002
        self.lambda_split_adv = 0.012
        self.lambda_gate_reg = 0.008
        self.external_conf_dropout_drug = 0.65
        self.external_conf_dropout_prot = 0.7
        self.reliability_seen_weight = 0.25
        self.reliability_conf_weight = 0.75
        self.enable_tapb_seq_randomization = True
        self.prot_mask_prob = 0.0
        self.prot_mut_prob = 0.08
        self.comp_mask_prob = 0.0
        self.comp_mut_prob = 0.0
        self.external_conf_dropout_comp = 0.0
        self.use_causal_splitter = True
        self.use_local_phys_fusion = False
        self.model_variant = {'drug_input': 'graph_plus_smiles', 'protein_encoder': 'mamba', 'drug_encoder': 'edge_aware_gine', 'fusion': 'phys_guided_pair'}

    def get_protein_physchem_prior(self, pro_seq_id):
        return self.protein_physchem_table[pro_seq_id]

    def apply_phys_mode(self, drug_phys, prot_phys, drug_batch, prot_mask, phys_mode='normal'):
        if phys_mode == 'normal':
            return (drug_phys, prot_phys)
        if phys_mode == 'zero':
            return (torch.zeros_like(drug_phys), torch.zeros_like(prot_phys))
        raise ValueError(f'Unsupported phys_mode: {phys_mode}')

    def _tapb_randomize_ids(self, seq_id, vocab_high, mask_prob=0.0, mutate_prob=0.0):
        if not self.training or not self.enable_tapb_seq_randomization:
            return seq_id
        out = seq_id.clone()
        valid = out != 0
        if mask_prob > 0:
            drop = valid & (torch.rand(out.shape, device=out.device) < mask_prob)
            out[drop] = 0
            valid = out != 0
        if mutate_prob > 0:
            mutate = valid & (torch.rand(out.shape, device=out.device) < mutate_prob)
            if mutate.any():
                out[mutate] = torch.randint(1, vocab_high + 1, (int(mutate.sum().item()),), device=out.device).to(out.dtype)
        return out

    def _maybe_dropout_external_conf(self, ext_conf_id, drop_prob=0.0):
        if ext_conf_id is None or not self.training or drop_prob <= 0:
            return ext_conf_id
        out = ext_conf_id.clone()
        valid = out >= 0
        out[valid & (torch.rand(out.shape, device=out.device) < drop_prob)] = -1
        return out

    def _encode_smiles_from_graph_attr(self, mol_graph, ref_tensor):
        smiles_id = getattr(mol_graph, 'smiles_id', None)
        if smiles_id is None:
            return None
        smiles_id = smiles_id.to(device=ref_tensor.device).long().clamp(0, self.smiles_vocab_size - 1)
        if smiles_id.dim() == 1:
            smiles_id = smiles_id.unsqueeze(0)
        mask = smiles_id != 0
        mask_f = mask.unsqueeze(-1).to(ref_tensor.dtype)
        x = self.comp_embedding(smiles_id) * mask_f
        x = self.pre_mamba_comp(x) * mask_f
        x = self.comp_mamba[0](x, key_padding_mask=~mask) * mask_f
        return 0.5 * masked_max(x, mask, dim=1) + 0.5 * masked_mean(x, mask, dim=1)

    def _conf_reliability(self, conf_prob):
        if conf_prob is None:
            return None
        conf_prob = conf_prob.clamp_min(1e-08)
        entropy = -(conf_prob * conf_prob.log()).sum(dim=-1) / math.log(conf_prob.size(-1))
        return (1.0 - entropy).clamp(0.0, 1.0)

    def _build_shortcut_scale(self, conf_prob, ext_conf_id, ref_tensor):
        seen = torch.zeros(ref_tensor.size(0), device=ref_tensor.device, dtype=ref_tensor.dtype)
        if ext_conf_id is not None:
            seen = (ext_conf_id.to(ref_tensor.device) >= 0).to(ref_tensor.dtype)
        conf = self._conf_reliability(conf_prob)
        conf = torch.ones_like(seen) if conf is None else conf.to(ref_tensor.dtype)
        return (self.reliability_seen_weight * seen + self.reliability_conf_weight * conf).clamp(0.05, 1.0).unsqueeze(-1)

    def _predict_from_parts(self, drug_x, prot_x, drug_phys_pool, prot_phys_pool):
        pair = torch.cat([drug_x, prot_x, torch.abs(drug_x - prot_x), drug_x * prot_x], dim=-1)
        phys_pair = torch.cat([drug_phys_pool, prot_phys_pool, torch.abs(drug_phys_pool - prot_phys_pool)], dim=-1)
        z = self.fusion_head(torch.cat([pair, phys_pair], dim=-1))
        return self.pred_net(z) + torch.sigmoid(self.pair_logit_scale) * self.pair_logit_head(pair)

    def forward(self, pro_seq_id, mol_graph, return_aux=False, return_details=False, pred_branch='hybrid', phys_mode='normal', custom_drug_phys=None, custom_prot_phys=None, external_drug_conf_id=None, external_prot_conf_id=None, compute_cf=True, return_vis=False):
        if pred_branch not in ['hybrid', 'causal', 'shortcut']:
            raise ValueError(f'Unsupported pred_branch: {pred_branch}')
        raw_drug_x = mol_graph.x
        prot_mask = pro_seq_id != 0
        pro_seq_id = self._tapb_randomize_ids(pro_seq_id, 25, self.prot_mask_prob, self.prot_mut_prob)
        external_drug_conf_id = self._maybe_dropout_external_conf(external_drug_conf_id, self.external_conf_dropout_drug)
        external_prot_conf_id = self._maybe_dropout_external_conf(external_prot_conf_id, self.external_conf_dropout_prot)
        drug_phys = self.drug_phys_proj(raw_drug_x) if custom_drug_phys is None else custom_drug_phys.to(raw_drug_x.device, dtype=raw_drug_x.dtype)
        prot_phys = self.get_protein_physchem_prior(pro_seq_id).to(raw_drug_x.dtype) if custom_prot_phys is None else custom_prot_phys.to(pro_seq_id.device, dtype=raw_drug_x.dtype)
        drug_phys, prot_phys = self.apply_phys_mode(drug_phys, prot_phys, mol_graph.batch, prot_mask, phys_mode)
        graph_x = self.mol_graph_net1(raw_drug_x, mol_graph.edge_index, getattr(mol_graph, 'edge_attr', None), mol_graph.batch)
        graph_pool = self.graph_multi_pool(graph_x, mol_graph.batch)
        comp_pool = self._encode_smiles_from_graph_attr(mol_graph, graph_pool)
        if comp_pool is not None and comp_pool.size(0) == graph_pool.size(0):
            comp_feat = self.comp_graph_fusion(torch.cat([graph_pool, comp_pool, torch.abs(graph_pool - comp_pool), graph_pool * comp_pool], dim=-1))
            graph_pool = graph_pool + torch.sigmoid(self.comp_graph_gate) * (comp_feat - graph_pool)
        prot_emb = self.prot_embedding(pro_seq_id)
        prot_mask_f = prot_mask.unsqueeze(-1).to(prot_emb.dtype)
        prot_emb = self.pre_mamba_prot(prot_emb * prot_mask_f) * prot_mask_f
        prot_emb = self.prot_mamba[0](prot_emb, key_padding_mask=~prot_mask) * prot_mask_f
        prot_pool = 0.5 * masked_max(prot_emb, prot_mask, 1) + 0.5 * masked_mean(prot_emb, prot_mask, 1)
        graph_dense, graph_mask = to_dense_batch(graph_x, mol_graph.batch)
        drug_phys_dense, _ = to_dense_batch(drug_phys, mol_graph.batch)
        drug_phys_pool = masked_mean(drug_phys_dense, graph_mask, 1)
        prot_phys_pool = masked_mean(prot_phys, prot_mask, 1)
        zero = prot_emb.new_zeros(())
        if self.use_causal_splitter:
            _, _, drug_causal, drug_shortcut, drug_causal_w, drug_shortcut_w, drug_aux = self.drug_causal_splitter(graph_dense, graph_mask, drug_phys_dense, external_drug_conf_id)
            _, _, prot_causal, prot_shortcut, prot_causal_w, prot_shortcut_w, prot_aux = self.prot_causal_splitter(prot_emb, prot_mask, prot_phys, external_prot_conf_id)
            drug_causal = drug_causal + torch.sigmoid(self.graph_pool_residual_gate) * (graph_pool - drug_causal)
            drug_shortcut = drug_shortcut + torch.sigmoid(self.graph_pool_residual_gate) * (graph_pool - drug_shortcut)
            prot_causal = prot_causal + torch.sigmoid(self.prot_pool_residual_gate) * (prot_pool - prot_causal)
            prot_shortcut = prot_shortcut + torch.sigmoid(self.prot_pool_residual_gate) * (prot_pool - prot_shortcut)
            shortcut_gate = torch.sigmoid(self.shortcut_residual_gate)
            drug_scale = self._build_shortcut_scale(drug_aux.get('conf_prob'), external_drug_conf_id, drug_causal)
            prot_scale = self._build_shortcut_scale(prot_aux.get('conf_prob'), external_prot_conf_id, prot_causal)
            drug_main = drug_causal + shortcut_gate * drug_scale * drug_shortcut
            prot_main = prot_causal + shortcut_gate * prot_scale * prot_shortcut
            drug_out, prot_out = (drug_causal, prot_causal) if pred_branch == 'causal' else (drug_shortcut, prot_shortcut) if pred_branch == 'shortcut' else (drug_main, prot_main)
            gcl_loss = self.lambda_split_sparse * (drug_aux['sparse_loss'] + prot_aux['sparse_loss']) + self.lambda_split_orth * (drug_aux['orth_loss'] + prot_aux['orth_loss']) + self.lambda_split_bias * (drug_aux['shortcut_bias_loss'] + prot_aux['shortcut_bias_loss']) + self.lambda_split_adv * (drug_aux['causal_adv_loss'] + prot_aux['causal_adv_loss']) + self.lambda_gate_reg * shortcut_gate.pow(2)
            conf_align_loss = drug_aux['conf_align_loss'] + prot_aux['conf_align_loss']
        else:
            drug_out, prot_out = (graph_pool, prot_pool)
            drug_causal = drug_shortcut = drug_out
            prot_causal = prot_shortcut = prot_out
            drug_aux = prot_aux = {'causal_ratio': zero}
            gcl_loss = conf_align_loss = zero
            shortcut_gate = torch.sigmoid(self.shortcut_residual_gate)
        pred = self._predict_from_parts(drug_out, prot_out, drug_phys_pool, prot_phys_pool)
        if return_aux:
            aux = {'gcl_loss': gcl_loss, 'conf_align_loss': conf_align_loss, 'drug_causal_ratio': drug_aux['causal_ratio'], 'prot_causal_ratio': prot_aux['causal_ratio'], 'comp_causal_ratio': zero, 'pred_causal': self._predict_from_parts(drug_causal, prot_causal, drug_phys_pool, prot_phys_pool), 'pred_cf_drug': pred.detach(), 'pred_cf_prot': pred.detach()}
            if return_details:
                return (pred, aux, {'pred_branch': pred_branch, 'phys_mode': phys_mode, 'shortcut_gate': shortcut_gate.detach()})
            return (pred, aux)
        if return_details:
            return (pred, {'pred_branch': pred_branch, 'phys_mode': phys_mode, 'shortcut_gate': shortcut_gate.detach()})
        return pred
