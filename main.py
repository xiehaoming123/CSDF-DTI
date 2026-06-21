import os
import json
import random
import argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from tqdm import tqdm

from dataset import ProtDrugSeqDatasetCLS
from early_stop import EarlyStopping
from model import MambaCPAModelWoPretrained


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_column(df, candidates, name):
    for col in candidates:
        if col in df.columns:
            return col
    raise RuntimeError(f"Cannot infer {name} column from {list(df.columns)}")


def infer_columns(df):
    drug_col = find_column(
        df,
        ["SMILES", "smiles", "Smiles", "compound_iso_smiles", "compound_smiles", "drug", "Drug", "drug_smiles", "compound"],
        "drug",
    )
    target_col = find_column(
        df,
        ["Protein", "protein", "Target", "target", "target_sequence", "protein_sequence", "sequence", "Sequence"],
        "target",
    )
    label_col = find_column(
        df,
        ["Y", "y", "label", "Label", "labels", "interaction", "Interaction"],
        "label",
    )
    return drug_col, target_col, label_col


def prior_table(df, entity_col, label_col):
    out = (
        df.groupby(entity_col)[label_col]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={entity_col: "entity", "count": "n", "sum": "pos"})
    )
    out["z"] = out["pos"] / out["n"].clip(lower=1)
    return out


def bin_prior(x):
    return pd.cut(
        x,
        bins=[0.0, 0.05, 0.25, 0.75, 0.95, 1.0],
        labels=False,
        include_lowest=True,
    )


def attach_prior_bins(df, drug_col, target_col, drug_prior, target_prior, unknown_bin=-1):
    out = df.copy()
    drug_map = drug_prior[["entity", "z"]].rename(columns={"entity": drug_col, "z": "drug_prior"})
    target_map = target_prior[["entity", "z"]].rename(columns={"entity": target_col, "z": "target_prior"})
    out = out.merge(drug_map, on=drug_col, how="left")
    out = out.merge(target_map, on=target_col, how="left")
    out["drug_prior_bin"] = bin_prior(out["drug_prior"]).fillna(unknown_bin).astype(int)
    out["target_prior_bin"] = bin_prior(out["target_prior"]).fillna(unknown_bin).astype(int)
    out["comp_prior_bin"] = out["drug_prior_bin"].astype(int)
    return out


class ConfounderDataset(Dataset):
    def __init__(self, base_dataset, frame):
        self.base_dataset = base_dataset
        self.frame = frame.reset_index(drop=True)
        self.drug_ids = self.frame["drug_prior_bin"].astype(int).to_numpy()
        self.prot_ids = self.frame["target_prior_bin"].astype(int).to_numpy()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        return item[0], item[1], item[2], item[3], int(self.drug_ids[index]), int(self.prot_ids[index])

    def collate_fn(self, batch):
        base_batch = [(x[0], x[1], x[2], x[3]) for x in batch]
        pro_seq_id, smiles_id, mol_graph, label = self.base_dataset.collate_fn(base_batch)
        mol_graph.smiles_id = smiles_id
        drug_conf_id = torch.tensor([x[4] for x in batch], dtype=torch.long)
        prot_conf_id = torch.tensor([x[5] for x in batch], dtype=torch.long)
        return pro_seq_id, mol_graph, label, drug_conf_id, prot_conf_id


def read_split(data_dir):
    train_path = data_dir / "train.csv"
    valid_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    for path in [train_path, valid_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(train_path), pd.read_csv(valid_path), pd.read_csv(test_path)


def prepare_frames(train_df, valid_df, test_df):
    drug_col, target_col, label_col = infer_columns(train_df)
    for frame in [train_df, valid_df, test_df]:
        frame[label_col] = pd.to_numeric(frame[label_col], errors="raise").astype(int)
    drug_prior = prior_table(train_df, drug_col, label_col)
    target_prior = prior_table(train_df, target_col, label_col)
    train_df = attach_prior_bins(train_df, drug_col, target_col, drug_prior, target_prior)
    valid_df = attach_prior_bins(valid_df, drug_col, target_col, drug_prior, target_prior)
    test_df = attach_prior_bins(test_df, drug_col, target_col, drug_prior, target_prior)
    return train_df, valid_df, test_df, label_col


def make_loaders(train_df, valid_df, test_df, label_col, batch_size, eval_batch_size, num_workers, balanced):
    train_set = ConfounderDataset(ProtDrugSeqDatasetCLS(train_df), train_df)
    valid_set = ConfounderDataset(ProtDrugSeqDatasetCLS(valid_df), valid_df)
    test_set = ConfounderDataset(ProtDrugSeqDatasetCLS(test_df), test_df)

    sampler = None
    if balanced:
        labels = train_df[label_col].astype(int).to_numpy()
        pos = max(1, int((labels == 1).sum()))
        neg = max(1, int((labels == 0).sum()))
        weights = np.where(labels == 1, 1.0 / pos, 1.0 / neg)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    loader_args = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_args.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
        collate_fn=train_set.collate_fn,
        **loader_args,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=valid_set.collate_fn,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn,
        **loader_args,
    )
    return train_loader, valid_loader, test_loader


def sanitize_logits(logits):
    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    return logits.clamp(min=-20.0, max=20.0)


def sanitize_probs(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(values, 1e-7, 1.0 - 1e-7)


def binary_metrics(labels, probs, threshold=None):
    labels = np.asarray(labels).reshape(-1)
    probs = sanitize_probs(probs)

    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    if threshold is None:
        precision_curve, recall_curve, thresholds = precision_recall_curve(labels, probs)
        if len(thresholds) == 0:
            threshold = 0.5
        else:
            f1_curve = 2 * precision_curve[:-1] * recall_curve[:-1] / (precision_curve[:-1] + recall_curve[:-1] + 1e-10)
            threshold = float(thresholds[int(np.argmax(f1_curve))]) if len(f1_curve) else 0.5

    pred = (probs >= threshold).astype(np.int32)
    acc = accuracy_score(labels, pred)
    precision = precision_score(labels, pred, zero_division=0)
    recall = recall_score(labels, pred, zero_division=0)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    mcc = matthews_corrcoef(labels, pred)
    return dict(auroc=auroc, auprc=auprc, f1=f1, acc=acc, mcc=mcc, threshold=float(threshold))


def rank_loss(logits, labels, margin=0.15, tau=0.15, max_pairs=4096):
    logits = logits.view(-1)
    labels = labels.float().view(-1)
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    diff = pos[:, None] - neg[None, :]
    if diff.numel() > max_pairs:
        flat = diff.reshape(-1)
        index = torch.randint(0, flat.numel(), (max_pairs,), device=flat.device)
        diff = flat[index]
    return F.softplus((margin - diff) / tau).mean()


def move_batch(batch, device):
    pro_seq_id, mol_graph, label, drug_conf_id, prot_conf_id = batch
    return (
        pro_seq_id.to(device),
        mol_graph.to(device),
        label.to(device),
        drug_conf_id.to(device),
        prot_conf_id.to(device),
    )


@torch.no_grad()
def evaluate(model, loader, device, branch="hybrid", phys_mode="normal", threshold=None):
    model.eval()
    labels, probs = [], []
    for batch in loader:
        pro_seq_id, mol_graph, label, drug_conf_id, prot_conf_id = move_batch(batch, device)
        logits = model(
            pro_seq_id,
            mol_graph,
            pred_branch=branch,
            phys_mode=phys_mode,
            external_drug_conf_id=drug_conf_id,
            external_prot_conf_id=prot_conf_id,
        )
        logits = sanitize_logits(logits)
        probs.append(torch.sigmoid(logits).view(-1).cpu().numpy())
        labels.append(label.view(-1).cpu().numpy())
    model.train()
    return binary_metrics(np.concatenate(labels), np.concatenate(probs), threshold)


def train_epoch(model, loader, optimizer, scaler, criterion, device, args):
    model.train()
    total_loss = 0.0
    total_count = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for batch in tqdm(loader, leave=False):
        pro_seq_id, mol_graph, label, drug_conf_id, prot_conf_id = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(autocast_device):
            logits, aux = model(
                pro_seq_id,
                mol_graph,
                return_aux=True,
                pred_branch="hybrid",
                phys_mode=args.train_phys_mode,
                external_drug_conf_id=drug_conf_id,
                external_prot_conf_id=prot_conf_id,
                compute_cf=args.lambda_cf > 0,
            )

            labels = label.view(-1).float()
            main_loss = criterion(logits.view(-1), labels).mean()
            causal_loss = criterion(aux["pred_causal"].view(-1), labels).mean()
            contrastive_loss = aux.get("gcl_loss", logits.new_zeros(()))
            confounder_loss = aux.get("conf_align_loss", logits.new_zeros(()))
            order_loss = rank_loss(logits.view(-1), labels)

            loss = (
                main_loss
                + args.lambda_causal * causal_loss
                + args.lambda_rank * order_loss
                + args.lambda_conf_align * confounder_loss
                + contrastive_loss
            )

            if args.lambda_cf > 0 and "pred_cf_drug" in aux and "pred_cf_prot" in aux:
                base_prob = torch.sigmoid(logits.view(-1)).detach()
                cf_loss = 0.5 * (
                    F.l1_loss(torch.sigmoid(aux["pred_cf_drug"].view(-1)), base_prob)
                    + F.l1_loss(torch.sigmoid(aux["pred_cf_prot"].view(-1)), base_prob)
                )
                loss = loss + args.lambda_cf * cf_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(label.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def configure_model(model, dataset, split):
    if hasattr(model, "enable_tapb_seq_randomization"):
        model.enable_tapb_seq_randomization = True
    if hasattr(model, "prot_mask_prob"):
        model.prot_mask_prob = 0.0
    if hasattr(model, "prot_mut_prob"):
        model.prot_mut_prob = 0.01 if dataset == "human" and split == "e4" else 0.035
    if hasattr(model, "external_conf_dropout_drug"):
        model.external_conf_dropout_drug = 0.60 if dataset == "human" and split == "e4" else 0.70
    if hasattr(model, "external_conf_dropout_prot"):
        model.external_conf_dropout_prot = 0.60 if dataset == "human" and split == "e4" else 0.80
    return model


def run_repeat(args, repeat):
    data_dir = Path(args.data_root) / args.dataset / args.split / str(repeat)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = read_split(data_dir)
    train_df, valid_df, test_df, label_col = prepare_frames(train_df, valid_df, test_df)

    loaders = make_loaders(
        train_df,
        valid_df,
        test_df,
        label_col,
        args.batch_size,
        args.eval_batch_size,
        args.num_workers,
        balanced=args.balanced_sampler,
    )
    train_loader, valid_loader, test_loader = loaders

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = configure_model(MambaCPAModelWoPretrained().to(device), args.dataset, args.split)

    labels = train_df[label_col].astype(float).to_numpy()
    pos = max(float(labels.sum()), 1.0)
    neg = max(float(len(labels) - labels.sum()), 1.0)
    pos_weight = torch.tensor([min(np.sqrt(neg / pos), args.max_pos_weight)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.05, 1e-6))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    checkpoint_path = output_dir / f"{args.dataset}-{args.split}-repeat{repeat}.pt"
    stopper = EarlyStopping(patience=args.patience, filename=str(checkpoint_path), mode="higher")

    best = None
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, scaler, criterion, device, args)
        valid_metrics = evaluate(model, valid_loader, device, branch=args.eval_branch, phys_mode=args.eval_phys_mode)
        scheduler.step()
        score = valid_metrics[args.monitor]
        early = stopper.step(score, model)
        row = {"repeat": repeat, "epoch": epoch, "loss": loss, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        print(json.dumps(row, ensure_ascii=True))
        if best is None or score > best[args.monitor]:
            best = dict(valid_metrics)
        if early and epoch + 1 >= args.min_epochs:
            break

    stopper.load_checkpoint(model)
    valid_metrics = evaluate(model, valid_loader, device, branch=args.eval_branch, phys_mode=args.eval_phys_mode)
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        branch=args.eval_branch,
        phys_mode=args.eval_phys_mode,
        threshold=valid_metrics["threshold"],
    )

    result = {
        "repeat": repeat,
        "checkpoint": str(checkpoint_path),
        **{f"valid_{k}": v for k, v in valid_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.environ.get("DATASET_NAME", "human"), choices=["human", "BindingDB"])
    parser.add_argument("--split", default=os.environ.get("SPLIT_MODE", "e4"))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "5")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lambda-causal", type=float, default=0.015)
    parser.add_argument("--lambda-rank", type=float, default=0.085)
    parser.add_argument("--lambda-conf-align", type=float, default=0.002)
    parser.add_argument("--lambda-cf", type=float, default=0.0)
    parser.add_argument("--max-pos-weight", type=float, default=8.0)
    parser.add_argument("--monitor", default="auprc", choices=["auprc", "auroc", "f1", "mcc"])
    parser.add_argument("--eval-branch", default="hybrid", choices=["hybrid", "causal", "shortcut"])
    parser.add_argument("--eval-phys-mode", default="normal", choices=["normal", "zero", "shuffle"])
    parser.add_argument("--train-phys-mode", default="normal", choices=["normal", "zero", "shuffle"])
    parser.add_argument("--balanced-sampler", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    results = []

    for repeat in range(args.repeats):
        set_seed(args.seed + repeat)
        result = run_repeat(args, repeat)
        results.append(result)
        print(json.dumps(result, ensure_ascii=True))

    out = pd.DataFrame(results)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_path = Path(args.output_dir) / f"{args.dataset}-{args.split}-summary.csv"
    out.to_csv(result_path, index=False)
    print(out.mean(numeric_only=True))
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
