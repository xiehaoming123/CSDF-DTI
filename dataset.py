import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from mol_graph import from_smiles


PROTEIN_VOCAB = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25,
}

SMILES_VOCAB = {
    "(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7,
    "@": 8, "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14,
    "P": 15, "R": 16, "T": 17, "V": 18, "Z": 19, "\\": 20,
    "b": 21, "d": 22, "f": 23, "h": 24, "l": 25, "n": 26, "r": 27,
    "t": 28, "#": 29, "%": 30, ")": 31, "+": 32, "-": 33,
    "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39,
    "=": 40, "A": 41, "C": 42, "E": 43, "G": 44, "I": 45,
    "K": 46, "M": 47, "O": 48, "S": 49, "U": 50, "W": 51,
    "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56, "e": 57,
    "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63,
    "y": 64,
}


def encode_text(text, max_len, vocab):
    out = np.zeros(max_len, dtype=np.int64)
    for i, char in enumerate(str(text)[:max_len]):
        out[i] = vocab.get(char, 0)
    return out


def label_smiles(line, max_len, vocab=SMILES_VOCAB):
    return encode_text(line, max_len, vocab)


def label_sequence(line, max_len, vocab=PROTEIN_VOCAB):
    return encode_text(line, max_len, vocab)


class ProtDrugSeqDatasetCLS(Dataset):
    def __init__(self, data_df, training=False, seq_len=1024, smiles_len=128):
        super().__init__()
        self.df = data_df.reset_index(drop=True)
        self.seqlen = int(seq_len)
        self.smilen = int(smiles_len)
        self.charseqset = PROTEIN_VOCAB
        self.charsmiset = SMILES_VOCAB
        self.charseqset_size = len(PROTEIN_VOCAB)
        self.charsmiset_size = len(SMILES_VOCAB)
        self.mol_graphs = {}
        self.smiles_cache = {}
        self.protein_cache = {}
        self._build_cache()

    def _build_cache(self):
        for smiles in self.df["SMILES"].dropna().unique():
            self.mol_graphs[smiles] = from_smiles(smiles)
            self.smiles_cache[smiles] = label_smiles(smiles, self.smilen, self.charsmiset)

        for sequence in self.df["Protein"].dropna().unique():
            self.protein_cache[sequence] = label_sequence(sequence, self.seqlen, self.charseqset)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row["SMILES"]
        protein = row["Protein"]
        label = torch.tensor([float(row["Y"])], dtype=torch.float32)
        pro_seq_id = torch.from_numpy(self.protein_cache[protein]).long()
        smiles_id = torch.from_numpy(self.smiles_cache[smiles]).long()
        mol_graph = self.mol_graphs[smiles]
        return pro_seq_id, smiles_id, mol_graph, label

    def collate_fn(self, data_list):
        pro_seq_id = torch.stack([item[0] for item in data_list], dim=0)
        smiles_id = torch.stack([item[1] for item in data_list], dim=0)
        label = torch.stack([item[3] for item in data_list], dim=0)
        mol_graph = Batch.from_data_list([item[2] for item in data_list])
        return pro_seq_id, smiles_id, mol_graph, label
