import numpy as np
import torch


ATOM_NUMBERS = list(range(119))
ATOM_CHIRALITY = {
    "CHI_UNSPECIFIED": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "CHI_TETRAHEDRAL_CW": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "CHI_TETRAHEDRAL_CCW": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "CHI_OTHER": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "CHI_TETRAHEDRAL": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "CHI_ALLENE": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "CHI_SQUAREPLANAR": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "CHI_TRIGONALBIPYRAMIDAL": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "CHI_OCTAHEDRAL": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}
ATOM_DEGREES = list(range(11))
ATOM_CHARGES = list(range(-5, 7))
ATOM_HS = list(range(9))
ATOM_RADICALS = list(range(5))
ATOM_HYBRIDIZATION = {
    "UNSPECIFIED": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "S": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "SP": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "SP2": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "SP3": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "SP3D": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "SP3D2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "OTHER": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}
BOND_TYPE = {
    "UNSPECIFIED": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "SINGLE": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "DOUBLE": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "TRIPLE": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "OTHER": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "AROMATIC": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "LONGRANGE": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}
BOND_STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]
BOOL_FEATURE = {
    False: [0.0, 1.0],
    True: [1.0, 0.0],
}


def safe_index(values, value):
    return values.index(value) if value in values else 0


def atom_features(atom):
    atomic_num = atom.GetAtomicNum()
    if atomic_num not in ATOM_NUMBERS:
        atomic_num = 0

    chirality = ATOM_CHIRALITY.get(str(atom.GetChiralTag()), ATOM_CHIRALITY["CHI_UNSPECIFIED"])
    hybridization = ATOM_HYBRIDIZATION.get(str(atom.GetHybridization()), ATOM_HYBRIDIZATION["OTHER"])

    row = [
        safe_index(ATOM_NUMBERS, atomic_num),
        *chirality,
        safe_index(ATOM_DEGREES, atom.GetTotalDegree()),
        safe_index(ATOM_CHARGES, atom.GetFormalCharge()),
        safe_index(ATOM_HS, atom.GetTotalNumHs()),
        safe_index(ATOM_RADICALS, atom.GetNumRadicalElectrons()),
        *hybridization,
        *BOOL_FEATURE[atom.GetIsAromatic()],
        *BOOL_FEATURE[atom.IsInRing()],
    ]
    return row


def node_encoding(mol):
    rows = [atom_features(atom) for atom in mol.GetAtoms()]
    return torch.tensor(rows, dtype=torch.float).view(-1, 26)


def bond_features(bond, long_range=False):
    if long_range:
        bond_type = BOND_TYPE["LONGRANGE"]
        stereo = 6
        conjugated = False
        aromatic = False
        in_ring = False
    else:
        bond_type = BOND_TYPE.get(str(bond.GetBondType()), BOND_TYPE["OTHER"])
        stereo = safe_index(BOND_STEREO, str(bond.GetStereo()))
        conjugated = bond.GetIsConjugated()
        aromatic = bond.GetIsAromatic()
        in_ring = bond.IsInRing()

    return [
        *bond_type,
        stereo,
        *BOOL_FEATURE[conjugated],
        *BOOL_FEATURE[aromatic],
        *BOOL_FEATURE[in_ring],
    ]


def edge_encoding(mol, add_fake_edges=False):
    num_atoms = mol.GetNumAtoms()
    neighbors = {i: set() for i in range(num_atoms)}
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        attr = bond_features(bond)
        neighbors[i].add(j)
        neighbors[j].add(i)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [attr, attr]

    if add_fake_edges and num_atoms > 1:
        selected = np.linspace(0, num_atoms, num=max(1, int(num_atoms * 0.05)), dtype=int, endpoint=False)
        for i in selected:
            for j in selected:
                if i == j or j in neighbors[i]:
                    continue
                attr = bond_features(None, long_range=True)
                edge_indices += [[i, j], [j, i]]
                edge_attrs += [attr, attr]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 14)
    return edge_index, edge_attr


def from_smiles(smiles, with_hydrogen=False, kekulize=False, one_hot=True, add_fake_edges=False):
    if not one_hot:
        raise ValueError("Only one-hot molecular graph features are available.")

    from rdkit import Chem, RDLogger
    from torch_geometric.data import Data

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        mol = Chem.MolFromSmiles("")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    x = node_encoding(mol)
    edge_index, edge_attr = edge_encoding(mol, add_fake_edges=add_fake_edges)

    if edge_index.numel() > 0:
        order = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index = edge_index[:, order]
        edge_attr = edge_attr[order]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
