import gentrl
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random
from utilities.candiconfig import CandiConfig
from rdkit import Chem

torch.cuda.set_device(0)

from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol

def get_num_rings_6(mol):
    r = mol.GetRingInfo()
    return len([x for x in r.AtomRings() if len(x) > 6])


def penalized_logP(mol_or_smiles, masked=False, default=-5):
    mol = get_mol(mol_or_smiles)
    if mol is None:
        return default
    reward = logP(mol) - SA(mol) - get_num_rings_6(mol)
    if masked and not mol_passes_filters(mol):
        return default
    return reward


config = CandiConfig(smiles_format=2, topn_fp_features=5, mode='threshold', max_fp_features=2048, threshold=0.3, morgan_radius=2)
with open(config.FpsSOM_model, 'rb') as infile:
    fps_som = pickle.load(infile)