# candi use generated promising candidate molecules as the LA to train the generating model

import gentrl
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random
from utilities.candiconfig import CandiConfig

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

# LA_smiles, _ = fps_som.get_LA_zinc_data(multiplier=10)
# LA_list = [s+',train' for s in LA_smiles]

# zinc_list = []
# with open(config.zinc_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         zinc_list.append(line.strip())

# zinc_list.remove('SMILES,SPLIT')
        
# mixed_list = zinc_list + LA_list

# random.shuffle(mixed_list)

# with open(config.mixed_dataset, 'w', encoding='utf-8') as f:
#     f.write('SMILES,SPLIT\n')
#     for s in mixed_list:
#         f.write(s+'\n')

# df = pd.read_csv(config.mixed_dataset)
# df = df[df['SPLIT'] == 'train']
# # df['plogP'] = df['SMILES'].apply(penalized_logP)
# df['reward'] = df['SMILES'].apply(fps_som.som_reward)
# df.to_csv(config.mixed_train_dataset, index=None)

md = gentrl.MolecularDataset(sources=[
    {'path':config.mixed_train_dataset,
     'smiles': 'SMILES',
     'prob': 1,
#      'plogP' : 'plogP',
     'reward' : 'reward',
    }], 
#    props=['plogP', 'reward'])
    props=['reward'])

from torch.utils.data import DataLoader
train_loader = DataLoader(md, batch_size=50, shuffle=True, num_workers=1, drop_last=True)

enc = gentrl.RNNEncoder(latent_size=50)
dec = gentrl.DilConvDecoder(latent_input_size=50)
model = gentrl.GENTRL(enc, dec, 50 * [('c', 20)], 1* [('c', 20)], beta=0.001)
model.cuda()

model.train_as_vaelp(train_loader, lr=1e-4)

model.save('./candi_saved_gentrl/')