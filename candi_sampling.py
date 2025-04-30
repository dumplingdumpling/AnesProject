import gentrl
import torch
import pickle
import pandas as pd
import numpy as np
from rdkit.Chem import Draw
from rdkit import RDLogger
from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol
import matplotlib.pyplot as plt
import random
import os
from utilities.candiconfig import CandiConfig

torch.cuda.set_device(0)
RDLogger.DisableLog('rdApp.*')


enc = gentrl.RNNEncoder(latent_size=50)
dec = gentrl.DilConvDecoder(latent_input_size=50)
model = gentrl.GENTRL(enc, dec, 50 * [('c', 20)], [('c', 20)], beta=0.001)
model.cuda();

model.load('candi_saved_gentrl/')
model.cuda()

# new, 2021-07-26 原版：生成size个候选分子， 改版：可以生成size个有效分子
def sample_from_model(model, fps_som, exist, size=1000, threshold = 0):
    generated = []
    grades = []
    
    num = 0
    
    while len(generated) < size:
        #sampled = model.sample(size//10)
        sampled = model.sample(500)
        if len(generated) >= num * size // 50:
            print("already have: " + str(len(generated)) + " molecules...")
            num += 1
#         sampled = model.sample(5000)
        for s in sampled:
            if get_mol(s) and len(s) > 10 and s not in exist and fps_som.som_reward(s) >= threshold:
                generated.append(s)
                grades.append(fps_som.som_reward(s))
                exist.add(s)
        #print(len(generated), len(grades))
        
            
    idxs = np.argsort(grades).tolist()
    idxs = idxs[::-1]
    
    smiles_list = [generated[i] for i in idxs]
    grades_list = [grades[i] for i in idxs]
    
    return smiles_list, grades_list


config = CandiConfig(smiles_format=2, topn_fp_features=5, mode='threshold', max_fp_features=2048, threshold=0.3, morgan_radius=2)
with open(config.FpsSOM_model, 'rb') as infile:
    fps_som = pickle.load(infile)


exist_smiles = set()


with open('./dataset/train_dataset.csv', 'r', encoding='utf-8') as f:
    for line in f:
        exist_smiles.add(line.strip().split(',')[0])

path = './generated_smiles/'
files = os.listdir(path)
csv_files = [path+file for file in files]

print(csv_files)

for file in csv_files:
    if os.path.isfile(file):
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                exist_smiles.add(line.strip().split(',')[1])
                

size_num = 50000
smiles_list, grades_list = sample_from_model(model, fps_som, exist_smiles, size = size_num)


topn = size_num // 2
# topn = 50000
topn = topn if topn < len(smiles_list) else len(smiles_list)

selected_smiles_list = smiles_list[:topn]
selected_grades_list = grades_list[:topn]

with open('candidate_smiles.txt', 'w', encoding='utf-8') as f:
    f.write("SMILES,SOM_REWARD\n")
    for i,s in enumerate(selected_smiles_list):
        f.write(s+','+str(selected_grades_list[i])+'\n')


import time

now_date = time.strftime("%Y_%m_%d", time.localtime())
pre_date = time.strftime("%Y%m%d", time.localtime())[2:]



smiles = []
som_reward = []
with open("candidate_smiles.txt", 'r', encoding='utf-8') as f:
    for line in f:
        sm, grade = line.strip().split(',')
        smiles.append(sm)
        som_reward.append(grade)

smiles = smiles[1:]
som_reward = som_reward[1:]
SA_grades = [SA(get_mol(sm)) for sm in smiles]

with open("./generated_smiles/candi_smiles_"+now_date+".csv",'w',encoding='utf-8') as f:
#     f.write("No,Smiles,SOM_reward,SA_grade\n")
    for i, sm in enumerate(smiles):
        line = pre_date+"_"+str(i+1)+','+ sm +','+ som_reward[i] +','+ str(SA_grades[i])+'\n'
        f.write(line)
    