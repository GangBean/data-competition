import pandas as pd
import numpy as np

from collections import Counter
from rdkit import Chem
from rdkit.Chem import (
    Draw, AllChem, DataStructs, Descriptors, MACCSkeys, Descriptors3D
)
from loguru import logger
from tqdm import tqdm


def morgan_atom_embedding(df):
    def smiles_to_morgan(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            additional_output = AllChem.AdditionalOutput()
            additional_output.CollectBitInfoMap()
            morgan_gen = AllChem.GetMorganGenerator()
            morgan_gen.GetSparseCountFingerprint(mol, additionalOutput=additional_output)
            info = additional_output.GetBitInfoMap()
            return info
        else:
            return np.zeros((13_279,))
    
    df['morgan_info'] = df['Smiles'].apply(smiles_to_morgan)

    def atom_count_array(info, max_count: int = 72):
        atoms = [max([value[0] for value in values]) for values in info.values()]
        atom_count = dict(Counter(atoms))
        output = {
            idx: 0 for idx in range(max_count)
        }
        output.update(atom_count)

        array = np.zeros(max_count)
        for idx, count in output.items():
            array[idx] = count
        return array
    
    return df['morgan_info'].apply(atom_count_array)

def baseline_fingerprint(df):
    CFG = {
            'NBITS':2048,
            'SEED':42,
        }
    
    def smiles_to_fingerprint_baseline(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
            return np.array(fp)
        else:
            return np.zeros((CFG['NBITS'],))
    
    return df['Smiles'].apply(smiles_to_fingerprint_baseline)

def morgan_embedding(df):
    raise NotImplementedError("수정이 필요합니다.")
    # logger.info(f"[SimpleDNNPreprocess] Transform to Morgan Embedding...")
    # def smiles_to_morgan(smiles):
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is not None:
    #         additional_output = AllChem.AdditionalOutput()
    #         additional_output.CollectBitInfoMap()
    #         morgan_gen = AllChem.GetMorganGenerator()
    #         morgan_gen.GetSparseCountFingerprint(mol, additionalOutput=additional_output)
    #         info = additional_output.GetBitInfoMap()
    #         return info
    #     else:
    #         return np.zeros((13_279,))
    
    # train_df['morgan_info'] = train_df['Smiles'].apply(smiles_to_morgan)
    # test_df['morgan_info'] = test_df['Smiles'].apply(smiles_to_morgan)

    # morgan_key_set = set() # aggregate total morgan key set from train and test data
    # train_df['morgan_info'].apply(lambda info: morgan_key_set.update(info.keys()))
    # test_df['morgan_info'].apply(lambda info: morgan_key_set.update(info.keys()))

    # morgan_key = sorted(list(morgan_key_set))
    # total_morgan_key2idx = {k:i for i, k in enumerate(morgan_key)} # morgan key to idx

    # def morgan_info_to_embedding(info, key_count=13_279, radius_count=4, max_len: int=1_000):
    #     embedding = np.zeros((key_count, radius_count, )) # init embedding

    #     for key, radius_list in info.items(): # iterate over morgan_infos
    #         radius_count = dict(Counter([radius[-1] for radius in radius_list])) # count radius's emerge count
    #         for radius, count in radius_count.items():
    #             embedding[total_morgan_key2idx[key], radius] = count

    #     return embedding[:max_len, :]
    
    # df['morgan_embedding'] = df['morgan_info'].apply(morgan_info_to_embedding)

    # logger.info("[SimpleDNNPreprocess] standardization...")
    # def standardization(embedding):
    #     mean = np.mean(embedding.flatten())
    #     std = np.std(embedding.flatten())

    #     return (embedding - mean) / std
    
    # df['morgan_embedding'] = df['morgan_embedding'].apply(standardization)
    # return df['morgan_embedding'].apply(standardization)

def similarities(df):
    raise NotImplementedError("수정이 필요합니다.")
    # logger.info("[SimpleDNNPreprocess] Similarities...")
    # def smiles_to_fingerprint(smiles):
    #     mol = Chem.MolFromSmiles(smiles)
    #     morgan_gen = AllChem.GetMorganGenerator()
    #     return morgan_gen.GetSparseCountFingerprint(mol)
    
    # df['fingerprint'] = df['Smiles'].apply(smiles_to_fingerprint)

    # train_fps = df['fingerprint'].tolist()

    # def similarities(fp, train_fp=train_fps):
    #     output = []
    #     for train_fp in train_fps:
    #         output.append(DataStructs.TanimotoSimilarity(fp,train_fp))
    #     return output

    # tqdm.pandas()
    # return df['fingerprint'].progress_apply(similarities)

def num_bonds():
    pass

def num_rings():
    pass

def kappa_1():
    pass

def kappa_2():
    pass

def kappa_3():
    pass

def maccs():
    pass

def all_desc(df):
    df['mol'] = df['Smiles'].apply(Chem.MolFromSmiles)

    def calc_mol_descriptors(mol):
        desc = Descriptors.CalcMolDescriptors(mol)
        output = []
        for _, value in desc.items():
            output.append(np.array(value, dtype='float32').flatten())
        return np.concat(output)
    
    return df['mol'].apply(calc_mol_descriptors)

def all_3d_desc(df):
    df['mol'] = df['Smiles'].apply(Chem.MolFromSmiles)

    def calc_all_3d_descriptors(mol):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # 3D 좌표 생성
        AllChem.UFFOptimizeMolecule(mol)  # 에너지 최소화
        desc_3d = Descriptors3D.CalcMolDescriptors3D(mol)
        output = []
        for _, value in desc_3d.items():
            output.append(np.array(value, dtype='float32').flatten())
        return np.concat(output)
    
    return df['mol'].progress_apply(calc_all_3d_descriptors)

FEATURES = {
    'morgan_embedding': morgan_embedding,
    'baseline_fingerprint': baseline_fingerprint, 
    'morgan_atom_embedding': morgan_atom_embedding,
    'similarities': similarities,
    'num_bonds': num_bonds,
    'num_rings': num_rings,
    'kappa_1': kappa_1,
    'kappa_2': kappa_2,
    'kappa_3': kappa_3,
    'maccs': maccs,
    'all_desc': all_desc,
    'all_3d_desc': all_3d_desc,
}