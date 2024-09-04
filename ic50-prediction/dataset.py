import os, random

from rdkit import Chem
from rdkit.Chem import Draw, AllChem

import pandas as pd
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from loguru import logger


class DataPreprocess:
    def __init__(self, 
                 data_dir: str, 
                 train_name: str = 'train.csv',
                 test_name: str = 'test.csv'):
        self.data_dir = data_dir
        self.train_name = train_name
        self.test_name = test_name
        self._load_datas()
        self._preprocess()
    
    def _load_datas(self):
        logger.info("[Preprocess] start loading datas...")
        self.train_df = pd.read_csv(os.path.join(self.data_dir, self.train_name))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, self.test_name))
        logger.info("[Preprocess] end loading datas...")

    def _preprocess(self):
        def img_of(smiles: str):
            return Draw.MolToImage(Chem.MolFromSmiles(smiles))
        
        logger.info("[Preprocess] start preprocess train data...")
        self.train_df['assay'] = self.train_df['Assay ChEMBL ID'].apply(lambda v: v[6:])
        self.train_df['document'] = self.train_df['Document ChEMBL ID'].apply(lambda v: v[6:])
        self.train_df['molecule'] = self.train_df['Molecule ChEMBL ID'].apply(lambda v: v[6:])
        self.train_df = self.train_df.drop(self.train_df.columns[self.train_df.nunique() == 1], axis=1)
        self.train_df['img'] = self.train_df['Smiles'].apply(lambda x: img_of(x))
        self.train_df = self.train_df.drop(['Standard Value', 'pChEMBL Value', 'Assay ChEMBL ID', 'Document ChEMBL ID', 'Molecule ChEMBL ID'], axis=1)
        
        logger.info("[Preprocess] start preprocess test data...")
        self.test_df['img'] = self.test_df['Smiles'].apply(lambda x: img_of(x))
        logger.info("[Preprocess] end preprocess datas...")

    def split(self, valid_ratio: float = .2):
        logger.info("[Preprocess] split train_df into train and valid...")
        indices: int = range(len(self.train_df))
        valid_indices = random.sample(indices, int(len(indices) * valid_ratio))
        train_indices = sorted(list(set(indices) - set(valid_indices)))

        train_df = self.train_df.iloc[train_indices]
        valid_df = self.train_df.iloc[valid_indices]
        
        assert len(train_df) + len(valid_df) == len(self.train_df), f"train valid split fail: {len(train_df)} , {len(valid_df)} / {len(self.train_df)}"

        return train_df, valid_df, self.test_df
    
class SimpleDNNPreprocess(DataPreprocess):
    def __init__(self, 
                 data_dir: str, 
                 train_name: str = 'train.csv',
                 test_name: str = 'test.csv'):
        super().__init__(data_dir, train_name, test_name)
    
    def _preprocess(self):
        def img_of(smiles: str):
            return np.array(Draw.MolToImage(Chem.MolFromSmiles(smiles)))
        
        # SMILES 데이터를 분자 지문으로 변환
        def smiles_to_fingerprint(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                return np.array(fp)
            else:
                return np.zeros((2048,))
        
        logger.info("[SimpleDNNPreprocess] start preprocess train data...")
        self.train_df = self.train_df.drop(self.train_df.columns[self.train_df.nunique() == 1], axis=1)

        self.train_df['assay'] = self.train_df['Assay ChEMBL ID'].apply(lambda v: v[6:])
        self.train_df['document'] = self.train_df['Document ChEMBL ID'].apply(lambda v: v[6:])
        self.train_df['molecule'] = self.train_df['Molecule ChEMBL ID'].apply(lambda v: v[6:])

        self.train_df['document'] = self.train_df['document'].astype(int)
        self.train_df['molecule'] = self.train_df['molecule'].astype(int)
        self.train_df['assay'] = self.train_df['assay'].astype(int)


        self.train_df['img'] = self.train_df['Smiles'].apply(img_of)
        self.train_df['fingerprint'] = self.train_df['Smiles'].apply(smiles_to_fingerprint)
        self.train_df = self.train_df.drop(['Standard Value', 'pChEMBL Value', 'Assay ChEMBL ID', 'Document ChEMBL ID', 'Molecule ChEMBL ID'], axis=1)

        self.train_df['X'] = self.train_df.apply(
            lambda row: # [row['document'], row['molecule'], row['assay']] + 
                        row['img'].flatten().tolist()
            , axis=1
        )
        self.train_df['X'] = self.train_df.apply(lambda row: np.concat([row['X'], row['fingerprint']]), axis=1)

        self.input_size = self.train_df['X'][0].shape # for model input size

        logger.info("[SimpleDNNPreprocess] start preprocess test data...")
        self.test_df['img'] = self.test_df['Smiles'].apply(img_of)
        self.test_df['fingerprint'] = self.test_df['Smiles'].apply(smiles_to_fingerprint)
        self.test_df['X'] = self.test_df.apply(
            lambda row: # [row['document'], row['molecule'], row['assay']] + 
                        row['img'].flatten().tolist()
            , axis=1
        )
        self.test_df['X'] = self.test_df.apply(lambda row: np.concat([row['X'], row['fingerprint']]), axis=1)
        logger.info("[SimpleDNNPreprocess] end preprocess datas...")


class IC50Dataset(Dataset):
    def __init__(self, data: pd.DataFrame, train: bool=True):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        return {
            'X': self.transform(item['img']).view(-1,).to(torch.float32),
            'Y': item['pIC50'].astype('float32'),
        } if self.train else {
            'X': self.transform(item['img']).view(-1,).to(torch.float32),
        }

class SimpleDNNDataset(Dataset):
    def __init__(self, data: pd.DataFrame, train: bool) -> None:
        super().__init__()
        self.data: pd.DataFrame = data
        self.train: bool = train

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        return {
            # 'X': item['X'].astype('float32'),
            'X': item['fingerprint'].astype('float32'),
            'Y': item['pIC50'].astype('float32'),
        } if self.train else {
            # 'X': item['X'].astype('float32'),
            'X': item['fingerprint'].astype('float32'),
        }
    