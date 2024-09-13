import os, random, inspect

import pandas as pd
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from loguru import logger

import features


class DataPreprocess:
    def __init__(self, 
                 data_dir: str,
                 features: list,
                 train_name: str = 'train.csv',
                 test_name: str = 'test.csv'):
        self.data_dir = data_dir
        self.features: list[str] = features
        self.train_name = train_name
        self.test_name = test_name
        self._load_datas()
        self._preprocess()

    def _add_features(self):
        logger.info(f"[Preprocess] Add features to data...")
        for idx, feature in enumerate(self.features):
            logger.info(f"[Preprocess] {idx + 1} / {len(self.features)}: add '{feature}' ...")
            self._add_feature(self.train_df, feature)
            self._add_feature(self.test_df, feature)

    def _add_feature(self, df: pd.DataFrame, feature: str):
        if feature in df.columns:
            logger.warning(f"[Preprocess] 이미 존재하여 skip 합니다: {feature}")
            return
        
        # func = FEATURES[feature]
        funcs = {name: obj for name, obj in inspect.getmembers(features) if inspect.isfunction(obj)}
        if feature not in funcs:
            raise NotImplementedError(f"Feature 함수가 구현되어있지 않습니다: {feature}")
        func = funcs[feature]
        params = inspect.signature(func).parameters.keys()
        ADDITIONAL_PARAMS = {
            'train_df': self.train_df,
            'test_df': self.test_df,
        }
        input_params = [df] + [param for name, param in ADDITIONAL_PARAMS.items() if name in params]
        df[feature] = func(*input_params)
    
    def _load_datas(self):
        logger.info("[Preprocess] start loading datas...")
        self.train_df = pd.read_csv(os.path.join(self.data_dir, self.train_name))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, self.test_name))
        logger.info("[Preprocess] end loading datas...")

    def _preprocess(self):
        self._add_features()

    def split(self, valid_ratio: float = .2):
        logger.info("[Preprocess] split train_df into train and valid...")
        indices: int = range(len(self.train_df))
        valid_indices = random.sample(indices, int(len(indices) * valid_ratio))
        train_indices = sorted(list(set(indices) - set(valid_indices)))

        train_df = self.train_df.iloc[train_indices]
        valid_df = self.train_df.iloc[valid_indices]
        
        assert len(train_df) + len(valid_df) == len(self.train_df), f"train valid split fail: {len(train_df)} , {len(valid_df)} / {len(self.train_df)}"

        return train_df, valid_df, self.test_df
    
    def k_fold_split(self, k_fold: int = 5, seed:int = 42):
        logger.info("[Preprocess] split info k-fold...")
        k_folded_datasets = dict()
        
        shuffled_df = self.train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        size = len(shuffled_df) // k_fold
        
        for fold in range(k_fold):
            valid_start_idx = fold * size
            valid_end_idx = min(len(shuffled_df), (fold + 1) * size)

            valid_df = shuffled_df.iloc[valid_start_idx:valid_end_idx]
            train_df = shuffled_df.drop(valid_df.index)
            k_folded_datasets[fold] = {
                'train_df': train_df,
                'valid_df': valid_df,
            }

        return k_folded_datasets, self.test_df

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
            'pIC50': item['pIC50'].astype('float32'),
            'IC50': item['IC50_nM'].astype('float32'),
        } if self.train else {
            'X': self.transform(item['img']).view(-1,).to(torch.float32),
        }

class SimpleDNNDataset(Dataset):
    def __init__(self, data: pd.DataFrame, features: list[str], train: bool) -> None:
        super().__init__()
        self.features: list[str] = features
        self.data: pd.DataFrame = self._transformed(data)
        self.train: bool = train

    def _transformed(self, data: pd.DataFrame) -> pd.DataFrame:
        # logger.info(f"{data['gasteiger'].iloc[0].shape}")
        def concatenate_features(row):
            EMBEDDING_FEATURES = [
                'morgan_embedding',
                'morgan_atom_embedding',
            ]
            return np.concatenate([
                row[feature].flatten() for feature in self.features if feature not in EMBEDDING_FEATURES
            ]).astype('float32')
        
        def concatenate_embedding_features(row):
            EMBEDDING_FEATURES = [
                'morgan_embedding',
                'morgan_atom_embedding',
            ]
            return np.concatenate([
                row[feature].flatten() for feature in self.features if feature in EMBEDDING_FEATURES
            ]).astype('float32')

        data.loc[:, 'X'] = data.apply(concatenate_features, axis=1)
        data.loc[:, 'embeddings'] = data.apply(concatenate_embedding_features, axis=1)
        self.input_dim = data['X'].iloc[0].shape[0]
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        return {
            # 'X': item['fingerprint'].astype('float32'),
            # 'X': item['morgan_embedding'].flatten().astype('float32'),
            # 'Similarities': np.array(item['similarities']).astype('float32'),
            'X': item['X'].astype('float32'),
            'embeddings': item['embeddings'].astype('float32'),
            'pIC50': item['pIC50'].astype('float32'),
            'IC50': item['IC50_nM'].astype('float32'),
        } if self.train else {
            # 'X': item['fingerprint'].astype('float32'),
            # 'X': item['morgan_embedding'].flatten().astype('float32'),
            # 'X': np.concat([item['morgan_atom_embedding'].astype('float32'), item['morgan_embedding'].flatten().astype('float32')]),
            # 'Similarities': np.array(item['similarities']).astype('float32'),
            'X': item['X'].astype('float32'),
            'embeddings': item['embeddings'].astype('float32'),
        }

XGBoostPreprocess = DataPreprocess
SimpleDNNPreprocess = DataPreprocess

class XGBoostDataset:
    def __init__(self, data, features, train: bool = True) -> None:
        self.features: list[str] = features
        self.data: pd.DataFrame = self._transformed(data)
        self.train: bool = train
    
    def _transformed(self, data: pd.DataFrame) -> pd.DataFrame:
        # logger.info(f"{data['gasteiger'].iloc[0].shape}")
        def concatenate_features(row):
            return np.concatenate([
                row[feature].flatten() for feature in self.features
            ]).astype('float32')

        data.loc[:, 'X'] = data.apply(concatenate_features, axis=1)
        return data
    
    def __call__(self) -> dict:
        return {
            'X': self.data['X'],
            'pIC50': self.data['pIC50'],
            'IC50': self.data['IC50_nM'],
        } if self.train else {
            'X': self.data['X'],
        }
