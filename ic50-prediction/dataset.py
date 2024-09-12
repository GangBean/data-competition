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
    
class SimpleDNNPreprocess(DataPreprocess):
    def __init__(self, 
                 data_dir: str, 
                 features: list,
                 train_name: str = 'train.csv',
                 test_name: str = 'test.csv',
                ):
        super().__init__(data_dir, features, train_name, test_name)
    
    # def _preprocess(self):
        # CFG = {
        #     'NBITS':2048,
        #     'SEED':42,
        # }
        
        # logger.info(f"[SimpleDNNPreprocess] baseline fingerprint...")
        # def smiles_to_fingerprint_baseline(smiles):
        #     mol = Chem.MolFromSmiles(smiles)
        #     if mol is not None:
        #         fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        #         return np.array(fp)
        #     else:
        #         return np.zeros((CFG['NBITS'],))
        
        # self.train_df['baseline_fingerprint'] = self.train_df['Smiles'].apply(smiles_to_fingerprint_baseline)
        # self.test_df['baseline_fingerprint'] = self.test_df['Smiles'].apply(smiles_to_fingerprint_baseline)
        
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
        
        # self.train_df['morgan_info'] = self.train_df['Smiles'].apply(smiles_to_morgan)
        # self.test_df['morgan_info'] = self.test_df['Smiles'].apply(smiles_to_morgan)

        # morgan_key_set = set() # aggregate total morgan key set from train and test data
        # self.train_df['morgan_info'].apply(lambda info: morgan_key_set.update(info.keys()))
        # self.test_df['morgan_info'].apply(lambda info: morgan_key_set.update(info.keys()))

        # morgan_key = sorted(list(morgan_key_set))
        # total_morgan_key2idx = {k:i for i, k in enumerate(morgan_key)} # morgan key to idx

        # def morgan_info_to_embedding(info, key_count=13_279, radius_count=4, max_len: int=1_000):
        #     embedding = np.zeros((key_count, radius_count, )) # init embedding

        #     for key, radius_list in info.items(): # iterate over morgan_infos
        #         radius_count = dict(Counter([radius[-1] for radius in radius_list])) # count radius's emerge count
        #         for radius, count in radius_count.items():
        #             embedding[total_morgan_key2idx[key], radius] = count

        #     return embedding[:max_len, :]
        
        # self.train_df['morgan_embedding'] = self.train_df['morgan_info'].apply(morgan_info_to_embedding)
        # self.test_df['morgan_embedding'] = self.test_df['morgan_info'].apply(morgan_info_to_embedding)

        # logger.info("[SimpleDNNPreprocess] standardization...")
        # def standardization(embedding):
        #     mean = np.mean(embedding.flatten())
        #     std = np.std(embedding.flatten())

        #     return (embedding - mean) / std
        
        # self.train_df['morgan_embedding'] = self.train_df['morgan_embedding'].apply(standardization)
        # self.test_df['morgan_embedding'] = self.train_df['morgan_embedding'].apply(standardization)

        # logger.info("[SimpleDNNPreprocess] Similarities...")
        # def smiles_to_fingerprint(smiles):
        #     mol = Chem.MolFromSmiles(smiles)
        #     morgan_gen = AllChem.GetMorganGenerator()
        #     return morgan_gen.GetSparseCountFingerprint(mol)
        
        # self.train_df['fingerprint'] = self.train_df['Smiles'].apply(smiles_to_fingerprint)
        # self.test_df['fingerprint'] = self.test_df['Smiles'].apply(smiles_to_fingerprint)

        # train_fps = self.train_df['fingerprint'].tolist()

        # def similarities(fp, train_fp=train_fps):
        #     output = []
        #     for train_fp in train_fps:
        #         output.append(DataStructs.TanimotoSimilarity(fp,train_fp))
        #     return output
    
        # tqdm.pandas()
        # self.train_df['similarities'] = self.train_df['fingerprint'].progress_apply(similarities)
        # self.test_df['similarities'] = self.test_df['fingerprint'].progress_apply(similarities)

        # logger.info("[SimpleDNNPreprocess] morgan atom embedding...")
        # def atom_count_array(info, max_count: int = 72):
        #     atoms = [max([value[0] for value in values]) for values in info.values()]
        #     atom_count = dict(Counter(atoms))
        #     output = {
        #         idx: 0 for idx in range(max_count)
        #     }
        #     output.update(atom_count)

        #     array = np.zeros(max_count)
        #     for idx, count in output.items():
        #         array[idx] = count
        #     return array
        
        # self.train_df['morgan_atom_embedding'] = self.train_df['morgan_info'].apply(atom_count_array)
        # self.test_df['morgan_atom_embedding'] = self.test_df['morgan_info'].apply(atom_count_array)

        # self.train_df['mol'] = self.train_df['Smiles'].apply(Chem.MolFromSmiles)
        # self.test_df['mol'] = self.test_df['Smiles'].apply(Chem.MolFromSmiles)
        
        # logger.info("[SimpleDNNPreprocess] Bonds num...")
        # self.train_df['num_bonds'] = self.train_df['mol'].apply(lambda mol: np.array([mol.GetNumBonds()]))
        # self.test_df['num_bonds'] = self.test_df['mol'].apply(lambda mol: np.array([mol.GetNumBonds()]))
        
        # logger.info("[SimpleDNNPreprocess] Rings num...")
        # self.train_df['num_rings'] = self.train_df['mol'].apply(lambda mol: np.array([Chem.rdMolDescriptors.CalcNumRings(mol)]))
        # self.test_df['num_rings'] = self.test_df['mol'].apply(lambda mol: np.array([Chem.rdMolDescriptors.CalcNumRings(mol)]))
        
        # logger.info("[SimpleDNNPreprocess] Kappa1,2,3 ...")
        # self.train_df['kappa_1'] = self.train_df['mol'].apply(lambda mol: np.array([Descriptors.Kappa1(mol)]))
        # self.train_df['kappa_2'] = self.train_df['mol'].apply(lambda mol: np.array([Descriptors.Kappa2(mol)]))
        # self.train_df['kappa_3'] = self.train_df['mol'].apply(lambda mol: np.array([Descriptors.Kappa3(mol)]))
        # self.test_df['kappa_1'] = self.test_df['mol'].apply(lambda mol: np.array([Descriptors.Kappa1(mol)]))
        # self.test_df['kappa_2'] = self.test_df['mol'].apply(lambda mol: np.array([Descriptors.Kappa2(mol)]))
        # self.test_df['kappa_3'] = self.test_df['mol'].apply(lambda mol: np.array([Descriptors.Kappa3(mol)]))

        # logger.info("[SimpleDNNPreprocess] MACCS ...")
        # self.train_df['maccs'] = self.train_df['mol'].apply(lambda mol: np.array(MACCSkeys.GenMACCSKeys(mol)))
        # self.test_df['maccs'] = self.test_df['mol'].apply(lambda mol: np.array(MACCSkeys.GenMACCSKeys(mol)))

        # logger.info("[SimpleDNNPreprocess] Gasteiger charges...")
        # def gasteiger_charge(mol, fixed_size=100):
        #     AllChem.ComputeGasteigerCharges(mol)
        #     charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
            
        #     if len(charges) < fixed_size:
        #         charges.extend([0.0] * (fixed_size - len(charges)))
        #     else:
        #         charges = charges[:fixed_size]
            
        #     return np.array(charges)
        
        # self.train_df['gasteiger'] = self.train_df['mol'].apply(gasteiger_charge)
        # self.test_df['gasteiger'] = self.test_df['mol'].apply(gasteiger_charge)

        # logger.info("[SimpleDNNPreprocess] moments...")
        # def calculate_inertial_moments(mol):
        #     mol = Chem.AddHs(mol)  # 수소 추가

        #     # 3D 구조 생성 및 최적화
        #     AllChem.EmbedMolecule(mol)
        #     AllChem.UFFOptimizeMolecule(mol)  # UFF 최적화

        #     # Conformer를 얻고 각 원자의 위치를 가져오기
        #     conf = mol.GetConformer()
        #     coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        #     coords = np.array(coords)
            
        #     # 분자의 무게 중심을 계산
        #     center_of_mass = np.mean(coords, axis=0)
        #     coords_centered = coords - center_of_mass

        #     # 관성 모멘트 계산
        #     Ixx = np.sum((coords_centered[:, 1]**2 + coords_centered[:, 2]**2))
        #     Iyy = np.sum((coords_centered[:, 0]**2 + coords_centered[:, 2]**2))
        #     Izz = np.sum((coords_centered[:, 0]**2 + coords_centered[:, 1]**2))
        #     Ixy = -np.sum(coords_centered[:, 0] * coords_centered[:, 1])
        #     Iyz = -np.sum(coords_centered[:, 1] * coords_centered[:, 2])
        #     Izx = -np.sum(coords_centered[:, 2] * coords_centered[:, 0])
            
        #     inertial_moments = np.array([Ixx, Iyy, Izz, Ixy, Iyz, Izx])
        #     return inertial_moments
        
        # self.train_df['moment'] = self.train_df['mol'].progress_apply(calculate_inertial_moments)
        # self.test_df['moment'] = self.test_df['mol'].progress_apply(calculate_inertial_moments)

        # logger.info("[SimpleDNNPreprocess] all descriptors...")
        # def calc_mol_descriptors(mol):
        #     desc = Descriptors.CalcMolDescriptors(mol)
        #     output = []
        #     for _, value in desc.items():
        #         output.append(np.array(value, dtype='float32').flatten())
        #     return np.concat(output)
        
        # self.train_df['all_desc'] = self.train_df['mol'].apply(calc_mol_descriptors)
        # self.test_df['all_desc'] = self.test_df['mol'].apply(calc_mol_descriptors)

        # logger.info("[SimpleDNNPreprocess] calulate 3d descriptors...")
        # def calc_all_3d_descriptors(mol):
        #     mol = Chem.AddHs(mol)
        #     AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # 3D 좌표 생성
        #     AllChem.UFFOptimizeMolecule(mol)  # 에너지 최소화
        #     desc_3d = Descriptors3D.CalcMolDescriptors3D(mol)
        #     output = []
        #     for _, value in desc_3d.items():
        #         output.append(np.array(value, dtype='float32').flatten())
        #     return np.concat(output)
        
        # self.train_df['all_3d_desc'] = self.train_df['mol'].progress_apply(calc_all_3d_descriptors)
        # self.test_df['all_3d_desc'] = self.test_df['mol'].progress_apply(calc_all_3d_descriptors)

        # logger.info("[SimpleDNNPreprocess] end preprocess datas...")


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
            # 'X': item['fingerprint'].astype('float32'),
            # 'X': item['morgan_embedding'].flatten().astype('float32'),
            'X': np.concat([item['morgan_atom_embedding'].astype('float32'), item['morgan_embedding'].flatten().astype('float32')]),
            'Similarities': np.array(item['similarities']).astype('float32'),
            'pIC50': item['pIC50'].astype('float32'),
            'IC50': item['IC50_nM'].astype('float32'),
        } if self.train else {
            # 'X': item['X'].astype('float32'),
            # 'X': item['fingerprint'].astype('float32'),
            # 'X': item['morgan_embedding'].flatten().astype('float32'),
            'X': np.concat([item['morgan_atom_embedding'].astype('float32'), item['morgan_embedding'].flatten().astype('float32')]),
            'Similarities': np.array(item['similarities']).astype('float32'),
        }

XGBoostPreprocess = SimpleDNNPreprocess

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
