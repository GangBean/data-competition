import os, copy

import torch
import torch.nn as nn
from torch.optim.adam import Optimizer, Adam
from sklearn.metrics import mean_squared_error

from tqdm import tqdm
from loguru import logger

from models import SimpleImageRegressor, SimpleDNN, XGBoost
from utils import pIC50_to_IC50
from metrics import score

import wandb
import pandas as pd
import numpy as np

class Trainer:
    def __init__(self, cfg, fold: int=0) -> None:
        self.cfg = cfg
        self.device = self._device()
        self.fold = fold

        self.model: nn.Module = self._model().to(self.device)
        self.optimizer: Optimizer = Adam(self.model.parameters(), lr=self.cfg.lr)
        self.loss = self._loss()

    def _set_runname(self):
        self.run_name = self.cfg.run_name

    def _device(self):
        if not torch.cuda.is_available():
            logger.warning(f"GPU 사용이 불가합니다. CPU로 강제 수행됩니다.")
            return 'cpu'
        if self.cfg.device.lower() not in ('cpu', 'cuda'):
            logger.warning(f"비정상적인 device 설정값 입니다. device 설정을 확인해주세요. CPU로 강제 수행됩니다.: {self.cfg.device}")
            return 'cpu'
        return self.cfg.device.lower()

    def _loss(self):
        loss = self.cfg.loss.lower()
        if loss == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"정의되지 않은 loss 입니다: {self.cfg.loss}")

    def _model(self):
        model = self.cfg.model_name.lower()
        if model == 'resnet':
            return SimpleImageRegressor(embedding_size=300 * 300 * 3)
        elif model == 'bert':
            return SimpleImageRegressor(embedding_size=300 * 300 * 3)
        elif model == 'dnn':
            return SimpleDNN(
                # input_dim=2_048 + 300 * 300 * 3
                input_dim= 1 # 13_279 * 4
                , embed_dim=self.cfg.embed_dim
                , layer_dims=self.cfg.layer_dims
                , type=self.cfg.type)
        else:
            raise ValueError(f"해당하는 모델이 존재하지 않습니다: {self.cfg.model_name}")
        
    def _save_best_model(self):
        logger.info(f"[Trainer] best model을 저장합니다.")
        model_filename: str = f'{os.path.join(self.cfg.model_dir, self.run_name)}_{self.fold}.pt'
        if not os.path.exists(self.cfg.model_dir):
            logger.info(f"[Trainer] model 저장 경로를 생성합니다: {self.cfg.model_dir}")
            os.makedirs(self.cfg.model_dir, exist_ok=True)
        torch.save(copy.deepcopy(self.model).cpu().state_dict(), model_filename)
        # self.model = self.model.to(self.device)

    def load_best_model(self):
        logger.info(f"[Trainer] best model을 불러옵니다.")
        model_filename: str = f'{os.path.join(self.cfg.model_dir, self.run_name)}_{self.fold}.pt'
        if not os.path.exists(model_filename):
            logger.warning(f"[Trainer] 해당 파일이 존재하지 않습니다: {model_filename}")
        self.model.load_state_dict(torch.load(model_filename, weights_only=True))
        self.model = self.model.to(self.device)

    def run(self, train_dataloader, valid_dataloader):
        self._set_runname()
        best_valid_loss = .0
        best_valid_score = .0
        patience = 0
        for epoch in range(self.cfg.epoch):
            train_loss, train_score = self.train(train_dataloader)
            valid_loss, valid_score = self.validate(valid_dataloader)
            logger.info(f'''\n[Trainer] epoch: {epoch + 1} > train loss: {train_loss:.4f} / train_score: {train_score:.4f}
                        valid loss: {valid_loss:.4f} / valid_score: {valid_score:.4f}''')

            if best_valid_loss == .0 or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_score = valid_score
                self._save_best_model()
                patience = 0
            
            if self.cfg.wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'train_score': train_score,
                    'valid_score': valid_score,
                })
            
            if patience >= self.cfg.patience:
                logger.info(f"[Trainer] 학습결과가 개선되지 않아 조기 종료합니다: {epoch + 1} / {self.cfg.epoch}")
                break
            else:
                patience += 1
        return best_valid_loss, best_valid_score

    def train(self, train_dataloader) -> float:
        self.model.train()
        train_loss: float = .0
        actual_pic50 = []
        actual_ic50 = []
        pred_pic50 = []
        for data in tqdm(train_dataloader):
            X, Y, similarities = data['X'].to(self.device), data['pIC50'].to(self.device), data['Similarities'].to(self.device)

            pred = self.model(X, similarities)
            loss: torch.Tensor = torch.sqrt(self.loss(pred.squeeze(), Y.squeeze()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            actual_pic50.extend(data['pIC50'].numpy())
            actual_ic50.extend(data['IC50'].numpy())
            pred_pic50.extend(pred.view(X.size(0),).detach().cpu().numpy())

        actual_pic50 = np.array(actual_pic50)
        actual_ic50 = np.array(actual_ic50)
        pred_pic50 = np.array(pred_pic50)

        return train_loss, score(rmse=train_loss, actual_ic50=actual_ic50, pred_pic50=pred_pic50, actual_pic50=actual_pic50)
    
    def validate(self, valid_dataloader) -> float:
        self.model.eval()
        valid_loss: float = .0
        actual_pic50 = []
        actual_ic50 = []
        pred_pic50 = []
        for data in tqdm(valid_dataloader):
            X, Y, similarities = data['X'].to(self.device), data['pIC50'].to(self.device), data['Similarities'].to(self.device)

            pred = self.model(X, similarities)
            loss: torch.Tensor = torch.sqrt(self.loss(pred.squeeze(), Y.squeeze()))

            valid_loss += loss.item()

            actual_pic50.extend(data['pIC50'].numpy())
            actual_ic50.extend(data['IC50'].numpy())
            pred_pic50.extend(pred.view(X.size(0)).detach().cpu().numpy())
        
        actual_pic50 = np.array(actual_pic50)
        actual_ic50 = np.array(actual_ic50)
        pred_pic50 = np.array(pred_pic50)

        return valid_loss, score(rmse=valid_loss, actual_ic50=actual_ic50, pred_pic50=pred_pic50, actual_pic50=actual_pic50)
    
    def evaluate(self, test_dataloader):
        self.model.eval()
        submission = []
        for data in tqdm(test_dataloader):
            X, similarities = data['X'].to(self.device), data['Similarities'].to(self.device)

            outputs = self.model(X, similarities)
            submission.extend(outputs.detach().cpu().numpy())

        return submission
    
    def inference(self, submission):
        sample_df = pd.read_csv('./data/sample_submission.csv')
        sample_df['IC50_nM'] = pIC50_to_IC50(np.array(submission).reshape(-1))

        output_name = self.cfg.run_name
        submission_dir = os.path.join(self.cfg.data_dir, 'submissions')
        if not os.path.exists(submission_dir):
            logger.info(f"[Trainer] submission 저장 경로를 생성합니다: {submission_dir}")
            os.makedirs(submission_dir, exist_ok=True)
        sample_df.to_csv(f'{submission_dir}/{output_name}.csv', index=False)


class XGBTrainer:
    def __init__(self, cfg, fold:int = 0) -> None:
        self.cfg = cfg
        self.device: str = self._device()
        self.model: XGBoost = XGBoost(cfg, device=self.device)
        self.fold: int = fold
    
    def run(self, train_data, valid_data):
        self._set_runname()
        train_loss, train_score = self.train(train_data)
        valid_loss, valid_score = self.validate(valid_data)
        logger.info(f'''\n[Trainer] train loss: {train_loss:.4f} / train_score: {train_score:.4f}
                    valid loss: {valid_loss:.4f} / valid_score: {valid_score:.4f}''')
        
        if self.cfg.wandb:
            wandb.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_score': train_score,
                'valid_score': valid_score,
            })
        
        self._save_best_model()

        return valid_loss, valid_score

    def train(self, train_data) -> float:
        train_loss: float = .0
        actual_pic50 = []
        actual_ic50 = []
        pred_pic50 = []

        train_X, train_Y = np.stack(train_data['X'].values), train_data['pIC50'].values
        actual_pic50 = train_data['pIC50']
        actual_ic50 = train_data['IC50']

        logger.info(f"[XGBTrainer] model fit...")
        self.model.fit(train_X, train_Y)
        pred = self.model.predict(train_X)

        train_loss = np.sqrt(mean_squared_error(pred, train_Y))

        actual_pic50 = np.array(actual_pic50)
        actual_ic50 = np.array(actual_ic50)
        pred_pic50 = np.array(pred)

        return train_loss, score(rmse=train_loss, actual_ic50=actual_ic50, pred_pic50=pred_pic50, actual_pic50=actual_pic50)
    
    def validate(self, valid_data) -> float:
        valid_loss: float = .0
        actual_pic50 = []
        actual_ic50 = []
        pred_pic50 = []

        valid_X, valid_Y = np.stack(valid_data['X'].values), valid_data['pIC50'].values
        actual_pic50 = valid_data['pIC50']
        actual_ic50 = valid_data['IC50']

        self.model.fit(valid_X, valid_Y)
        pred = self.model.predict(valid_X)

        valid_loss = np.sqrt(mean_squared_error(pred, valid_Y))

        actual_pic50 = np.array(actual_pic50)
        actual_ic50 = np.array(actual_ic50)
        pred_pic50 = np.array(pred)

        return valid_loss, score(rmse=valid_loss, actual_ic50=actual_ic50, pred_pic50=pred_pic50, actual_pic50=actual_pic50)
    
    def evaluate(self, test_data):
        submission = []

        test_X = np.stack(test_data['X'].values)
        pred = self.model.predict(test_X)
        submission.extend(pred)

        return submission
    
    def inference(self, submission):
        sample_df = pd.read_csv('./data/sample_submission.csv')
        sample_df['IC50_nM'] = pIC50_to_IC50(np.array(submission).reshape(-1))

        output_name = self.cfg.run_name
        submission_dir = os.path.join(self.cfg.data_dir, 'submissions')
        if not os.path.exists(submission_dir):
            logger.info(f"[Trainer] submission 저장 경로를 생성합니다: {submission_dir}")
            os.makedirs(submission_dir, exist_ok=True)
        sample_df.to_csv(f'{submission_dir}/{output_name}.csv', index=False)

    def _set_runname(self):
        self.run_name = self.cfg.run_name

    def _save_best_model(self):
        logger.info(f"[Trainer] best model을 저장합니다.")
        model_filename: str = f'{os.path.join(self.cfg.model_dir, self.run_name)}_{self.fold}.json'
        if not os.path.exists(self.cfg.model_dir):
            logger.info(f"[Trainer] model 저장 경로를 생성합니다: {self.cfg.model_dir}")
            os.makedirs(self.cfg.model_dir, exist_ok=True)
        self.model.save_model(model_filename)

    def load_best_model(self):
        logger.info(f"[Trainer] best model을 불러옵니다.")
        model_filename: str = f'{os.path.join(self.cfg.model_dir, self.run_name)}_{self.fold}.json'
        if not os.path.exists(model_filename):
            logger.warning(f"[Trainer] 해당 파일이 존재하지 않습니다: {model_filename}")
        self.model.load_model(model_filename)

    def _device(self):
        if not torch.cuda.is_available():
            logger.warning(f"GPU 사용이 불가합니다. CPU로 강제 수행됩니다.")
            return 'cpu'
        if self.cfg.device.lower() not in ('cpu', 'cuda'):
            logger.warning(f"비정상적인 device 설정값 입니다. device 설정을 확인해주세요. CPU로 강제 수행됩니다.: {self.cfg.device}")
            return 'cpu'
        return self.cfg.device.lower()
