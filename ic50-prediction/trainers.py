import os

import torch
import torch.nn as nn
from torch.optim.adam import Optimizer, Adam

from tqdm import tqdm
from loguru import logger

from datetime import datetime
from models import SimpleImageRegressor
import pytz

class Trainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = self._device()

        self.model: nn.Module = self._model().to(self.device)
        self.optimizer: Optimizer = Adam(self.model.parameters(), lr=self.cfg.lr)
        self.loss = self._loss()

    def _set_runname(self):
        name: str = f"{self.cfg.model_name}_{datetime.now(pytz.timezone('Asia/Seoul'))}"
        self.run_name = name

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
        else:
            raise ValueError(f"해당하는 모델이 존재하지 않습니다: {self.cfg.model_name}")
        
    def _save_best_model(self):
        logger.info(f"[Trainer] best model을 저장합니다.")
        model_filename: str = f'{os.path.join(self.cfg.model_dir, self.run_name)}.pt'
        if not os.path.exists(self.cfg.model_dir):
            logger.info(f"[Trainer] model 저장 경로를 생성합니다: {self.cfg.model_dir}")
            os.makedirs(self.cfg.model_dir, exist_ok=True)
        torch.save(self.model.cpu().state_dict(), model_filename)

    def load_best_model(self):
        logger.warning(f"[Trainer] best model을 불러옵니다.")
        model_filename: str = f'{os.path.join(self.cfg.model_dir, self.run_name)}.pt'
        if not os.path.exists(model_filename):
            logger.warning(f"[Trainer] 해당 파일이 존재하지 않습니다: {model_filename}")
        self.model.load_state_dict(torch.load(model_filename, weights_only=True))
        self.model = self.model.to(self.device)

    def run(self, train_dataloader, valid_dataloader):
        self._set_runname()
        best_loss = .0
        for epoch in range(self.cfg.epoch):
            train_loss: float = self.train(train_dataloader)
            valid_loss: float = self.validate(valid_dataloader)
            logger.info(f"epoch: {epoch+1}/{self.cfg.epoch} >> train loss: {train_loss:.4f}")
            logger.info(f"valid loss: {valid_loss:.4f}")

            if best_loss == .0 or valid_loss < best_loss:
                best_loss = valid_loss
                self._save_best_model()

    def train(self, train_dataloader) -> float:
        self.model.train()
        train_loss: float = .0
        for data in tqdm(train_dataloader):
            X, Y = data['X'].to(self.device), data['Y'].to(self.device)

            pred = self.model(X)
            loss: torch.Tensor = self.loss(pred.squeeze(), Y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss
    
    def validate(self, valid_dataloader) -> float:
        self.model.eval()
        valid_loss: float = .0
        for data in tqdm(valid_dataloader):
            X, Y = data['X'].to(self.device), data['Y'].to(self.device)

            pred = self.model(X)
            loss: torch.Tensor = self.loss(pred.squeeze(), Y)

            valid_loss += loss.item()

        return valid_loss
    
    def evaluate(self, test_dataloader):
        self.model.eval()
        submission = []
        for data in tqdm(test_dataloader):
            images = data['X']
            outputs = self.model(images)
            submission.extend(outputs.detach().numpy())

        return submission
