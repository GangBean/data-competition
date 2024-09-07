import torch
import torch.nn as nn

from loguru import logger

class SimpleImageRegressor(nn.Module):
    def __init__(self, embedding_size):
        super(SimpleImageRegressor, self).__init__()
        self.fc = nn.Linear(embedding_size, 1)  # 간단한 선형 회귀 모델

    def forward(self, x):
        return self.fc(x)

class SimpleDNN(nn.Module):
    def __init__(self, input_dim: int, layer_dims: list[int], embed_dim: int, dropout_rate: float=.5):
        super(SimpleDNN, self).__init__()
        self.input_dim: int = input_dim
        self.layer_dims: list[int] = [input_dim * embed_dim] + layer_dims
        self.embed_dim: int = embed_dim
        self.dropout_rate: float = dropout_rate
        self.layers: nn.Module = self._layers()
        self.embedding: nn.Module = CountMorganEmbedding(self.embed_dim)

    def _layers(self):
        layers = []
        for i in range(len(self.layer_dims) - 1):
            layers.append(nn.LayerNorm(self.layer_dims[i]))
            layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(self.layer_dims[-1], 1)) # FC layer

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self._transform(x)
        # logger.info(f"before embedding: {x.size()}")
        x = self.embedding(x)
        # logger.info(f"after embedding: {x.size()}")
        x = x.view(x.size(0), -1)
        # logger.info(f"after view: {x.size()}")
        return self.layers(x)
    
    def _transform(self, x):
        # logger.info(f"transform input: {x.size()}")
        batch_size, indice_size = x.size()
        indices = torch.arange(indice_size) + 1
        mask = torch.zeros_like(x).int()
        mask[x != 0] = 1
        mask = mask.view(batch_size, -1)
        output = mask * indices.to(mask.device)
        return output

class CountMorganEmbedding(nn.Module):
    def __init__(self, embed_dim:int, bit_size:int = 13_279, radius_size:int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.bit_size = bit_size
        self.radius_size = radius_size
        self.embedding = nn.Embedding(self.bit_size * self.radius_size + 1, self.embed_dim, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)