import torch.nn as nn

class SimpleImageRegressor(nn.Module):
    def __init__(self, embedding_size):
        super(SimpleImageRegressor, self).__init__()
        self.fc = nn.Linear(embedding_size, 1)  # 간단한 선형 회귀 모델

    def forward(self, x):
        return self.fc(x)

class SimpleDNN(nn.Module):
    def __init__(self, input_dim: int, emb_dims: list[int], dropout_rate: float=.5):
        super(SimpleDNN, self).__init__()
        self.input_dim: int = input_dim
        self.emb_dims: list[int] = [input_dim] + emb_dims
        self.dropout_rate: float = dropout_rate
        self.layers: nn.Module = self._layers()

    def _layers(self):
        layers = []
        for i in range(len(self.emb_dims) - 1):
            layers.append(nn.LayerNorm(self.emb_dims[i]))
            layers.append(nn.Linear(self.emb_dims[i], self.emb_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(self.emb_dims[-1], 1)) # FC layer

        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
