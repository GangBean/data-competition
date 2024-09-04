import torch.nn as nn

class SimpleImageRegressor(nn.Module):
    def __init__(self, embedding_size):
        super(SimpleImageRegressor, self).__init__()
        self.fc = nn.Linear(embedding_size, 1)  # 간단한 회귀 모델

    def forward(self, x):
        return self.fc(x)
